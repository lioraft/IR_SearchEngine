'''
This file is the search engine for the search_frontend.py file. It is responsible for
the backend of the search engine and it's logic. the functions will be called by the frontend.
'''
import gzip
import io
import pickle
import re
from collections import Counter
import gensim
import numpy as np
import inverted_index_gcp
from pyspark.sql import SparkSession
import google.cloud.storage as storage
import pandas as pd
import pdb


# create a spark session
spark = SparkSession.builder.appName("search").getOrCreate()
bucket_name = "ass3_new"
# Set the project ID
project_id = "assignment3-413720"
# Initialize Storage client with project ID
storage_client = storage.Client(project=project_id)

class searchEngine:
    ''' This class is the search engine for the search_frontend.py file.'''
    # constructor that initializes the index and loads the data of the index
    def __init__(self):
        # create the index, right now it's empty
        self.index = inverted_index_gcp.InvertedIndex()
        # load index from gcp bucket. (same index from assignment 3)
        index = self.load_pkl_from_bucket(bucket_name, "final_index/index.pkl")

        #### related to text indexing ####
        # Load the df pickle file and convert it into a Python object. This is a dictionary of the form {term: df}
        self.index.df = index.df
        # load docs_len from pickle. this is a dictionary of the form {doc_id: doc_len}
        self.index.doc_len = self.load_pkl_from_bucket(bucket_name, "final_index/docs_len.pkl")
        # load idf from pickle. this is a dictionary of the form {term: idf}
        self.index.idf = self.load_pkl_from_bucket(bucket_name, "final_index/idf.pkl")
        # load avg_doc_len from pickle. this is a float representing the average document length
        self.index.avg_doc_len = self.load_pkl_from_bucket(bucket_name, "final_index/avg_doc_len.pkl")
        # load doc vec sqr from pickle. this is a dictionary of the form {doc_id: sqrt_sum_tf}
        self.index.doc_vector_sqr = self.load_pkl_from_bucket(bucket_name, "final_index/doc_vec_sqr.pkl")
        # load posting locations.
        self.index.posting_locs = index.posting_locs

        # load page rank from csv. this is a dataframe that will be converted to a dictionary in the form {doc_id: page_rank}
        self.index.pr = self.load_pr_from_bucket(bucket_name, "final_index/page_rank.csv.gz")

        #### related to titles indexing ####
        # load tf_idf of titles from csv. this is a dict of the following form {term: [(doc_id, tf_idf), ...]}, based on the titles
        self.index.tf_idf_title = self.load_tfidf_from_bucket(bucket_name,"final_index/tfidf_titles.csv.gz")
        # load doc_sqr_root of title from pkl: dictionary {doc_id: sqrt_sum_tf}, meaning if doc_id has 3 words with tf 1, 2, 3, the value will be sqrt(1^2 + 2^2 + 3^2)
        self.index.doc_vector_sqr_title = self.load_pkl_from_bucket(bucket_name, "final_index/doc_vec_sqr_titles.pkl")
        # dictionary of the form {doc_id: title} for the docs in the corpus
        self.id_to_title = self.load_pkl_from_bucket(bucket_name, "final_index/id_title_dict.pkl")

        #### gensim word2vec model ####
        self.word2vec = self.load_word2vec_from_bucket(bucket_name, "final_index/GoogleNews-vectors-negative300.bin.gz")
        # local load - we used for pycharm, not for gcp instance
        #self.word2vec = gensim.models.KeyedVectors.load('C:\\Users\\Lior\\Desktop\\Semester 5\\IR\\project related\\word2vec.model')

        #### saving results for current search ###
        self.title_results = {}
        self.text_results = {}
        self.pr_results = {}
        self.final_results = {}


    ''' Load data from GCP bucket '''
    def load_pkl_from_bucket(self, bucket_name, file_path):
        """
        Load a pkl file from a Google Cloud Storage bucket.

        Args:
        - bucket_name: Name of the Google Cloud Storage bucket.
        - file_path: Path to the pickle file within the bucket.

        Returns:
        - The contents of the file as a Python object.
        """
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_path)
        contents = blob.download_as_string()
        return pickle.loads(contents)

    def load_pr_from_bucket(self, bucket_name, file_path):
        """
        Load a DataFrame from a CSV file stored in a Google Cloud Storage bucket,
        and convert it to a dictionary of page rank.

        Args:
        - bucket_name: Name of the Google Cloud Storage bucket.
        - file_path: Path to the CSV file within the bucket.

        Returns:
        - A dictionary of the form {doc_id: page_rank}.
        """
        # get df from bucket
        df = self.load_df_from_bucket_to_gcp(bucket_name, file_path)
        # convert the DataFrame to a dictionary and return it
        dict_from_df = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        return dict_from_df

    # function to load a DataFrame from a CSV file in a GCP bucket
    def load_df_from_bucket_to_gcp(self, bucket_name, file_path):
        '''
        This function loads a DataFrame from a CSV file stored in a Google Cloud Storage bucket and returns
        it as a pandas DataFrame.
        Parameters:
        -----------
        bucket_name: str
            A string representing the name of the bucket.
        file_path: str
            A string representing the path to the file within the bucket.
        Returns:
        --------
        df: pandas DataFrame
        '''
        # get the bucket
        bucket = storage_client.get_bucket(bucket_name)
        # get the blob (file)
        blob = bucket.blob(file_path)
        # download the file contents as a string
        compressed_content = blob.download_as_string()
        # decompress the content
        csv_content = gzip.decompress(compressed_content).decode('utf-8')
        # convert CSV content to pandas DataFrame
        df = pd.read_csv(io.StringIO(csv_content))
        return df

    # load word2vec model from bucket
    def load_word2vec_from_bucket(self, bucket_name, file_path):
        """
        Load a word2vec model from a Google Cloud Storage bucket.

        Args:
        - bucket_name: Name of the Google Cloud Storage bucket.
        - file_path: Path to the compressed model file within the bucket.

        Returns:
        - The word2vec model.
        """
        # set path
        path = f'gs://{bucket_name}/{file_path}'
        # return model
        return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    def load_tfidf_from_bucket(self, bucket_name, file_path):
        ''' This function gets the tf-idf data from a DataFrame and returns a dictionary.
        Parameters:
        -----------
        bucket_name: str
            A string representing the name of the bucket.
        file_path: str
            A string representing the path to the file within the bucket.
        Returns:
        --------
        res: dict
            A dictionary of the form {term: [(doc_id, tf-idf), ...]} representing the tf-idf of titles.
        '''
        # get df from bucket
        df = self.load_df_from_bucket_to_gcp(bucket_name, file_path)
        # Create an empty dictionary to store the data
        result_dict = {}
        # Iterate over the rows of the DataFrame
        for index, row in df.iterrows():
            # Extract relevant data from the DataFrame row
            key = str(row.iloc[0])  # term
            value1 = str(row.iloc[1])  # doc id
            value2 = float(row.iloc[2])  # tf-idf
            # Check if the key already exists in the dictionary
            if key in result_dict:
                # If the key exists, append the tuple to the existing list of tuples
                result_dict[key].append((value1, value2))
            else:
                # If the key does not exist, create a new list with the tuple and assign it to the key
                result_dict[key] = [(value1, value2)]
        return result_dict


    """Process Query before search"""
    # query expansion function
    def expandQuery(self, tokQuery, numWord=3):
        '''
        This function expands the query using word2vec model.
        Parameters:
        -----------
        tokQuery: list
            A list of the query words.
        numWord: int
            An integer representing the number of words to expand each one of the tokens of the query with,
            default is 3.
        Returns:
        --------
        list
            A list of the query words after expanding the query.
        '''
        # create a list to store the expanded query
        expanded_query = tokQuery.copy()
        # iterate over the query words
        for word in tokQuery:
            # check if word is stopword or not in vocab, if so, skip it
            if inverted_index_gcp.InvertedIndex.isStopWord(word) or not word in self.word2vec.key_to_index:
                continue
            # get the top n similar words to the word in the query
            similar_words = self.word2vec.most_similar(word, topn=numWord)
            # iterate over the similar words
            for sim_word in similar_words:
                # add the similar word to the expanded query
                expanded_query.append(sim_word[0])
        # return the expanded query
        return expanded_query


    # tokenizing function
    def tokenize(self, text):
        '''
        This function tokenizes the text using the staff-provided tokenizer from Assignment 3 (GCP part).
        Parameters:
        -----------
        text: str
            The text string.
        Returns:
        --------
        list
            A list of the words in the text, in lowercase and without punctuation.
        '''
        # regular expression to find words
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
        # return a list of words in the text in lowercase
        return [token.group() for token in RE_WORD.finditer(text.lower())]

    # function to preprocess the query
    def processQuery(self, query, isStem=False):
        ''' This function tokenizes the query using the staff-provided tokenizer from
            Assignment 3 (GCP part) to do the tokenization and remove stopwords.
        Parameters:
        -----------
        query: list
            A list of the query words.
        isStem: bool
            A boolean representing whether to stem the query or not.
        Returns:
        --------
        tokens: list
            A list of the distinct query words after removing stopwords (and stemming if isStem is True).
        '''
        # remove stopwords and stem the query
        tokens = inverted_index_gcp.InvertedIndex.filter_tokens(query, isStem)
        # return the tokens of the query
        return tokens

    '''Vectorize Query and Documents for cosine similarity calculation and ranking'''
    def vectorizeQuery(self, query):
        ''' This function vectorizes the query and the documents in the index.
        Parameters:
        -----------
        query: list
            A list of the query words.
        Returns:
        --------
        Counter(query): Counter
            A Counter object of the form {term: count} representing the query.
        '''
        return Counter(query)

    ''' Process the doc from index and return vectors for cosine similarity calculation and ranking
    This function is for text indexing and title indexing. We tried to use tf-idf for the cosine similarity dot product calculation of title,
    instead of using the tf of the doc. We used regular cos sim vectors for the text indexing.
    '''
    def processDocsCosSim(self, query_terms, isTitle=False):
        '''
        This function processes the documents in the index and returns the vectors of the documents for cosine similarity calculation and ranking.
        Parameters:
        -----------
        query_terms: list
            A list of the query words.
        isTitle: bool
            A boolean representing whether it's the title or the text.
        Returns:
        --------
        docs_vectors: dict
            A dictionary of the form {doc_id: [count1, count2, ...]} representing the vectors of the documents.
        '''
        # create empty dict for vectors of query, keys are doc_ids and values are lists of tf-idf vectors
        docs_vectors = {}
        # create base vector for each doc_id, with 0s for each term in the query
        base_vector = [0 for term in query_terms]
        # create index for the base vector
        index = 0
        # if title, get tf-idf dict from index
        if isTitle:
            tf_idf_dict = self.index.tf_idf_title
        # iterate over the terms in the query
        for term in query_terms:
            # get posting list iterator for term
            if isTitle:
                # check if the term is in the tf-idf dict
                if term in tf_idf_dict:
                    posting_list_iter = tf_idf_dict[term]
                else:
                    posting_list_iter = []
            else:
                posting_list_iter = self.index.read_a_posting_list("postings_gcp", term, bucket_name)
                # if the term is not in the index, skip it
                if posting_list_iter is None:
                    continue
            # iterate over the posting list
            for doc_id, tf in posting_list_iter:
                # if the doc_id is not in the dict, add it with the base vector
                if doc_id not in docs_vectors:
                    docs_vectors[doc_id] = base_vector.copy()
                # add the tf to the list of vectors for the doc
                docs_vectors[doc_id][index] = tf
            # increment the index
            index += 1
        # return the vectors of the docs
        return docs_vectors

    """Similarity and Ranking"""
    def TFIDF(self, proc_query):
        ''' This function returns up to a 100 of your best search results for the query,
        based on the tf-idf of the documents with the query.
        Parameters:
        -----------
        proc_query: Counter
            A dictionary of the form {term: count} representing the query.
        Returns:
        --------
        list
            A list of the top 100 results of the search based on the tf-idf of the documents with the query.
        '''
        # get the keys of the query vector, meaning the distinct query words
        query_terms = list(proc_query.keys())
        # create empty dict for docs tf-idf scores
        docs_tfidf = {}
        # iterate over the terms in the query
        for term in query_terms:
            # get posting list iterator for term
            posting_list_iter = self.index.read_a_posting_list("postings_gcp", term, bucket_name)
            # if the term is not in the index, skip it
            if posting_list_iter is None:
                continue
            # stem the term in order to get idf
            stem_term = self.index.stemmer.stem(term)
            # check if term is in idf dict, if not, skip it
            if stem_term not in self.index.idf:
                continue
            # get idf for term and add 1 to avoid division by zero
            idf = self.index.idf[stem_term] + 1
            # iterate over the posting list
            for doc_id, tf in posting_list_iter:
                # calculate tf(i,j)
                tfij = tf / self.index.doc_len[doc_id]
                # calculate tf-idf
                tfidf = tfij * idf
                # if the doc_id is not in the dict, add it
                if doc_id not in docs_tfidf:
                    docs_tfidf[doc_id] = 0
                # add the tf-idf to the dict
                docs_tfidf[doc_id] += tfidf
        # sort the dict by value in descending order of tf-idf and save the top 100 results
        self.text_results = dict(sorted(docs_tfidf.items(), key=lambda item: item[1], reverse=True)[:100])

    def cosineSimilarity(self, query_vector, docs_vector, isTitle=False):
        '''
        This function calculates the cosine similarity between the query and the documents.
        Parameters:
        -----------
        query_vector: list
            A list of the form [count1, count2, ...] representing the query.
        docs_vector: dict
            A dictionary of the form {doc_id: [count1, count2, ...]} representing the documents.
        isTitle: bool
            A boolean representing whether it's similarity between titles or between text.
        Returns:
        --------
        cosine_sim_docs: dict
            A dictionary of the form {doc_id: cosine_similarity} representing the cosine similarity of the documents with the query.
        '''
        # get the dot product similarity of the query and the documents
        cosine_sim_docs = self.dotProduct_sim(query_vector, docs_vector)

        for doc_id in cosine_sim_docs.keys():
            if isTitle:
                # divide by the root of sum of squares of tf in each document, meaning by tf(i,j) i is word and j is doc
                cosine_sim_docs[doc_id] = cosine_sim_docs[doc_id] / (self.index.doc_vector_sqr_title[int(doc_id)] * np.sqrt(np.sum(np.array(query_vector)**2)))
            else:
                # divide by the root of sum of squares of tf in each document, meaning by tf(i,j) i is word and j is doc
                cosine_sim_docs[doc_id] = cosine_sim_docs[doc_id] / (self.index.doc_vector_sqr[int(doc_id)] * np.sqrt(np.sum(np.array(query_vector)**2)))
        return cosine_sim_docs

    # dot product similarity, will be used in cosine similarity
    def dotProduct_sim(self, query_vector, docs_vector):
        '''
        This function calculates the dot product similarity between the query and the documents.
        Parameters:
        -----------
        query_vector: list
            A list of the form [count1, count2, ...] representing the query.
        docs_vector: dict
            A dictionary of the form {doc_id: [count1, count2, ...]} representing the documents.
        Returns:
        --------
        DP_sim_docs: dict
            A dictionary of the form {doc_id: dot_product} representing the dot product similarity of the documents with the query.
        '''
        # create a dictionary to store the cosine similarity of each doc
        DP_sim_docs = {}
        # iterate over the docs
        for doc, vector in docs_vector.items():
            # create a variable to store the dot product
            dot_product = 0
            # iterate over the elements of the doc vector
            for i in range(len(vector)):
                # calculate the dot product of the doc vector and the query vector and add it to the dot_product
                dot_product += vector[i] * query_vector[i]
            # add the dot product to the dictionary of dot product similarities
            DP_sim_docs[str(doc)] = dot_product
        return DP_sim_docs

    def BM25(self, proc_query, b=0.75, k1=1.2, k3=1):
        ''' This function returns up to a 100 of your best search results for the query.
        Parameters:
        -----------
        proc_query: dict
            A dictionary of the form {term: count} representing the query.
        b: float
            A float representing the b parameter in the BM25 formula.
        k1: float
            A float representing the k1 parameter in the BM25 formula.
        '''
        # get the keys of the query vector, meaning the distinct query words
        query_terms = list(proc_query.keys())
        # create a dictionary to store the BM25 of each doc
        BM25_docs = {}
        # iterate over terms
        for term in query_terms:
            # get posting list iterator for term
            posting_list_iter = self.index.read_a_posting_list("postings_gcp", term, bucket_name)
            # if the term is not in the index, skip it
            if posting_list_iter is None:
                continue
            # stem the term in order to get idf
            stem_term = self.index.stemmer.stem(term)
            # check if term is in idf dict, if not, skip it
            if stem_term not in self.index.idf:
                continue
            # get idf for term and add 1 to avoid division by zero
            idf = self.index.idf[stem_term] + 1
            # iterate over the posting list
            for doc_id, tf in posting_list_iter:
                # calculate tf(i,j)
                tfij = (k1+1) * tf / (k1 * (1 - b + b * self.index.doc_len[doc_id] / self.index.avg_doc_len) + tf)
                # calculate tf(i,q)
                tfiq = (k3+1) * proc_query[term] / (k3 + proc_query[term])
                # calculate BM25
                BM25 = idf * tfij * tfiq
                str_doc_id = str(doc_id)
                # if the doc_id is not in the dict, add it
                if str_doc_id not in BM25_docs:
                    BM25_docs[str_doc_id] = 0
                # add the BM25 to the dict
                BM25_docs[str_doc_id] += BM25
        # sort the dict by value in descending order of BM25 and save the top 100 results
        self.text_results = dict(sorted(BM25_docs.items(), key=lambda item: item[1], reverse=True)[:100])

    """Search functions"""
    def searchByCosineSimilarity(self, proc_query, isTitle=False):
        ''' This function returns the relevant documents based on cosine similarity.
        Parameters:
        -----------
        proc_query: dict
            A dictionary of the form {term: count} representing the query.
        isTitle: bool
            A boolean representing whether it's similarity between titles or between text.
        Returns:
        --------
        sorted_sim_dict: dict
            A dictionary of the form {doc_id: cosine_similarity} representing the cosine similarity of the documents with the query.
            The dictionary is sorted by cosine similarity in descending order.
        '''
        # get the keys of the query vector, meaning the distinct query words
        query_terms = list(proc_query.keys())
        # get the vectors of the docs that match the query
        docs_vectors = self.processDocsCosSim(query_terms, isTitle)
        # convert the query vector to a list of values
        query_vector = list(proc_query.values())
        similarity_dict = self.cosineSimilarity(query_vector, docs_vectors, isTitle)
        # sort the similarity dict by value in descending order and save the top 100 results
        sorted_sim_dict = dict(sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)[:100])
        # save results based on title or text
        if isTitle:
            self.title_results = sorted_sim_dict
        else:
            self.text_results = sorted_sim_dict

    def search(self, query):
        '''
        This function returns the top 100 results of the search for the query.
        Parameters:
        -----------
        query: str
            The query string.
        What the function does:
        -----------------------
        The function tokenizes the query and expands it if it's too short.
        Then it vectorizes the expanded query and searches the title and text.
        It then combines the results of the two searches and adds the page rank to the results.
        Returns:
        --------
        list
            A list of the top 100 results of the search, containing tuples of (doc_id, title).
        '''
        # tokenize the query
        tokenized = self.tokenize(query)
        # if length of tokenized is 0, return empty list
        if len(tokenized) == 0:
            return []
        # if length of query is 1, meaning it's too short, so expand it by 3 words
        if len(tokenized) == 1:
            tokenized = self.expandQuery(tokenized, 3)
        # process the query, stemming it and removing stopwords
        queryTokTitle = self.processQuery(tokenized, True)
        queryTokText = self.processQuery(tokenized, False)
        # vectorize the expanded query, returning a counter of the expanded query tokens and their counts
        queryCountTitle = self.vectorizeQuery(queryTokTitle)
        queryCountText = self.vectorizeQuery(queryTokText)
        # search by bm25 for text
        self.BM25(queryCountText)
        # search by cosine similarity for title
        self.searchByCosineSimilarity(queryCountTitle, True)
        # create a list of dictionaries
        dicts = [self.title_results, self.text_results]
        # create percentage list
        p = [0.6, 0.2]
        # combine the results of the two searches
        self.combineReulsts(dicts, p)
        # add page rank to the results
        res = self.addPageRank(self.final_results, 0.2)
        # reset all dictionaries
        self.resetAfterSearch()
        # map each id to its title in tuples
        res = [(k, self.id_to_title[int(k)]) for k, v in res.items()]
        # return top 100 results
        return res[:100]


    """ Helper functions """

    def combineReulsts(self, dicts, p):
        '''
        This function combines the results of the two searches.
        Parameters:
        -----------
        dicts: list
            A list of the dictionaries representing the results of the searches to combine
            with current self.final_results.
        p: list
            A list of the percentages to use for the weighted sum of the results.
        Returns:
        --------
        list
            A list of the top 100 results of the combined search.
        '''
        # create index
        index = 0
        # iterate over the dictionaries
        for d in dicts:
            # iterate over the items in the dictionary
            for k, v in d.items():
                # if the key is not in the dict, add it
                if k not in self.final_results:
                    self.final_results[k] = 0
                # add the value to the dict
                self.final_results[k] += v * p[index]
            # increment the index
            index += 1
        # sort the dict by value in descending order and store in final results
        self.final_results = dict(sorted(self.final_results.items(), key=lambda item: item[1], reverse=True))

    # function for adding page rank to score
    def addPageRank(self, sum_dict, p=0.1):
        '''
        This function adds the page rank to the score of the documents and stores it in a dictionary.
        Parameters:
        -----------
        docs: dict
            A dictionary of the form {doc_id: score} representing the score of the documents.
        p: float
            A float representing the percentage to use for the weighted sum of the results.
        Returns:
        --------
        dict
            A sorted dictionary of the form {doc_id: score} representing the score of the documents with the page rank added.
        '''
        # iterate over the items in the dictionary
        for k, v in sum_dict.items():
            # if the key is not in the dict, add it
            if int(k) in self.index.pr:
                sum_dict[k] += self.index.pr[int(k)] * p
        # sort the dict by value in descending order and store in final results
        return dict(sorted(sum_dict.items(), key=lambda item: item[1], reverse=True))

    def resetAfterSearch(self):
        '''
        Reset all dictionaries created during search.
        '''
        self.final_results = {}
        self.pr_results = {}
        self.text_results = {}
        self.title_results = {}
