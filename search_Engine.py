'''
This file is the search engine for the search_frontend.py file. It is responsible for
the backend of the search engine and it's logic. the functions will be called by the frontend.
'''
import gzip
import io
import pdb
import pickle
import re
import tempfile
from collections import Counter

import gensim
import numpy as np
import inverted_index_gcp
from pyspark.sql import SparkSession
import google.cloud.storage as storage


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

        # load index from gcp bucket
        index = self.load_pkl_from_bucket(bucket_name, "postings_gcp/index.pkl")

        #### related to text indexing ####
        # Load the df pickle file and convert it into a Python object. This is a dictionary of the form {term: df}
        self.index.df = index.df
        # load docs_len from pickle. this is a dictionary of the form {doc_id: doc_len}
        self.index.doc_len = self.load_pkl_from_bucket(bucket_name, "text_index/docs_len.pkl")
        # load idf from pickle. this is a dictionary of the form {term: idf}
        self.index.idf = self.load_pkl_from_bucket(bucket_name, "text_index/idf.pkl")
        # load avg_doc_len from pickle. this is a float representing the average document length
        self.index.avg_doc_len = self.load_pkl_from_bucket(bucket_name, "text_index/avg_doc_len.pkl")
        # load doc vec sqr from pickle. this is a dictionary of the form {doc_id: sqrt_sum_tf}
        self.index.doc_vector_sqr = self.load_pkl_from_bucket(bucket_name, "text_index/doc_vec_sqr.pkl")
        # load posting locs from bucket.
        self.index.posting_locs = index.posting_locs

        # load page rank from csv. this is a dataframe of the form {doc_id: page_rank}
        # self.index.pr = self.load_df_from_bucket_to_pc(bucket_name, "text_index/page_rank/part-00000-7a6aa0cc-aeb2-4b4d-9e04-cd1c22588e11-c000.csv.gz")

        #### related to titles indexing ####
        # load tf_idf of titles from csv. this is a dataframe of the following form {term: [(doc_id, tf_idf), ...]}
        self.index.tf_idf_title = self.load_df_from_bucket_to_pc(bucket_name,"title_index/tf_idf_compressed/part-00000-b775031b-bf6d-4dda-9feb-cea9ccee9f1d-c000.csv.gz")
        # load doc_sqr_root of title from pkl: dictionary {doc_id: sqrt_sum_tf}, meaning if doc_id has 3 words with tf 1, 2, 3, the value will be sqrt(1^2 + 2^2 + 3^2)
        self.index.doc_vector_sqr_title = self.load_pkl_from_bucket(bucket_name, "title_index/doc_vec_sqr.pkl")

        #### gensim word2vec model ####
        # self.word2vec = self.load_word2vec_from_bucket(bucket_name, "GoogleNews-vectors-negative300.bin.gz")
        # right now loading locally
        self.word2vec = gensim.models.KeyedVectors.load('C:\\Users\\Lior\\Desktop\\Semester 5\\IR\\project related\\word2vec.model')


    ''' Load data from GCP bucket '''
    def load_pkl_from_bucket(self, bucket_name, file_path):
        """
        Load a pkl file from a Google Cloud Storage bucket.

        Args:
        - bucket_name: Name of the Google Cloud Storage bucket.
        - file_path: Path to the file within the bucket.

        Returns:
        - The contents of the file.
        """
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_path)
        contents = blob.download_as_string()
        return pickle.loads(contents)

    def load_df_from_bucket_to_pc(self, bucket_name, file_path):
        """
        Load a DataFrame from a CSV file stored in a Google Cloud Storage bucket.

        Args:
        - bucket_name: Name of the Google Cloud Storage bucket.
        - file_path: Path to the CSV file within the bucket.

        Returns:
        - A DataFrame containing the contents of the CSV file.
        """
        # Get the GCS bucket and blob
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_path)
        # Download the file contents as a string
        file_contents = blob.download_as_string()
        with gzip.GzipFile(fileobj=io.BytesIO(file_contents)) as gz:
            file_contents = gz.read()
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_contents)
            temp_file_path = temp_file.name
        # Read the CSV data into a Spark DataFrame
        df = spark.read.csv(temp_file_path, header=True, inferSchema=True)
        return df

    # function to load a DataFrame from a CSV file in a GCP bucket
    def load_df_from_bucket_to_gcp(self, bucket_name, file_path):
        return spark.read.format("csv").option("header", "true") .option("inferSchema", "true") .load("gs://{}/{}".format(bucket_name, file_path))

    # load word2vec model from bucket
    def load_word2vec_from_bucket(self, bucket_name, file_path):
        """
        Load a word2vec model from a Google Cloud Storage bucket.

        Args:
        - bucket_name: Name of the Google Cloud Storage bucket.
        - file_path: Path to the file within the bucket.

        Returns:
        - The word2vec model.
        """
        # set path
        path = f'gs://{bucket_name}/{file_path}'
        # return model
        return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    # function we used to create the word2vec model
    def create_word2vec(self):
        ''' This function creates the word2vec model from the GoogleNews-vectors-negative300.bin.gz file and saves it to disk.'''
        # Load Google's pre-trained Word2Vec model.
        path = 'GoogleNews-vectors-negative300.bin.gz'
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        # save the model to disk
        model.save('word2vec.model')


    ''' DataFrame processing function, for title indexing '''
    def filterTFIDF(self, df, query_terms):
        ''' This function filters the tf-idf data from a DataFrame and returns a dictionary.
        Parameters:
        -----------
        df: DataFrame
            A DataFrame of the form {term: [(doc_id, tf-idf), ...]}.
        query_terms: list
            A list of the distinct query words.
        Returns:
        --------
        all_data: dict
            A dictionary of the form {term: [(doc_id, tf-idf), ...]} representing the tf-idf of the documents with the query.
        '''
        # filter df by query terms
        filtered_df = df.filter(df[0].isin(query_terms))
        # Create an empty dictionary to store the data
        all_data = {}
        # Collect the DataFrame into the driver node as a list of rows
        rows = filtered_df.collect()
        # Iterate over the rows of the DataFrame
        for row in rows:
            # Extract relevant data from the DataFrame row
            key = str(row[0])  # term
            value1 = str(row[1])  # doc id
            value2 = float(row[2])  # tf-idf
            # Check if the key exists in the dictionary
            if key not in all_data:
                all_data[key] = []
            # Append the values to the list associated with the key
            all_data[key].append((value1, value2))
        return all_data

    def filterKeyValueDF(self, df, docs):
        ''' This function loads key and value from DataFrame and returns a dictionary.
        Parameters:
        -----------
        df: DataFrame
            A DataFrame of the form {doc_id: page_rank} or any other {key:value}.
        docs: list
            A list of the distinct keys, in this case, the doc_ids.
        Returns:
        --------
        df_dict: dict
            A dictionary of the form {key: value}, where first column is key and second column is value.
        '''
        # convert all keys to int
        docs = [int(doc) for doc in docs]
        # filter df by query terms
        filtered_df = df.filter(df[0].isin(docs))
        # Create an empty dictionary to store the data
        df_dict = {}
        # Collect the DataFrame into the driver node as a list of rows
        rows = filtered_df.collect()
        # Iterate over the rows of the DataFrame
        for row in rows:
            # Extract relevant data from the DataFrame row
            key = str(row[0])  #  first column is the key
            value = float(row[1])  #  second column contains value
            # Add the key-value pair to the dictionary
            df_dict[key] = value
        # sort the dictionary by value in descending order
        df_dict = dict(sorted(df_dict.items(), key=lambda item: item[1], reverse=True))
        return df_dict


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
            An integer representing the number of words to expand the query with.
        Returns:
        --------
        list
            A list of the distinct query words after expanding the query.
        '''
        # create a list to store the expanded query
        expanded_query = tokQuery.copy()
        # iterate over the query words
        for word in tokQuery:
            # check if word is stopword or not in vocab, if so, skip it
            if inverted_index_gcp.InvertedIndex.isStopWord(word) or not word in self.word2vec.key_to_index:
                continue
            # get the most similar words to the word in the query
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
            A list of the words in the text.
        '''
        # regular expression to find words
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
        # return a list of words in the text in lowercase
        return [token.group() for token in RE_WORD.finditer(text.lower())]

    # function to preprocess the query
    def processQuery(self, query):
        ''' This function tokenizes the query using the staff-provided tokenizer from
            Assignment 3 (GCP part) to do the tokenization and remove stopwords.
        Parameters:
        -----------
        query: list
            A list of the query words.
        Returns:
        --------
        tokens: list
            A list of the distinct query words after removing stopwords and stemming.
        '''
        # remove stopwords and stem the query
        tokens = inverted_index_gcp.InvertedIndex.filter_tokens(query, True)
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

    """Process the doc from index and return vectors for cosine similarity calculation and ranking
    This function is for title indexing. We tried to use tf-idf for the cosine similarity dot product calculation,
    instead of using the tf of the query and the tf of the doc.
    """

    ''' Process the doc from index and return vectors for cosine similarity calculation and ranking
    This function is for text indexing and title indexing. We tried to use tf-idf for the cosine similarity dot product calculation of title,
    instead of using the tf of the query and the tf of the doc. We used regular cos sim vectors for the text indexing.
    '''
    def processDocsCosSim(self, query_terms, isTitle=False):
        # create empty dict for vectors of query, keys are doc_ids and values are lists of tf-idf vectors
        docs_vectors = {}
        # create base vector for each doc_id, with 0s for each term in the query
        base_vector = [0 for term in query_terms]
        # create index for the base vector
        index = 0
        # if title, get tf-idf dict from index
        if isTitle:
            tf_idf_dict = self.filterTFIDF(self.index.tf_idf_title, query_terms)
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
        ''' This function returns up to a 100 of your best search results for the query.
        Parameters:
        -----------
        proc_query: Counter
            A dictionary of the form {term: count} representing the query.
        Returns:
        --------
        list
            A list of the top 100 results of the search.
        '''
        # get the keys of the query vector, meaning the distinct query words
        query_terms = list(proc_query.keys())
        # create empty dict for docs tf-idf scores
        docs_tfidf = {}
        # iterate over the terms in the query
        for term in query_terms:
            # get posting list iterator for term
            posting_list_iter = self.index.read_a_posting_list("postings_gcp", term, bucket_name)
            # iterate over the posting list
            for doc_id, tf in posting_list_iter:
                # calculate tf(i,j)
                tfij = tf / self.index.doc_len[doc_id]
                # get idf
                idf = self.index.idf[term]
                # calculate tf-idf
                tfidf = tfij * idf
                # if the doc_id is not in the dict, add it
                if doc_id not in docs_tfidf:
                    docs_tfidf[doc_id] = 0
                # add the tf-idf to the dict
                docs_tfidf[doc_id] += tfidf
        # sort the dict by value in descending order of tf-idf
        sorted_sim_dict = dict(sorted(docs_tfidf.items(), key=lambda item: item[1], reverse=True))
        # return the dict
        return sorted_sim_dict

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
            DP_sim_docs[doc] = dot_product
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
            # get idf for term and add 1 to avoid division by zero
            idf = self.index.idf[term] + 1
            # get posting list iterator for term
            posting_list_iter = self.index.read_a_posting_list("postings_gcp", term, bucket_name)
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
        # sort the dict by value in descending order of BM25
        sorted_sim_dict = dict(sorted(BM25_docs.items(), key=lambda item: item[1], reverse=True))
        return sorted_sim_dict

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
        # sort the similarity dict by value in descending order
        sorted_sim_dict = dict(sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True))
        # return ids of docs sorted by cosine similarity
        return sorted_sim_dict

    def search(self, query):
        # tokenize the query
        tokenized = self.tokenize(query)

        # if length of tokens is less than 5, return the query up to 5 words
        if len(tokenized) == 1:
            tokenized = self.expandQuery(tokenized, 4)
        else:
            if len(tokenized) == 2:
                tokenized = self.expandQuery(tokenized, 2)
        queryTok = self.processQuery(tokenized)
        # vectorize the expanded query, returning a counter of the expanded query tokens and their counts
        query_counter = self.vectorizeQuery(queryTok)
        # search by cosine similarity on titles
        cosSimTitle = self.searchByCosineSimilarity(query_counter, True)
        cosSimText = self.searchByCosineSimilarity(query_counter, False)
        # create a list of dictionaries
        dicts = [cosSimTitle, cosSimText]
        # create percentage list
        p = [0.3, 0.6]
        # try to combine the results of the two searches
        return self.combineReulsts(dicts, p)


    """ Helper function """

    def combineReulsts(self, dicts, p, res=100):
        '''
        This function combines the results of the two searches.
        Parameters:
        -----------
        dicts: list
            A list of the dictionaries representing the results of the searches to combine.
        p: list
            A list of the percentages to use for the weighted sum of the results.
        res: int
            An integer representing the number of results to return.
        Returns:
        --------
        list
            A list of the top 100 results of the combined search.
        '''
        # combine the results of the two searches
        summed_dict = {}
        # create index
        index = 0
        # iterate over the dictionaries
        for d in dicts:
            # iterate over the items in the dictionary
            for k, v in d.items():
                # if the key is not in the dict, add it
                if k not in summed_dict:
                    summed_dict[k] = 0
                # add the value to the dict
                summed_dict[k] += v * p[index]
            # increment the index
            index += 1
        # sort the dict by value in descending order
        sorted_dict = dict(sorted(summed_dict.items(), key=lambda item: item[1], reverse=True))
        # return the top 100 results
        return list(sorted_dict.items())[0:res]

    # function for adding page rank to score
    def addPageRank(self, summed_dict):
        '''
        This function adds the page rank to the score of the documents.
        Parameters:
        -----------
        summed_dict: dict
            A dictionary of the form {doc_id: score} representing the score of the documents.
        Returns:
        --------
        dict
            A dictionary of the form {doc_id: score} representing the score of the documents with the page rank added to the score.
        '''
        # get page rank for each doc in the dict and append it to the value of the dick
        pr_dict = self.filterKeyValueDF(self.index.pr, summed_dict.keys())
        # add each value to it's matching key
        for k, v in summed_dict.items():
            summed_dict[k] += 0.1 * pr_dict[k]
        # return result
        return summed_dict
