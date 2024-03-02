'''
This file is the search engine for the search_frontend.py file. It is responsible for
the backend of the search engine and it's logic. the functions will be called by the frontend.
'''
import csv
import os
import pickle
import re
from collections import Counter
import numpy as np
import inverted_index_gcp
from google.cloud import storage
from pyspark.sql import SparkSession
from io import BytesIO

# create a spark session
spark = SparkSession.builder.appName("search").getOrCreate()

class searchEngine:
    ''' This class is the search engine for the search_frontend.py file.'''
    # constructor that initializes the index and loads the data of the index
    def __init__(self):
        # create the index, right now it's empty
        self.index = inverted_index_gcp.InvertedIndex()
        # Load the df pickle file and convert it into a Python object. This is a dictionary of the form {term: df}
        self.index.df = self.load_pkl_from_bucket("ass3_new", "title_index/df.pkl")
        # load docs_len from pickle. this is a dictionary of the form {doc_id: doc_len}
        self.index.doc_len = self.load_pkl_from_bucket("ass3_new", "title_index/docs_len.pkl")
        # load tf_idf from csv. this is a dictionary of the following form {term: [(doc_id, tf_idf), ...]}
        self.index.tf_idf = self.load_tfidf("ass3_new", "title_index/tf_idf_compressed/")
        # load doc_sqr_root from csv: dictionary of {doc_id: sqrt_sum_tf}
        # meaning if doc_id has 3 words with tf 1, 2, 3, the value will be sqrt(1^2 + 2^2 + 3^2)
        self.index.doc_vector_sqr = self.load_doc_sqr_root("ass3_new", "title_index/doc_sqrt_sum_tf_compressed/")

    def load_pkl_from_bucket(self, bucket_name, file_path):
        """
        Load a pkl file from a Google Cloud Storage bucket.

        Args:
        - bucket_name: Name of the Google Cloud Storage bucket.
        - file_path: Path to the file within the bucket.

        Returns:
        - The contents of the file.
        """
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_path)
        contents = blob.download_as_string()
        return pickle.loads(contents)

    def load_df_from_bucket(self, bucket_name, file_path):
        """
        Load a DataFrame from a CSV file stored in a Google Cloud Storage bucket.

        Args:
        - bucket_name: Name of the Google Cloud Storage bucket.
        - file_path: Path to the CSV file within the bucket.

        Returns:
        - A DataFrame containing the contents of the CSV file.
        """
        # Create a Cloud Storage client
        storage_client = storage.Client()
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)
        # Define the blob (file) path
        blob = bucket.blob(file_path)
        # Download the CSV file as a string
        csv_string = blob.download_as_string().decode('utf-8')
        # Read the CSV string as a DataFrame
        df = spark.read.csv(spark.sparkContext.parallelize([csv_string]), header=True, inferSchema=True)
        return df


    ''' loading data from bucket to dictionary'''

    def load_tfidf(self, bucket_name, file_path):
        ''' This function loads tf_idf data from a DataFrame.
        '''
        # get df from bucket
        df = self.load_df_from_bucket(bucket_name, file_path)
        # Collect the DataFrame into the driver node as a list of rows
        rows = df.collect()
        # Create an empty dictionary to store the data
        all_data = {}
        # Iterate over the rows of the DataFrame
        for row in rows:
            # Extract relevant data from the DataFrame row
            key = row[0]  # Assuming the first column is the key
            value1 = row[1]  # Assuming the second column is the first value
            value2 = row[2]  # Assuming the third column is the second value
            # Check if the key exists in the dictionary
            if key not in all_data:
                all_data[key] = []
            # Append the values to the list associated with the key
            all_data[key].append((value1, float(value2)))
        return all_data

    def load_doc_sqr_root(self, bucket_name, file_path):
        ''' This function loads square root of vector data for cosine similarity from a DataFrame.
        '''
        # get df from bucket
        df = self.load_df_from_bucket(bucket_name, file_path)
        # create empty dictionary to store the data
        all_data = {}
        # Collect the DataFrame into the driver node as a list of rows
        rows = df.collect()
        # Iterate over the rows of the DataFrame
        for row in rows:
            # Extract relevant data from the DataFrame row
            key = row[0]  # Assuming the first column is the key
            value = float(row[1])  # Assuming the second column contains the square root of the vector
            # Add the key-value pair to the dictionary
            all_data[key] = value
        return all_data


    """Process Query before search"""
    # tokenizing function
    def tokenize(self, text):
        # regular expression to find words
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
        # return a list of words in the text in lowercase
        return [token.group() for token in RE_WORD.finditer(text.lower())]

    # function to preprocess the query
    def processQuery(self, query):
        ''' This function tokenizes the query using the staff-provided tokenizer from
            Assignment 3 (GCP part) to do the tokenization and remove stopwords.
        '''
        # tokenize the query
        tokenized = self.tokenize(query)
        # remove stopwords and stem the query
        tokens = inverted_index_gcp.InvertedIndex.filter_tokens(tokenized, True)
        # return the tokens of the query
        return tokens

    '''Vectorize Query and Documents for cosine similarity calculation and ranking'''
    def vectorizeQuery(self, query):
        ''' This function vectorizes the query and the documents in the index.
        It takes in a list of tokens representing the query and returns a counter
        of the query tokens.
        '''
        return Counter(query)

    """Process the doc from index and return vectors for cosine similarity calculation and ranking"""
    def processDocs(self, query_terms):
        ''' This function processes the documents in the index that match the query.
        '''
        # get tf-idf dict from index
        tf_idf_dict = self.index.tf_idf
        # create empty dict for vectors of query, keys are doc_ids and values are lists of tf-idf vectors
        docs_vectors = {}
        # lambda function that takes the first element of all tuples in the tf-idf dict
        get_doc_ids_for_term = lambda term: [x[0] for x in tf_idf_dict.get(term, [])]
        # get the doc_ids for each term in the query
        doc_ids_list = [get_doc_ids_for_term(term) for term in query_terms]
        # convert doc_ids_list to a set
        doc_ids = set([doc_id for sublist in doc_ids_list for doc_id in sublist])
        # iterate over terms
        for term in query_terms:
            # iterate over relevant doc_ids
            for doc_id in doc_ids:
                # try to get the tf-idf for the term in the doc
                tf_idf = next((tup for tup in tf_idf_dict[term] if doc_id == tup[0]), None)
                # if term is in the doc it should return a tuple, if not it should return None
                if tf_idf:
                    # if the doc_id is not in the dict, add it
                    if doc_id not in docs_vectors:
                        docs_vectors[doc_id] = []
                    # append the tf-idf to the list of vectors for the doc
                    docs_vectors[doc_id].append(tf_idf[1])
                else:
                    if doc_id not in docs_vectors:
                        docs_vectors[doc_id] = []
                    # if the term is not in the doc, append 0 to the list of vectors for the doc
                    docs_vectors[doc_id].append(0)
        # return the vectors of the docs
        return docs_vectors

    """Similarity and Ranking"""

    # cosine similarity: dot product of the query vector and the doc vector divided by the root of sum of squares of tf in each document
    def cosineSimilarity(self,query_vector, docs_vector):
        # create a dictionary to store the cosine similarity of each doc
        cosine_sim_docs = {}
        for doc, vector in docs_vector.items():
            dot_product = 0
            for i in range(len(vector)):
                dot_product += vector[i] * query_vector[i]
            cosine_sim_docs[doc] = dot_product

        for doc_id in cosine_sim_docs.keys():
            # divide by the root of sum of squares of tf in each document, meaning by tf(i,j) i is word and j is doc
            cosine_sim_docs[doc_id] = cosine_sim_docs[doc_id] / (self.index.doc_vector_sqr.get(doc_id) * np.sqrt(np.sum(np.array(query_vector)**2)))
            #divide by the length of the document, meaning number of tokens in doc - not needed for now
            # cosine_sim_docs[doc_id] = cosine_sim_docs[doc_id] / (np.sqrt(self.index.doc_len.get(int(doc_id))) * np.sqrt(np.sum(np.array(query_vector)**2)))
        return cosine_sim_docs

    # dot product similarity, will be used in cosine similarity
    def dotProduct_sim(query_vector, docs_vector):
        # create a dictionary to store the cosine similarity of each doc
        DP_sim_docs = {}
        for doc, vector in docs_vector.items():
            dot_product = 0
            for i in range(len(vector)):
                dot_product += vector[i] * query_vector[i]
            DP_sim_docs[doc] = dot_product
        return DP_sim_docs

    """Search functions"""
    def searchByCosineSimilarity(self, query):
        ''' This function returns up to a 100 of your best search results for the query.
        '''
        # preprocess the query
        queryTok = self.processQuery(query)
        # vectorize the query, returning a counter of the query tokens and their counts
        query_counter = self.vectorizeQuery(queryTok)
        # get the keys of the query vector, meaning the distinct query words
        query_terms = list(query_counter.keys())
        # get the vectors of the docs that match the query
        docs_vectors = self.processDocs(query_terms)
        # convert the query vector to a list of values
        query_vector = list(query_counter.values())
        similarity_dict = self.cosineSimilarity(query_vector, docs_vectors)
        # sort the similarity dict by value in descending order
        sorted_sim_dict = dict(sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True))
        # return the top 50 results
        # return list(sorted_sim_dict.keys())[:50]
        return sorted_sim_dict
