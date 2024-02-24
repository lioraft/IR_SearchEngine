'''
This file is the search engine for the search_frontend.py file. It is responsible for
the backend of the search engine and it's logic. the functions will be called by the frontend.
'''
import re
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import inverted_index_gcp

class searchEngine:
    def __init__(self, index_path, bucket_name=None):
        self.index = inverted_index_gcp.InvertedIndex.read_index(".", index_path, bucket_name)

    """Process Query before search"""
    # tokenizing function
    def tokenize(self, text):
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
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

    def vectorizeQuery(self, query):
        ''' This function vectorizes the query and the documents in the index.
        It takes in a list of tokens representing the query and returns a counter
        of the query tokens.
        '''
        return Counter(query)

    """Process the doc from index"""
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
    # TODO: add bm25 method
    def cosineSimilarity(self,query_terms, query_vector, docs_vector):
        # create a dictionary to store the cosine similarity of each doc
        cosine_sim_docs = {}
        for doc, vector in docs_vector.items():
            dot_product = 0
            for i in range(len(vector)):
                dot_product += vector[i] * query_vector[i]
            cosine_sim_docs[doc] = dot_product

        for doc_id in cosine_sim_docs:
            # divide by the total sqr of the documents tf-idf
            # cosine_sim_docs[doc_id] = cosine_sim_docs[doc_id] / (np.sqrt(self.index.doc_tfidf_sqr.get(doc_id)) * np.sqrt(np.sum(np.array(query_vector)**2)))
            #divide by the length of the document
            cosine_sim_docs[doc_id] = cosine_sim_docs[doc_id] / (np.sqrt(self.index.doc_len.get(doc_id)) * np.sqrt(np.sum(np.array(query_vector)**2)))
        return cosine_sim_docs

    def dotProduct_sim(query_vector, docs_vector):
        # create a dictionary to store the cosine similarity of each doc
        DP_sim_docs = {}
        for doc, vector in docs_vector.items():
            dot_product = 0
            for i in range(len(vector)):
                dot_product += vector[i] * query_vector[i]
            DP_sim_docs[doc] = dot_product
        return DP_sim_docs

    """Search"""
    # TODO: complete method
    def search(self, query):
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
        similarity_dict = self.cosineSimilarity(query_terms,query_vector, docs_vectors)
        # sort the similarity dict by value in descending order
        sorted_sim_dict = dict(sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True))
        # return the top 50 results
        # return list(sorted_sim_dict.keys())[:50]
        return sorted_sim_dict
