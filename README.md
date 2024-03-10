# **Information Retrieval Project**
This project contains the implementation of a search engine that we built for IR course at Ben Gurion University.
## Directories
- **colab** - the tests notebook we ran on colab, and the write to bucket notebook which we used in order to write pkl, bin and compressed csv files that contain the objects required for the index.
- **docs** - documentation relevant for the assignment, such as report of the experiment, graphs notebook and presentation for class.
- **gcp** - shell scripts for vm instance on gcp.
- **initialization** - shell scripts for initializing clusters with the matching dependencies needed for computations.
- **metrics** - detailed statistics for the experiments and major versions we created.
- **tests** - contains test queries provided by course staff.
## Code Files
All the code files are written in python.
- **inverted_index_gcp.py** - the implementation of the Inverted Index, which has the following features:
  - `df` - a dictionary of { term : num of appearances in corpus }
  - `_posting_list` - variable that will store current posting list.
  - `doc_len` - a dictionary of { doc_id : length }
  - `doc_vec_sqr` - a dictionary of { doc_id : sqrt(sum (tf)^2) }
  - `pr` - a dictionary of { doc_id : page rank score }
  - `stemmer` - a porter stemmer
  - `avg_doc_len` - average length of all docs in corpus
  - `tf_idf_title` - a dictionary that contains { term : [(doc_id, tf),...] } for each term and title in the corpus
  - `doc_vec_sqr_title` - same as doc_vec_sqr, but on titles
    
  It also implements function related to reading, writing, tokenizing and filtering.
 - **search_Engine.py** - this is the backend of the search engine. it includes all pieces of code related to the logic of the engine. it initialized an object of Inverted Index, reads all data from bucket and converts it to dictionaries with the loading functions.
   It also has local variables for storing the query's temporary result until they are sent to user.
   The `searchEngine` class within this module offers the following

   **Functionality:**
    - Loading data from a Google Cloud Storage (GCP) bucket, including inverted indices, document lengths, IDF scores, document vectors, page ranks, and word embeddings.
    - Processing queries before search, including query expansion and tokenization.
    - Vectorizing queries and documents for cosine similarity calculation and ranking.
    - Implementing various search algorithms such as TF-IDF, BM25, and cosine similarity.
    - Combining search results from title and text indices, weighting them appropriately.
    - Adding page rank scores to the search results for relevance ranking.
    - Resetting dictionaries after each search operation.
      
    **Dependencies**
      
      The module relies on several packages, including:
      - `gensim`: for loading and utilizing word embeddings.
      - `numpy`: for numerical operations.
      - `pandas`: for data manipulation.
      - `google-cloud-storage`: for interacting with Google Cloud Storage.
  - **search_frontend.py** - a server which accepts requests by implementing flask app. access to endpoints is made by using the search function, which returns up to 100 search results for a given query.
  - **testEngine.py** - a tests file based on the colab notebook. this file was used for getting statistics of queries from **queries_train.json** and writing them to csv file.
## How to run queries on the search engine
     http://34.72.210.232:8080/search?query=YOUR+QUERY
   
  
