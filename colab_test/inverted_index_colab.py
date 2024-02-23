import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import os
import re
from operator import itemgetter
from pathlib import Path
import pickle
from contextlib import closing
from nltk.stem.porter import *
from nltk.corpus import stopwords
import math
from graphframes import *
from pyspark.sql.functions import col


BLOCK_SIZE = 1999998

class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb') 
                          for i in itertools.count())
        self._f = next(self._file_gen)
    
    def write(self, b):
      locs = []
      while len(b) > 0:
        pos = self._f.tell()
        remaining = BLOCK_SIZE - pos
        # if the current file is full, close and open a new one.
        if remaining == 0:  
          self._f.close()
          self._f = next(self._file_gen)
          pos, remaining = 0, BLOCK_SIZE
        self._f.write(b[:remaining])
        locs.append((self._f.name, pos))
        b = b[remaining:]
      return locs

    def close(self):
      self._f.close()

class MultiFileReader:
  """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """
  def __init__(self):
    self._open_files = {}

  def read(self, locs, n_bytes):
    b = []
    for f_name, offset in locs:
      if f_name not in self._open_files:
        self._open_files[f_name] = open(f_name, 'rb')
      f = self._open_files[f_name]
      f.seek(offset)
      n_read = min(n_bytes, BLOCK_SIZE - offset)
      b.append(f.read(n_read))
      n_bytes -= n_read
    return b''.join(b)
  
  def close(self):
    for f in self._open_files.values():
      f.close()

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()
    return False

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this 
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer

class InvertedIndex:  
  def __init__(self, docs={}):
    """ Initializes the inverted index and add documents to it (if provided).
    Parameters:
    -----------
      docs: dict mapping doc_id to list of tokens
    """
    # add a porter stemmer
    self.stemmer = PorterStemmer()
    # stores document frequency per term
    self.df = Counter()
    # stores total frequency per term
    self.term_total = Counter()
    # stores posting list per term while building the index (internally), 
    # otherwise too big to store in memory.
    self._posting_list = defaultdict(list)
    # mapping a term to posting file locations, which is a list of 
    # (file_name, offset) pairs. Since posting lists are big we are going to
    # write them to disk and just save their location in this list. We are 
    # using the MultiFileWriter helper class to write fixed-size files and store
    # for each term/posting list its list of locations. The offset represents 
    # the number of bytes from the beginning of the file where the posting list
    # starts. 
    self.posting_locs = defaultdict(list)
    self.n = 0
    for doc_id, tokens in docs.items():
      self.add_doc(doc_id, tokens)

    # calculate tf-idf for each term in the index and store
    self.tf_idf = {}
    self.calculate_tf_idf()

    # store page rank, right now it None
    self.pr = None

  def add_doc(self, doc_id, tokens):
    """ Adds a document to the index with a given `doc_id` and tokens. It counts
        the tf of tokens, then update the index (in memory, no storage 
        side-effects).
    """
    self.n = self.n+1
    # remove stopwords without stemming
    tokens = self.filter_tokens(tokens)
    # remove stopwords and stem the tokens
    #tokens = self.filter_tokens(tokens, is_stem=True)
    w2cnt = Counter(tokens)
    self.term_total.update(w2cnt)
    for w, cnt in w2cnt.items():
      self.df[w] = self.df.get(w, 0) + 1
      self._posting_list[w].append((doc_id, cnt))


  def write_index(self, base_dir, name):
    """ Write the in-memory index to disk. Results in the file: 
        (1) `name`.pkl containing the global term stats (e.g. df).
    """
    self._write_globals(base_dir, name)

  def _write_globals(self, base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
      pickle.dump(self, f)

  def __getstate__(self):
    """ Modify how the object is pickled by removing the internal posting lists
        from the object's state dictionary. 
    """
    state = self.__dict__.copy()
    del state['_posting_list']
    return state

  def posting_lists_iter(self):
    """ A generator that reads one posting list from disk and yields 
        a (word:str, [(doc_id:int, tf:int), ...]) tuple.
    """
    with closing(MultiFileReader()) as reader:
      for w, locs in self.posting_locs.items():
        b = reader.read(locs, self.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(self.df[w]):
          doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
          tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
          posting_list.append((doc_id, tf))
        yield w, posting_list


  @staticmethod
  def read_index(base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
      return pickle.load(f)

  @staticmethod
  def delete_index(base_dir, name):
    path_globals = Path(base_dir) / f'{name}.pkl'
    path_globals.unlink()
    for p in Path(base_dir).rglob(f'{name}_*.bin'):
      p.unlink()


  @staticmethod
  def write_a_posting_list(b_w_pl):
    ''' Takes a (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...]) 
    and writes it out to disk as files named {bucket_id}_XXX.bin under the 
    current directory. Returns a posting locations dictionary that maps each 
    word to the list of files and offsets that contain its posting list.
    Parameters:
    -----------
      b_w_pl: tuple
        Containing a bucket id and all (word, posting list) pairs in that bucket
        (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
    Return:
      posting_locs: dict
        Posting locations for each of the words written out in this bucket.
    '''
    posting_locs = defaultdict(list)
    bucket, list_w_pl = b_w_pl

    with closing(MultiFileWriter('.', bucket)) as writer:
      for w, pl in list_w_pl: 
        # convert to bytes
        b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                      for doc_id, tf in pl])
        # write to file(s)
        locs = writer.write(b)
      # save file locations to index
        posting_locs[w].extend(locs)
    return posting_locs
  
  def filter_tokens(self, tokens, is_stem=False):
    """ Remove stopwords and stem the tokens. function takes in a list of tokens, and removes stop words from list of tokens."""
    # create list of stop words
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
    all_stopwords = english_stopwords.union(corpus_stopwords)
    # if not stemming, remove stop words
    if not is_stem:
        tokens = [t for t in tokens if t not in all_stopwords]
    else:
        # if stemming, stem the tokens
        tokens = [self.stemmer.stem(t) for t in tokens if t not in all_stopwords]
    return tokens

  def calculate_tf_idf(self):
      """Calculate TF-IDF for each term in a document and returns a dictionary where keys are terms,
      and values is list of tuples where each tuple consists of (doc_id, tf-idf)."""
      idf = {term: math.log(self.n / df) for term, df in self.df.items()}
      tf_idf = defaultdict(list)
      for term, pl in self._posting_list.items():
          for doc_id, tf in pl:
              tf_idf[term].append((doc_id, tf * idf[term]))
      self.tf_idf = tf_idf

  def generate_graph(self, pages):
        ''' Compute the directed graph generated by wiki links.
        Parameters:
        -----------
          pages: RDD
            An RDD where each row consists of one wikipedia articles with 'id' and
            'anchor_text'.
        Returns:
        --------
          edges: RDD
            An RDD where each row represents an edge in the directed graph created by
            the wikipedia links. The first entry should the source page id and the
            second entry is the destination page id. No duplicates should be present.
          vertices: RDD
            An RDD where each row represents a vetrix (node) in the directed graph
            created by the wikipedia links. No duplicates should be present.
        '''
        # extract source page id and destination page ids from each row
        edges = pages.flatMap(lambda row: [(row.id, link.id) for link in row.anchor_text])
        # remove duplicates from edges RDD
        edges = edges.distinct()
        # extract unique vertices from both source and destination ids
        vertices_source = edges.map(lambda edge: edge[0])
        vertices_dest = edges.map(lambda edge: edge[1])
        # combine source and destination vertices and remove duplicates
        vertices = (vertices_source.union(vertices_dest).distinct()).map(lambda x: (x,))
        edgesDF = edges.toDF(['src', 'dst']).repartition(4, 'src')
        verticesDF = vertices.toDF(['id']).repartition(4, 'id')
        g = GraphFrame(verticesDF, edgesDF)
        pr_results = g.pageRank(resetProbability=0.15, maxIter=10)
        pr = pr_results.vertices.select("id", "pagerank")
        self.pr = pr.sort(col('pagerank').desc())
