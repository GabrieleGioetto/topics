from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import random
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
import re
import string
from stop_words import get_stop_words
import sys
import os
import json
import glob




# https://github.com/adjidieng/ETM/blob/master/scripts/data_20ng.py


# Maximum / minimum document frequency
max_df = 0.7
min_df = 10  # choose desired value for min_df

# Read stopwords
stops = get_stop_words('italian')

def giant_list(json_files):
    return [read_json_file(path) for path in json_files]

i = 0

def read_json_file(path_to_file):
    global i
    i += 1
    if ( i % 200 == 0):
        print(f"STEP {i}\n")
    with open(path_to_file) as p:
        return json.load(p)["text"]


# Read data
print('reading data...')

path_zero = glob.glob('./data/news_000*')
path_one = glob.glob('./data/news_001*')

path = path_zero + path_one

data = giant_list(path)


init_docs = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', doc) for doc in data]

def contains_punctuation(w):
    return any(char in string.punctuation for char in w)

def contains_numeric(w):
    return any(char.isdigit() for char in w)

# print(init_docs)
    
init_docs = [[w.lower() for w in doc if not contains_punctuation(w)] for doc in init_docs]
init_docs = [[w for w in doc if not contains_numeric(w)] for doc in init_docs]
init_docs = [[w for w in doc if len(w)>1] for doc in init_docs]
init_docs = [" ".join(doc) for doc in init_docs]

# Create count vectorizer
print('counting document frequency of words...')
cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
cvz = cvectorizer.fit_transform(init_docs).sign()

# Get vocabulary
print('building the vocabulary...')
sum_counts = cvz.sum(axis=0)
v_size = sum_counts.shape[1]
sum_counts_np = np.zeros(v_size, dtype=int)
for v in range(v_size):
    sum_counts_np[v] = sum_counts[0,v]
word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])
del cvectorizer
print('  initial vocabulary size: {}'.format(v_size))

# Sort elements in vocabulary
idx_sort = np.argsort(sum_counts_np)
vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)]

# Filter out stopwords (if any)
vocab_aux = [w for w in vocab_aux if w not in stops]
print('  vocabulary size after removing stopwords from list: {}'.format(len(vocab_aux)))

# Create dictionary and inverse dictionary
vocab = vocab_aux
del vocab_aux
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])

# Split in train/test/valid
# print('tokenizing documents and splitting into train/test/valid...')
# num_docs_tr = len(init_docs_tr)
# trSize = num_docs_tr-100
# tsSize = len(init_docs_ts)
# vaSize = 100
# idx_permute = np.random.permutation(num_docs_tr).astype(int)

# Remove words not in train_data
# vocab = list(set([w for idx_d in range(trSize) for w in init_docs[idx_permute[idx_d]].split() if w in word2id]))
# word2id = dict([(w, j) for j, w in enumerate(vocab)])
# id2word = dict([(j, w) for j, w in enumerate(vocab)])
# print('  vocabulary after removing words not in train: {}'.format(len(vocab)))

# Split in train/test/valid
# docs_tr = [[word2id[w] for w in init_docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
# docs_va = [[word2id[w] for w in init_docs[idx_permute[idx_d+trSize]].split() if w in word2id] for idx_d in range(vaSize)]
# docs_ts = [[word2id[w] for w in init_docs[idx_d+num_docs_tr].split() if w in word2id] for idx_d in range(tsSize)]

# print('  number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
# print('  number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))
# print('  number of documents (valid): {} [this should be equal to {}]'.format(len(docs_va), vaSize))

# Remove empty documents
# print('removing empty documents...')

# def remove_empty(in_docs):
#     return [doc for doc in in_docs if doc!=[]]


# docs_tr = remove_empty(docs_tr)
# docs_ts = remove_empty(docs_ts)
# docs_va = remove_empty(docs_va)

# Remove test documents with length=1
# docs_ts = [doc for doc in docs_ts if len(doc)>1]

# Split test set in 2 halves
# print('splitting test documents in 2 halves...')
# docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
# docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]

# Getting lists of words and doc_indices
# print('creating lists of words...')

# def create_list_words(in_docs):
#     return [x for y in in_docs for x in y]

# words_tr = create_list_words(docs_tr)
# words_ts = create_list_words(docs_ts)
# words_ts_h1 = create_list_words(docs_ts_h1)
# words_ts_h2 = create_list_words(docs_ts_h2)
# words_va = create_list_words(docs_va)

# print('  len(words_tr): ', len(words_tr))
# print('  len(words_ts): ', len(words_ts))
# print('  len(words_ts_h1): ', len(words_ts_h1))
# print('  len(words_ts_h2): ', len(words_ts_h2))
# print('  len(words_va): ', len(words_va))

# Get doc indices
# print('getting doc indices...')

# def create_doc_indices(in_docs):
#     aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
#     return [int(x) for y in aux for x in y]

# doc_indices_tr = create_doc_indices(docs_tr)
# doc_indices_ts = create_doc_indices(docs_ts)
# doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
# doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
# doc_indices_va = create_doc_indices(docs_va)

# print('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
# print('  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts)), len(docs_ts)))
# print('  len(np.unique(doc_indices_ts_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h1)), len(docs_ts_h1)))
# print('  len(np.unique(doc_indices_ts_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h2)), len(docs_ts_h2)))
# print('  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_va)), len(docs_va)))

# # Number of documents in each set
# n_docs_tr = len(docs_tr)
# n_docs_ts = len(docs_ts)
# n_docs_ts_h1 = len(docs_ts_h1)
# n_docs_ts_h2 = len(docs_ts_h2)
# n_docs_va = len(docs_va)

# # Remove unused variables
# del docs_tr
# del docs_ts
# del docs_ts_h1
# del docs_ts_h2
# del docs_va

# Create bow representation
# print('creating bow representation...')

# def create_bow(doc_indices, words, n_docs, vocab_size):
#     return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

# bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
# bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))
# bow_ts_h1 = create_bow(doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len(vocab))
# bow_ts_h2 = create_bow(doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len(vocab))
# bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len(vocab))

# del words_tr
# del words_ts
# del words_ts_h1
# del words_ts_h2
# del words_va
# del doc_indices_tr
# del doc_indices_ts
# del doc_indices_ts_h1
# del doc_indices_ts_h2
# del doc_indices_va

# Write the vocabulary to a file
path_save = './min_df_' + str(min_df) + '/'
if not os.path.isdir(path_save):
    os.system('mkdir -p ' + path_save)

with open(path_save + 'vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
del vocab
