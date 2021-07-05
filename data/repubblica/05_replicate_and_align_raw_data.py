#python 3.8
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer

from tqdm import tqdm

from utils import load_sparse, save_sparse, save_jsonlist, save_json

import glob
import spacy
import sys


def process_raw_doc(doc, vocab, lemmatizer):
    with open(doc) as f:
        data = json.load(f)
        text = data["text"]
                                                                    
        tokenized = word_tokenize(text)
        lemmatized = [        
            lemmatizer(w)[0].lemma_.lower() for w in tokenized      
        ]

        # print(lemmatized)
        # sys.exit()
        return [word for word in lemmatized if word in vocab]


def toks_to_onehot(doc, vocab):
    tokens =  [vocab[word] for word in doc]                             
    return np.bincount(tokens, minlength=len(vocab))


if __name__ == "__main__":
    ## Re-processing
    REMOVE_HEADER = False

    with open("./aligned/train.vocab.json", "r") as infile:
        vocab = json.load(infile)
        vocab_dict = dict(zip(vocab, range(len(vocab))))
    
    # Read in the ProdLDA 20ng data
    orig_counts_train = load_sparse("./aligned/train.npz").todense()
    orig_counts_test = load_sparse("./aligned/test.npz").todense()


    # Get the original raw text
    # raw_train = fetch_20newsgroups(data_home="./intermediate", subset="train")
    # raw_test = fetch_20newsgroups(data_home="./intermediate", subset="test")

    raw_train_file_names = glob.glob("./data/news_*")[:15150]
    raw_test_file_names = glob.glob("./data/news_*")[15151:20201]

    # print(len(raw_train))
    # print(raw_train[:10])
    # print(len(raw_test))

    # Sample di prova
    orig_counts_train = orig_counts_train[:20]
    orig_counts_test = orig_counts_test[:10]
    raw_train_file_names = raw_train_file_names[:20]
    raw_test_file_names = raw_test_file_names[:10]

    print("1")

    # Turn the raw text into count data
    # wnl = WordNetLemmatizer()
    wnl = spacy.load('it_core_news_sm')

    raw_tokens_train = [
        process_raw_doc(doc, vocab_dict, wnl) 
        for doc in tqdm(raw_train_file_names)
    ]
    raw_tokens_test = [
        process_raw_doc(doc, vocab_dict, wnl) 
        for doc in tqdm(raw_test_file_names)
    ]
    raw_counts_train = np.array([toks_to_onehot(doc, vocab_dict) for doc in raw_tokens_train])
    raw_counts_test = np.array([toks_to_onehot(doc, vocab_dict) for doc in raw_tokens_test])

    print("2")


    # filter out the zero-counts
    nonzero_train = raw_counts_train.sum(1) > 0
    nonzero_test = raw_counts_test.sum(1) > 0

    print(raw_counts_train.sum(1))
    print(raw_counts_test.sum(1))

    raw_ids_train = [idx for idx, keep in enumerate(nonzero_train) if keep]
    raw_ids_test = [idx for idx, keep in enumerate(nonzero_test) if keep]

    raw_tokens_train = [' '.join(raw_tokens_train[idx]) for idx in range(len(raw_tokens_train))]
    raw_tokens_test = [' '.join(raw_tokens_test[idx]) for idx in range(len(raw_tokens_test))]

    raw_counts_train = raw_counts_train[nonzero_train]
    raw_counts_test = raw_counts_test[nonzero_test]

    raw_data_train = [
        {'id': idx, 'text': raw_train.data[idx], 'fpath': raw_train.filenames[idx]}
        for idx in raw_ids_train
    ]
    raw_data_test = [
        {'id': idx, 'text': raw_test.data[idx], 'fpath': raw_test.filenames[idx]}
        for idx in raw_ids_test
    ]
    if REMOVE_HEADER:
        for doc in raw_data_train:
            doc['text'] = doc['text'][doc['text'].index('\n\n'):]
        for doc in raw_data_test:
            doc['text'] = doc['text'][doc['text'].index('\n\n'):]

    print("3")

    # keep this pseudo-ProdLDA version
    Path("replicated").mkdir(exist_ok=True)
    save_sparse(sparse.coo_matrix(raw_counts_train), "./replicated/train.npz")
    save_sparse(sparse.coo_matrix(raw_counts_test), "./replicated/test.npz")

    save_json(vocab, "./replicated/train.vocab.json")

    save_json(raw_tokens_train, "./replicated/train.tokens.json")
    save_json(raw_tokens_test, "./replicated/test.tokens.json")

    save_jsonlist(raw_data_train, "./replicated/train.jsonlist")
    save_jsonlist(raw_data_test, "./replicated/test.jsonlist")

    save_json([d['id'] for d in raw_data_train], "./replicated/train.ids.json")
    save_json([d['id'] for d in raw_data_test], "./replicated/test.ids.json")

    # Alignment -- currently ok, but not great

    print("4")

    # tf-idf transform
    tfidf = TfidfTransformer()
    tfidf.fit(np.vstack([
        orig_counts_train, orig_counts_test, raw_counts_train, raw_counts_test]
    ))

    orig_tfidf_train = tfidf.transform(orig_counts_train)
    orig_tfidf_test = tfidf.transform(orig_counts_test)

    raw_tfidf_train = tfidf.transform(raw_counts_train)
    raw_tfidf_test = tfidf.transform(raw_counts_test) 

    # get the distances (takes a couple minutes)
    dists_train = cdist(orig_tfidf_train.todense(), raw_tfidf_train.todense(), 'euclidean')
    dists_test = cdist(orig_tfidf_test.todense(), raw_tfidf_test.todense(), 'euclidean')

    min_dists_train = dists_train.argmin(1)
    min_dists_test = dists_test.argmin(1)

    print("5")

    # do the alignment
    aligned_tokens_train = [raw_tokens_train[idx] for idx in min_dists_train]
    aligned_tokens_test = [raw_tokens_test[idx] for idx in min_dists_test]

    aligned_data_train = [raw_data_train[idx] for idx in min_dists_train]
    aligned_data_test = [raw_data_test[idx] for idx in min_dists_test]

    assert(len(aligned_data_train) == orig_counts_train.shape[0])
    assert(len(aligned_data_test) == orig_counts_test.shape[0])

    print("6")

    # save the aligned data
    save_json(aligned_tokens_train, "./aligned/train.tokens.json")
    save_json(aligned_tokens_test, "./aligned/test.tokens.json")

    save_jsonlist(aligned_data_train, "./aligned/train.jsonlist")
    save_jsonlist(aligned_data_test, "./aligned/test.jsonlist")

    save_json([d['id'] for d in aligned_data_train], "./aligned/train.ids.json")
    save_json([d['id'] for d in aligned_data_test], "./aligned/test.ids.json")