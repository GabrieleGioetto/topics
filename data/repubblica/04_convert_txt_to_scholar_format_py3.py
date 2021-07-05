# python 3.8
from pathlib import Path
import json
from sklearn.model_selection import train_test_split


from scipy import sparse
import numpy as np

from utils import save_sparse, load_json, save_json

if __name__ == "__main__":
    outdir = Path("aligned")
    outdir.mkdir(exist_ok=True)

    # data files
    result = np.loadtxt("intermediate1/result.txt")

    print("LOADED")

    train, test = train_test_split(result, train_size=0.75, shuffle=False)

    print("CREATED TRAIN AND TEST")

    np.savetxt('intermediate1/train.txt', train, delimiter=' ')
    print("SAVED TRAIN")

    np.savetxt('intermediate1/test.txt', test, delimiter=' ')
    print("SAVED TEST")
    

    train = sparse.coo_matrix(train)
    test = sparse.coo_matrix(test)

    save_sparse(train, "aligned/train")
    save_sparse(test,"aligned/test")

    print("SAVED SPARSE")

    # reorder vocabulary, save as list
    # vocab = load_json("intermediate7/vocab_dict.json")
    # vocab = [# ensure correct order
    #     k for k, v in sorted(vocab.items(), key=lambda kv: kv[1])
    # ]
    # save_json(vocab, "aligned/train.vocab.json")