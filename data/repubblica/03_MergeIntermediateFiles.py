import glob
import pandas as pd
import numpy as np
import pickle
import os

with open("min_df_10/vocab.pkl", "rb") as infile:
    vocab = pickle.load(infile)
    vocab_size = len(vocab)

# print(vocab)


path = glob.glob('./intermediate*/result.txt')

# a = np.asarray([[1,2,3], [3,4,5]])
# b = np.asarray([[6,7,8], [9,10,11]])
# c = np.asarray([[6,7,8], [9,10,11]])

# lista = [a,b,c]
# print(np.vstack(lista))


print(path)
print("\n")

listMatrix = []

for f in path:
    print(f)   
    matrix = np.loadtxt(f)
    listMatrix.append(matrix)

    print(matrix.shape)

finalMatrix = np.vstack(listMatrix)

if os.path.isdir(f"intermediate") == False:
    os.mkdir(f"intermediate")


np.savetxt('intermediate/test.txt', finalMatrix, delimiter=' ')
