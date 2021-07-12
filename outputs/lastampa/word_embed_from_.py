import fasttext.util
import fasttext
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import sys

result = []
words = []
word_inglesi = ["mf","gp","qe","to","is","not","that","this","que","atp","dax","fed","wta","pmi","bund", "pogba","belen","jorge","inps","dani"]


with open("result_PCA.txt","r") as f:
    for line in f.readlines():
        fields = line.split(",")
        if(fields[0] not in word_inglesi):
            result.append([float(fields[1])*10,float(fields[2])*10])
            words.append(fields[0])

# print(result)
result = np.array(result)


# result_np = np.asarray(result)
# np.savetxt('result_PCA.txt', result_np, delimiter=',')

fig, ax = plt.subplots()
ax.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
	ax.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.savefig('PCA.png', bbox_inches="tight")
