import fasttext.util
import fasttext
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

idTopics = []
with open("risutati.txt") as f:
    for i, line in enumerate(f):
        if(i == 3):
            idTopics = list(map(lambda x: int(float(x)), line.split("\t")))

words = []
with open("topics.txt") as f:
    for i, line in enumerate(f):
        if(i in idTopics):
            for word in line.split("\t")[:10]:
                words.append(word)
            



print(idTopics)

fasttext.util.download_model('it', if_exists='ignore')  # Italian
print("DOWNLOADED")

ft = fasttext.load_model('cc.it.300.bin')
print("LOADED")

fasttext.util.reduce_model(ft, 100)
print("REDUCED")

# print(ft.get_word_vector('cane').shape)

# define training data
sentences = [["ciao","cane"],
			["gatto","balena"]]

values = []
# words = []

print("words")
print(words)


i = 0
for word in words:
    values.append(ft.get_word_vector(word))

# model = Word2Vec(sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = values

print("X")
print(X)

pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])

result_np = np.asarray(result)
np.savetxt('result_PCA.txt', result_np, delimiter=',')

fig, ax = plt.subplots()
for i, word in enumerate(words):
	ax.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.savefig('PCA.png', bbox_inches="tight")
