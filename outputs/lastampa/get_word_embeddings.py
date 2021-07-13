import fasttext.util
import fasttext
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns


idTopics = []
with open("risutati.txt") as f:
    for i, line in enumerate(f):
        if(i == 3):
            idTopics = list(map(lambda x: int(float(x)), line.split("\t")))


word_inglesi = ["atp","dax","fed","wta","pmi","bund", "pogba","belen","jorge","inps","dani"]

words = []
word_topics = []

with open("topics.txt") as f:
    for i, line in enumerate(f):
        if(i in idTopics):
            for word in line.split(" ")[:10]:
                if word not in word_inglesi:
                    words.append(word)
                    word_topics.append(i)
            
fasttext.util.download_model('it', if_exists='ignore')  # Italian
print("DOWNLOADED")

ft = fasttext.load_model('cc.it.300.bin')
print("LOADED")

fasttext.util.reduce_model(ft, 100)
print("REDUCED")

# print(ft.get_word_vector('cane').shape)

# define training data
# sentences = [["ciao","cane"],
# 			["gatto","balena"]]

values = []
# words = []

print("Words:")
print(words)


i = 0
for word in words:
    value = ft.get_word_vector(word)
    values.append(ft.get_word_vector(word))
    print(f"{word} : {value}")

# model = Word2Vec(sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = values

print(f"X shape 0: {len(X)}")
print(f"X shape 1: {len(X[0])}")

# print("X")
# print(X)

sns.palplot(sns.color_palette("muted"))

# Get Unique continents
color_labels = list(set(word_topics))

# List of colors in the color palettes
rgb_values = sns.color_palette("Set2", len(color_labels))

# Map continents to the colors
color_map = dict(zip(color_labels, rgb_values))



pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection

with open("result_PCA.txt","w") as f:
    for i, word in enumerate(words):
        f.write(f"{word_topics[i]},{word},{result[i][0]},{result[i][1]}\n")


print(list(map(lambda x: color_map[x], word_topics)))
plt.scatter(result[:, 0], result[:, 1], c=list(map(lambda x: color_map[x], word_topics)))
    
    # result_np = np.asarray(result)
    # np.savetxt('result_PCA.txt', result_np, delimiter=',')

fig, ax = plt.subplots()
# for i, word in enumerate(words):
# 	ax.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.savefig('PCA.png', bbox_inches="tight")
