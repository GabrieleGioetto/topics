import fasttext.util
import fasttext
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns

result = []
words = []
word_inglesi = ["mf", "gp", "qe", "to", "is", "not", "that", "this", "que", "atp",
                "dax", "fed", "wta", "pmi", "bund", "pogba", "belen", "jorge", "inps", "dani"]
word_topics = []

sport = [1, 2, 3, 4, 15, 17, 45]

with open("result_PCA.txt", "r") as f:
    for line in f.readlines():
        fields = line.split(",")
        if(fields[1] not in word_inglesi):
            result.append([float(fields[2])*10, float(fields[3])*10])
            words.append(fields[1])
            word_topics.append(fields[0])

# print(result)
result = np.array(result)


# result_np = np.asarray(result)
# np.savetxt('result_PCA.txt', result_np, delimiter=',')

sns.palplot(sns.color_palette("muted"))

# Get Unique continents
color_labels = list(set([0, 1]))

# List of colors in the color palettes
rgb_values = sns.color_palette("Set2", len(color_labels))

# Map continents to the colors
color_map = dict(zip(color_labels, rgb_values))

# print(
#     list(map(lambda x: color_map[0] if x in sport else color_map[1], word_topics)))

print(color_map)
print(word_topics)
print(sport)

fig, ax = plt.subplots()
ax.scatter(result[:, 0], result[:, 1], c=list(
    map(lambda x: color_map[0] if int(x) in sport else color_map[1], word_topics)))


# ax.scatter(result[:, 0], result[:, 1])
# for i, word in enumerate(words):
#     ax.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.savefig('PCA.png', bbox_inches="tight")

# ----------------------------------------------------------------------

# Get Unique continents
color_labels = list(set(word_topics))

# List of colors in the color palettes
rgb_values = sns.color_palette("Set2", len(color_labels))

# Map continents to the colors
color_map = dict(zip(color_labels, rgb_values))

fig, ax = plt.subplots()
ax.scatter(result[:, 0], result[:, 1], c=list(
    map(lambda x: color_map[x], word_topics)))


# ax.scatter(result[:, 0], result[:, 1])
# for i, word in enumerate(words):
#     ax.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.savefig('PCA_all_topcics.png', bbox_inches="tight")
