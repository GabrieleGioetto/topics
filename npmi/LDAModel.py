from gensim.models.ldamodel import LdaModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import string
import numpy as np
from tqdm import tqdm

print('Starting programm...')
print('Downloading dataset...')
dataset = fetch_20newsgroups(shuffle=True,
                            random_state=32,
                            remove = ('header', 'footers', 'qutes'))

for idx in range(10):
    print(dataset.data[idx], '\n\n', '#'*100, '\n\n')

#Put the data in a pandas dataframe format (articolo - numero_topic)
news_df = pd.DataFrame({'News': dataset.data,
                        'Target': dataset.target})
#Create the label in which you can see the topic name format (articolo - numero_topic, nome_topic)
news_df['Label'] = news_df['Target'].apply(lambda x: dataset.target_names[x])

#Preprocessing (To change to fit the one of the paper)
#This one is made of remove non alphabetic characters, stopwords taken from nltk and lemmatize
def clean_text(sentence):
    pattern = re.compile(r'[^a-z]')
    sentence = sentence.lower()
    sentence = pattern.sub(' ', sentence).strip()

    #Tokenize
    word_list = word_tokenize(sentence)
    #Stopwords
    stopwords_list = set(stopwords.words('english'))

    word_list = [word for word in word_list if word not in stopwords_list and len(word) > 2]
    lemma = WordNetLemmatizer()
    word_list = [lemma.lemmatize(word) for word in word_list]
    sentence = ' '.join(word_list)

    return sentence
tqdm.pandas()
news_df['News'] = news_df['News'].progress_apply(lambda x: clean_text(str(x)))
tfidf_vec = TfidfVectorizer(tokenizer = lambda x: str(x).split())
X = tfidf_vec.fit_transform(news_df['News'])
print(X.shape)

#lda = LdaModel(X, num_topics=20)
lda_model_sklearn = LatentDirichletAllocation(n_components=20,
                                              random_state=12,
                                              learning_method='online',
                                              max_iter=5,
                                              learning_offset=50,
                                              verbose=1)
lda_model_sklearn.fit(X)
doc_topic_lda = lda_model_sklearn.transform(X)
print("Closing...")

