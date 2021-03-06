# python 2.7
import os
import pickle
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import glob

import numpy as np


def saveData(data, outputFolder):
    df = pd.DataFrame(data=data, columns=['sentences'])

    # print(df)

    vectorizer = CountVectorizer(vocabulary=vocab, min_df=0, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(df['sentences'].values)
    result = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names())
    # print(result)
    # print(result.values.sum())


    # save the data

    if os.path.isdir(f"{outputFolder}") == False:
        os.mkdir(f"{outputFolder}")

    result.to_csv(f"{outputFolder}/result.txt", header=None, index=None, sep=' ', mode='a')


if __name__ == "__main__":
    # load the data

    with open("min_df_10/vocab.pkl", "rb") as infile:
        vocab = pickle.load(infile)
        vocab_size = len(vocab)

    i = 0
    step_save = 20000


    def giant_list(json_files):
        global i

        data = []
        for path in json_files:
            i += 1
            data.append(read_json_file(path))
            if ( i % 200 == 0):
                print(f"STEP {i}")
            if( i % step_save == 0):
                print(f"SAVED {int(i/step_save)}")
                saveData(data, int(i/step_save) )
                data = []

        return data



    def read_json_file(path_to_file):
        with open(path_to_file) as p:
            try:
                text = json.load(p)["text"]
                if text is not None:
                    return text.lower()
            except ValueError as e:
                print("ERRORE")
                return ""



    # Read data
    print('reading data...')

    inputFolder = "./dataProva"
    # outputFolder = f"intermediate{int(i/step_save) + 1}"
    outputFolder = f"intermediateProva"

    path = glob.glob(f'{inputFolder}/*')
    data = giant_list(path)
    # saveData(data, int(i/step_save) + 1)
    saveData(data, outputFolder)

    with open(f"{outputFolder}/vocab_dict.json", "w") as outfile:
        json.dump(vocab, outfile, ensure_ascii=False)
