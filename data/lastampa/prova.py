import json
import glob
from utils import load_jsonlist, save_json
from sklearn.model_selection import train_test_split
import numpy as np

a = np.arange(20)

train, test = train_test_split(a, train_size=0.99, shuffle=False)

print("train")
print(train)
print("test")
print(test)

# d = np.load('./aligned/test.npz')
# a = np.load('./aligned/train.npz')

# for k in a.files:
#     print(k)

# print(a["data"].shape)
# print(d["data"].shape)





# Opening JSON file
# raw_train_file_names = glob.glob("./data/news_*")[:2]

# with open(raw_train_file_names[0]) as f:
#     print(f.read())

# --------------------------------------------------------------------------
  
# with open("./replicated/dev/train.ids.json") as f:
    
#     tokens = load_jsonlist("./replicated/dev/train.tokens.json")

#     print(f"len tokens: {len(tokens[0])}")


#     data = json.load(f)

#     i = 0
#     count = 0
#     indexes = []
#     for id in data:
#         if(id == i):
#             i+=1
#         else:
#             print(f"i : {i}")
#             indexes.append(i)
#             i+=2
#             count+=1

#     print(f"Count: {count}")

#     indexes = indexes[::-1]

#     print(indexes)
#     tokens = tokens[0]

#     for i in indexes:
#         print(f"i pop: {i}")
#         tokens.pop(i)

#     print(f"len tokens: {len(tokens)}")

#     save_json(tokens, "./replicated/dev/train.tokens_min.json")

#     print("saved")


# with open("./data/train.txt","r") as f:
#     print(f.read(1000))