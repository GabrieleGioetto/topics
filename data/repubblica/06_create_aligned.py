import glob
from utils import load_sparse,load_json, save_sparse, save_jsonlist, save_json

counts_train = load_sparse(Path(dev_dir, "train.npz"))
tokens_train = load_json(Path(dev_dir, "train.tokens.json"))
counts_test = load_sparse(Path(dev_dir, "test.npz"))
tokens_test = load_json(Path(dev_dir, "test.tokens.json"))

raw_train_file_names = glob.glob("./data/news_*")[:15150]
raw_test_file_names = glob.glob("./data/news_*")[15151:20201]


raw_data_train = [
    {'id': idx, 'text': raw_train.data[idx], 'fpath': raw_train.filenames[idx]}
    for idx in raw_ids_train
]


raw_data_test = [
    {'id': idx, 'text': raw_test.data[idx], 'fpath': raw_test.filenames[idx]}
    for idx in raw_ids_test
]
save_jsonlist(raw_data_train, "./replicated/train.jsonlist")
save_jsonlist(raw_data_test, "./replicated/test.jsonlist")
save_json([d['id'] for d in raw_data_train], "./replicated/train.ids.json")
save_json([d['id'] for d in raw_data_test], "./replicated/test.ids.json")
#  Alignment -- currently ok, but not great
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