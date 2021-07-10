import os
import sys
import argparse

import gensim
import git
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import file_handling as fh
from scholar import Scholar
from compute_npmi import compute_npmi_at_n_during_training


def main(call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    parser.add_argument(
        "-k",
        dest="n_topics",
        type=int,
        default=20,
        help="Size of latent representation (~num topics)",
    )
    parser.add_argument(
        "-l",
        dest="learning_rate",
        type=float,
        default=0.002,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--eta_bn_anneal_step_const",
        type=float,
        default=0.75,
        help="When to terminate batch-norm annealing, as a percentage of total epochs"
    )

    parser.add_argument(
        "-m",
        dest="momentum",
        type=float,
        default=0.99,
        help="beta1 for Adam",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=200,
        help="Size of minibatches",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait without improvement to dev-metric",
    )
    parser.add_argument(
        "--dev_metric",
        dest="dev_metric",
        type=str,
        default="perplexity",  # TODO: constrain options
        help="Optimize accuracy, perplexity, or internal npmi",
    )
    parser.add_argument(
        "--npmi_words",
        type=int,
        default=10,
        help="Number of words to use when calculating npmi"
    )
    parser.add_argument(
        "--train_prefix",
        type=str,
        default="train",
        help="Prefix of train set",
    )
    parser.add_argument(
        "--dev_prefix",
        type=str,
        default=None,
        help="Prefix of dev set.",
    )
    parser.add_argument(
        "--test_prefix",
        type=str,
        default=None,
        help="Prefix of test set",
    )
    parser.add_argument(
        "--no_bow_reconstruction_loss",
        action="store_false",
        dest="reconstruct_bow",
        default=True,
        help="Include the standard reconstruction of document word counts",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Read labels from input_dir/[train|test].labels.csv",
    )
    parser.add_argument(
        "--prior_covars",
        type=str,
        default=None,
        help="Read prior covariates from files with these names (comma-separated)",
    )
    parser.add_argument(
        "--topic_covars",
        type=str,
        default=None,
        help="Read topic covariates from files with these names (comma-separated)",
    )
    parser.add_argument(
        "--interactions",
        action="store_true",
        default=False,
        help="Use interactions between topics and topic covariates",
    )
    parser.add_argument(
        "--no_covars_predict",
        action="store_false",
        dest="covars_predict",
        default=True,
        help="Do not use covariates as input to classifier",
    )
    parser.add_argument(
        "--no_topics_predict",
        action="store_false",
        dest="topics_predict",
        default=True,
        help="Do not use topics as input to classifier",
    )
    parser.add_argument(
        "--min_prior_covar_count",
        type=int,
        default=None,
        help="Drop prior covariates with less than this many non-zero values in the training dataa",
    )
    parser.add_argument(
        "--min_topic_covar_count",
        type=int,
        default=None,
        help="Drop topic covariates with less than this many non-zero values in the training dataa",
    )
    parser.add_argument(
        "--classifier_loss_weight",
        type=float,
        default=1.0,
        help="Weight to give portion of loss from classification",
    )
    parser.add_argument(
        "-r",
        action="store_true",
        default=False,
        help="Use default regularization",
    )
    parser.add_argument(
        "--l1_topics",
        type=float,
        default=0.0,
        help="Regularization strength on topic weights",
    )
    parser.add_argument(
        "--l1_topic_covars",
        type=float,
        default=0.0,
        help="Regularization strength on topic covariate weights",
    )
    parser.add_argument(
        "--l1_interactions",
        type=float,
        default=0.0,
        help="Regularization strength on topic covariate interaction weights",
    )
    parser.add_argument(
        "--l2_prior_covars",
        type=float,
        default=0.0,
        help="Regularization strength on prior covariate weights",
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        default=False,
        help="Restart training with model in output-dir",
    )

    parser.add_argument(
        "--save_at_training_end",
        action="store_true",
        default=False,
        help="Save model at the end of training",
    )

    parser.add_argument(
        "--emb_dim",
        type=int,
        default=300,
        help="Dimension of input embeddings",
    )

    parser.add_argument(
        "--background_embeddings",
        nargs="?",
        const='random',
        help="`--background-embeddings <optional path to embeddings>`"
    )
    parser.add_argument(
        "--deviation_embeddings",
        nargs="?",
        const='random',
        help="`--deviation-embeddings <optional path to embeddings>`"
    )
    parser.add_argument(
        "--deviation_embedding_covar",
        help="The covariate by which to vary the embeddings"
    )

    parser.add_argument(
        "--fix_background_embeddings",
        dest="update_background_embeddings",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--fix_deviation_embeddings",
        dest="update_deviation_embeddings",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--ignore_deviation_embeddings",
        action="store_true",
        default=False,
        help="Experimental baseline to maintain parameter number",
    )
    parser.add_argument(
        "--zero_out_embeddings",
        action="store_true",
        default=False,
        help="Experimental switch to set all embeddings to 0",
    )

    parser.add_argument(
        "--doc_reps_dir",
        help="Use document representation & specify the location",
    )
    parser.add_argument(
        "--doc_reconstruction_weight",
        type=float,
        default=None,
        help="How much to weigh doc repesentation reconstruction (0 means none)",
    )
    parser.add_argument(
        "--doc_reconstruction_temp",
        type=float,
        default=None,
        help="Temperature to use when softmaxing over the doc reconstruction logits",
    )
    parser.add_argument(
        "--doc_reconstruction_min_count",
        type=float,
        default=0.,
        help="Minimum pseudo-count to accept",
    )
    parser.add_argument(
        "--doc_reconstruction_logit_clipping",
        type=float,
        default=None,
        help="Keep only the teacher logits corresponding to the top `N * x` unique words for each doc",
    )
    parser.add_argument(
        "--attend_over_doc_reps",
        action="store_true",
        default=False,
        help="Attend over the doc-representation sequence",
    )
    parser.add_argument(
        "--use_doc_layer",
        action="store_true",
        default=False,
        help="Use a document projection layer",
    )
    parser.add_argument(
        "--classify_from_doc_reps",
        action="store_true",
        help="Use document representations to classify?"
    )
    parser.add_argument(
        "--randomize_doc_reps",
        action="store_true",
        help="Baseline to randomize the document representations"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Hyperparameter for logistic normal prior",
    )
    parser.add_argument(
        "--no_bg",
        action="store_true",
        default=False,
        help="Do not use background freq",
    )
    parser.add_argument(
        "--dev_folds",
        type=int,
        default=0,
        help="Number of dev folds. Ignored if --dev-prefix is used. default=%default"
    )
    parser.add_argument(
        "--dev_fold",
        type=int,
        default=0,
        help="Fold to use as dev (if dev_folds > 0). Ignored if --dev-prefix is used. default=%default",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="GPU to use"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed"
    )

    options = parser.parse_args(call)

    input_dir = options.input_directory

    if options.r:
        options.l1_topics = 1.0
        options.l1_topic_covars = 1.0
        options.l1_interactions = 1.0

    if options.dev_prefix:
        options.dev_folds = 0
        options.dev_fold = 0

    if options.seed is not None:
        rng = np.random.RandomState(options.seed)
        seed = options.seed
    else:
        rng = np.random.RandomState(np.random.randint(0, 100000))
        seed = None

    train_X, vocab, train_row_selector, train_ids = load_word_counts(
        input_dir, options.train_prefix
    )
    train_labels, label_type, label_names, n_labels = load_labels(
        input_dir, options.train_prefix, train_row_selector, options.labels
    )
    (
        train_prior_covars,
        prior_covar_selector,
        prior_covar_names,
        n_prior_covars,
    ) = load_covariates(
        input_dir,
        options.train_prefix,
        train_row_selector,
        options.prior_covars,
        options.min_prior_covar_count,
    )
    (
        train_topic_covars,
        topic_covar_selector,
        topic_covar_names,
        n_topic_covars,
    ) = load_covariates(
        input_dir,
        options.train_prefix,
        train_row_selector,
        options.topic_covars,
        options.min_topic_covar_count,
    )


    print("Loading document representations")
    train_doc_reps = load_doc_reps(
        options.doc_reps_dir,
        prefix=options.train_prefix,
        row_selector=train_row_selector,
        use_sequences=options.attend_over_doc_reps,
    )
    options.n_train, vocab_size = train_X.shape
    options.n_labels = n_labels

    if (
        options.doc_reconstruction_logit_clipping is not None
        and options.doc_reconstruction_logit_clipping > 0
    ):
        # limit the document representations to the top k labels
        doc_tokens = np.array((train_X > 0).sum(1)).reshape(-1)

        for i, (row, total) in enumerate(zip(train_doc_reps, doc_tokens)):
            k = options.doc_reconstruction_logit_clipping * total  # keep this many logits
            if k < vocab_size:
                min_logit = np.quantile(row, 1 - k / vocab_size)
                train_doc_reps[i, train_doc_reps[i] < min_logit] = -np.inf

    if n_labels > 0:
        print("Train label proportions:", np.mean(train_labels, axis=0))

    embeddings = {}
    if options.background_embeddings:
        fpath = None if options.background_embeddings == 'random' else options.background_embeddings
        embeddings['background'] = load_word_vectors(
            fpath=fpath,  # if None, they are randomly initialized
            emb_dim=options.emb_dim,
            update_embeddings=options.update_background_embeddings,
            rng=rng,
            vocab=vocab,
        )



    model_fpath = os.path.join(options.output_dir, "torch_model.pt")
    model, _ = load_scholar_model(model_fpath, embeddings)
    model.eval()



    print("Saving document representations")
    save_document_representations(
        model,
        train_X,
        train_labels,
        train_prior_covars,
        train_topic_covars,
        train_doc_reps,
        train_ids,
        options.output_dir,
        "train",
        batch_size=options.batch_size,
    )


def load_scholar_model(inpath, embeddings=None, map_location=None):
    """
    Load the Scholar model
    """
    checkpoint = torch.load(inpath, map_location=map_location)
    scholar_kwargs = checkpoint["scholar_kwargs"]
    scholar_kwargs["init_embeddings"] = embeddings

    model = Scholar(**scholar_kwargs)
    model._model.load_state_dict(checkpoint["model_state_dict"])
    model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, checkpoint


def get_minibatch(X, Y, PC, TC, DR, batch, batch_size=200):
    # Get a particular non-random segment of the data
    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / float(batch_size)))
    if batch < n_batches - 1:
        ixs = np.arange(batch * batch_size, (batch + 1) * batch_size)
    else:
        ixs = np.arange(batch * batch_size, n_items)

    X_mb = X[ixs, :].astype("float32")
    X_mb = X_mb.todense()

    if Y is not None:
        Y_mb = Y[ixs, :].astype("float32")
    else:
        Y_mb = None

    if PC is not None:
        PC_mb = PC[ixs, :].astype("float32")
    else:
        PC_mb = None

    if TC is not None:
        TC_mb = TC[ixs, :].astype("float32")
    else:
        TC_mb = None

    if DR is not None:
        DR_mb = DR[ixs, :].astype("float32")
    else:
        DR_mb = None

    return X_mb, Y_mb, PC_mb, TC_mb, DR_mb


def save_document_representations(
    model, X, Y, PC, TC, DR, ids, output_dir, partition, batch_size=200
):
    # compute the mean of the posterior of the latent representation for each documetn and save it
    if Y is not None:
        Y = np.zeros_like(Y)

    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / batch_size))
    thetas = []

    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs, batch_drs = get_minibatch(
            X, Y, PC, TC, DR, i, batch_size
        )

        print("SHAPE: ")
        print(batch_xs.shape)
        # print(batch_ys.shape)
        # print(batch_pcs.shape)
        # print(batch_tcs.shape)
        # print(batch_drs.shape)
        thetas.append(
            model.compute_theta(batch_xs, batch_ys,
                                batch_pcs, batch_tcs, batch_drs)
        )
    theta = np.vstack(thetas)

    np.savez(
        os.path.join(output_dir, "theta." + partition + ".npz"), theta=theta, ids=ids
    )

def load_word_counts(input_dir, input_prefix, vocab=None):
    print("Loading data")
    # laod the word counts and convert to a dense matrix
    X = fh.load_sparse(os.path.join(input_dir, input_prefix + ".npz"))
    X = X.astype(np.float32)
    # load the vocabulary
    if vocab is None:
        vocab = fh.read_json(os.path.join(
            input_dir, input_prefix + ".vocab.json"))
    n_items, vocab_size = X.shape
    assert vocab_size == len(vocab)
    print("Loaded %d documents with %d features" % (n_items, vocab_size))

    ids = fh.read_json(os.path.join(input_dir, input_prefix + ".ids.json"))

    # filter out empty documents and return a boolean selector for filtering labels and covariates
    row_selector = np.array(X.sum(axis=1) > 0, dtype=bool).reshape(-1)
    print("Found %d non-empty documents" % np.sum(row_selector))
    X = X[row_selector, :]
    ids = [doc_id for i, doc_id in enumerate(ids) if row_selector[i]]

    return X, vocab, row_selector, ids

def load_doc_reps(input_dir, prefix, row_selector, use_sequences=False):
    """
    Load document representations, an [num_docs x doc_dim] matrix
    """
    if input_dir is not None:
        doc_rep_fpath = os.path.join(input_dir, f"{prefix}.npy")
        doc_reps = np.load(doc_rep_fpath)
        if not use_sequences:
            return doc_reps[row_selector, :]

        tokens_fpath = os.path.join(input_dir, f"{prefix}.tokens.npy")
        tokens = np.load(tokens_fpath)[:, :, None]
        mask = tokens > 0
        doc_reps = np.insert(doc_reps, [0], mask, axis=2)
        return doc_reps[row_selector, :]


def train_dev_split(options, rng):
    # randomly split into train and dev
    if options.dev_folds > 0:
        n_dev = int(options.n_train / options.dev_folds)
        indices = np.array(range(options.n_train), dtype=int)
        rng.shuffle(indices)
        if options.dev_fold < options.dev_folds - 1:
            dev_indices = indices[
                n_dev * options.dev_fold: n_dev * (options.dev_fold + 1)
            ]
        else:
            dev_indices = indices[n_dev * options.dev_fold:]
        train_indices = list(set(indices) - set(dev_indices))
        return train_indices, dev_indices

    else:
        return None, None


def split_matrix(train_X, train_indices, dev_indices):
    # split a matrix (word counts, labels, or covariates), into train and dev
    if train_X is not None and dev_indices is not None:
        dev_X = train_X[dev_indices, :]
        train_X = train_X[train_indices, :]
        return train_X, dev_X
    else:
        return train_X, None


def get_init_bg(data):
    # Compute the log background frequency of all words
    sums = np.sum(data, axis=0) + 1
    print("Computing background frequencies")
    print(
        "Min/max word counts in training data: %d %d"
        % (int(np.min(sums)), int(np.max(sums)))
    )
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
    return bg.reshape(-1)


def load_word_vectors(fpath, emb_dim, update_embeddings, rng, vocab):

    # load word2vec vectors if given
    if fpath is not None:
        vocab_size = len(vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        # randomly initialize word vectors for each term in the vocabualry
        embeddings = np.array(
            rng.rand(emb_dim, vocab_size) * 0.25 - 0.5, dtype=np.float32
        )
        count = 0
        print("Loading word vectors")
        # load the word2vec vectors
        if fpath.endswith('.model'):
            pretrained = gensim.models.Word2Vec.load(fpath)
        else:
            pretrained = gensim.models.KeyedVectors.load_word2vec_format(
                fpath, binary=fpath.endswith('.bin')
            )

        # replace the randomly initialized vectors with the word2vec ones for any that are available
        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                embeddings[:, index] = pretrained[word]

        print("Found embeddings for %d words" % count)
    else:

        update_embeddings = True  # always true if unspecified
        embeddings = None

    return embeddings, update_embeddings


def load_covariates(
    input_dir,
    input_prefix,
    row_selector,
    covars_to_load,
    min_count=None,
    covariate_selector=None,
):

    covariates = None
    covariate_names = None
    n_covariates = 0
    if covars_to_load is not None:
        covariate_list = []
        covariate_names_list = []
        covar_file_names = covars_to_load.split(",")
        # split the given covariate names by commas, and load each one
        for covar_file_name in covar_file_names:
            covariates_file = os.path.join(
                input_dir, input_prefix + "." + covar_file_name + ".csv"
            )
            if os.path.exists(covariates_file):
                print("Loading covariates from", covariates_file)
                temp = pd.read_csv(covariates_file, header=0, index_col=0)
                covariate_names = covar_file_name + '_' + temp.columns
                covariates = np.array(temp.values, dtype=np.float32)
                # select the rows that match the non-empty documents (from load_word_counts)
                covariates = covariates[row_selector, :]
                covariate_list.append(covariates)
                covariate_names_list.extend(covariate_names)

            else:
                raise (
                    FileNotFoundError(
                        "Covariates file {:s} not found".format(
                            covariates_file)
                    )
                )

        # combine the separate covariates into a single matrix
        covariates = np.hstack(covariate_list)
        covariate_names = covariate_names_list

        _, n_covariates = covariates.shape

        # if a covariate_selector has been given (from a previous call of load_covariates), drop columns
        if covariate_selector is not None:
            covariates = covariates[:, covariate_selector]
            covariate_names = [
                name for i, name in enumerate(covariate_names) if covariate_selector[i]
            ]
            n_covariates = len(covariate_names)
        # otherwise, choose which columns to drop based on how common they are (for binary covariates)
        elif min_count is not None and int(min_count) > 0:
            print("Removing rare covariates")
            covar_sums = covariates.sum(axis=0).reshape((n_covariates,))
            covariate_selector = covar_sums > int(min_count)
            covariates = covariates[:, covariate_selector]
            covariate_names = [
                name for i, name in enumerate(covariate_names) if covariate_selector[i]
            ]
            n_covariates = len(covariate_names)

    return covariates, covariate_selector, covariate_names, n_covariates



def load_labels(input_dir, input_prefix, row_selector, labels=None):
    label_type = None
    label_names = None
    n_labels = 0
    # load the label file if given
    if labels is not None:
        label_file = os.path.join(
            input_dir, input_prefix + "." + labels + ".csv"
        )
        if os.path.exists(label_file):
            print("Loading labels from", label_file)
            temp = pd.read_csv(label_file, header=0, index_col=0)
            label_names = temp.columns
            labels = np.array(temp.values)
            # select the rows that match the non-empty documents (from load_word_counts)
            labels = labels[row_selector, :]
            n, n_labels = labels.shape
            print("Found %d labels" % n_labels)
        else:
            raise (FileNotFoundError(
                "Label file {:s} not found".format(label_file)))

    return labels, label_type, label_names, n_labels



if __name__ == "__main__":
    main()
