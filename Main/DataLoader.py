#!/usr/bin/env python
# DataLoader.py

import os
import numpy
import six.moves.cPickle as pickle
import gzip


def prepare_data(seqs, labels, maxlen=None):
    """ Create matrices from the datasets. """

    lengths = [len(s) for s in seqs]

    # If maxlen != None, consider the max sequence length as the maxlen value
    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)

    # llist: a list of sentences from the seq
    llist = numpy.zeros((maxlen, n_samples)).astype('int64')
    llist_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    labels = numpy.array(labels).astype('int32')

    for idx, s in enumerate(seqs):
        llist[:lengths[idx], idx] = s
        llist_mask[:lengths[idx], idx] = 1.

    return llist, llist_mask, labels


def get_dataset_file(dataset, default_dataset, origin):
    """
    Look for it as if it was a full path, if not,
    try local file, if not try in the data directory.
    Finally, Download dataset if it is not present
    """

    data_dir, data_file = os.path.split(dataset)

    if not os.path.isfile(dataset):
        if data_dir == "":
            dataset = os.path.join(os.path.split(__file__)[0], dataset)
        from six.moves import urllib
        print('Downloading data from %s' % origin)
        print('dataset : %s' % dataset)
        urllib.request.urlretrieve(origin, dataset)

    return dataset


def load_data(path="imdb.pkl", n_words=10000, valid_portion=0.1, maxlen=None, sort_by_len=True):
    """ Loads train, valid and test datasets """

    # Load the dataset
    path = get_dataset_file(path, "imdb.pkl", "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_dataset = pickle.load(f)
    test_dataset = pickle.load(f)
    f.close()

    # If maxlen != None, consider the max sequence length we use in the train/valid set
    if maxlen:
        train_dataset_x = []
        train_dataset_y = []
        for x, y in zip(train_dataset[0], train_dataset[1]):
            if len(x) < maxlen:
                train_dataset_x.append(x)
                train_dataset_y.append(y)
        train_dataset = (train_dataset_x, train_dataset_y)
        del train_dataset_x, train_dataset_y

    # split training set into validation set
    train_dataset_x, train_dataset_y = train_dataset
    n_samples = len(train_dataset_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_dataset_x = [train_dataset_x[s] for s in sidx[n_train:]]
    valid_dataset_y = [train_dataset_y[s] for s in sidx[n_train:]]
    train_dataset_x = [train_dataset_x[s] for s in sidx[:n_train]]
    train_dataset_y = [train_dataset_y[s] for s in sidx[:n_train]]

    train_dataset = (train_dataset_x, train_dataset_y)
    valid_dataset = (valid_dataset_x, valid_dataset_y)

    # The number of words to keep in the vocabulary. All extra words are set to unknown(1).
    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_dataset_x, test_dataset_y = test_dataset
    valid_dataset_x, valid_dataset_y = valid_dataset
    train_dataset_x, train_dataset_y = train_dataset

    train_dataset_x = remove_unk(train_dataset_x)
    valid_dataset_x = remove_unk(valid_dataset_x)
    test_dataset_x = remove_unk(test_dataset_x)

    # Sort by the sequence length for the train, valid and test set.
    # This allow faster execution as it cause less padding per mini-batch.
    # Another mechanism must be used to shuffle the train set at each epoch.
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_dataset_x)
        test_dataset_x = [test_dataset_x[i] for i in sorted_index]
        test_dataset_y = [test_dataset_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_dataset_x)
        valid_dataset_x = [valid_dataset_x[i] for i in sorted_index]
        valid_dataset_y = [valid_dataset_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_dataset_x)
        train_dataset_x = [train_dataset_x[i] for i in sorted_index]
        train_dataset_y = [train_dataset_y[i] for i in sorted_index]

    train = (train_dataset_x, train_dataset_y)
    valid = (valid_dataset_x, valid_dataset_y)
    test = (test_dataset_x, test_dataset_y)

    return train, valid, test
