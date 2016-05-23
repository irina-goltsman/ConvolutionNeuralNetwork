# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '..')
import data_preprocessing as dp
import time

from sklearn.feature_extraction.text import CountVectorizer
import cPickle


def load_data(data_files):
    data = []
    for label in [0, 1]:
        with open(data_files[label], "rb") as f:
            for line in f:
                data.append([line.strip(), label])
    data = pd.DataFrame(data, columns=["text", "label"])
    return data


def make_vocab_list(data):
    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(data["text"])
    return vectorizer.vocabulary_


def load_bin_vec(fname, vocab):
    """
    Нет смысла созранять всю модель Word2Vec, сохраню только те слова, которые встречаются в датасете.
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(model, vocab, min_df=1, k=300):
    """
        Расширяем модель неизвестными словами, рандомно докидывая вектора -0.25 до 0.25
    """
    for word in vocab:
        if word not in model and vocab[word] >= min_df:
            model[word] = np.random.uniform(-0.25,0.25,k)


def get_embedding_matrix(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W_matrix = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W_matrix[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W_matrix[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W_matrix, word_idx_map


if __name__ == "__main__":
    start_time = time.time()
    model_path = "../models/GoogleNews-vectors-negative300.bin"
    data_files = ["../data/rt-polaritydata/rt-polarity.pos", "../data/rt-polaritydata/rt-polarity.neg"]
    print "loading data...\n",
    data = load_data(data_files)
    print "data loaded!"
    data["text"] = data["text"].apply(dp.clean_str)
    vocabulary = make_vocab_list(data)
    max_l = np.max(vocabulary.values())
    print "number of sentences: " + str(len(data))
    print "vocab size: " + str(len(vocabulary))

    print "Word embedding model is loading from %s." % model_path
    word_vec = load_bin_vec(model_path, vocabulary)
    print "Word embedding model has been loaded."
    print "num words already in word2vec: " + str(len(word_vec))

    # Расширяем модель неизвестными словами, рандомно докидывая вектора -0.25 до 0.25
    add_unknown_words(word_vec, vocabulary)
    # W - матрица всех слов из модели word2vec и word_idx_map - словарь, по слову можно узнать id, чтобы вызвать W[id]
    W_matrix, word_idx_map = get_embedding_matrix(word_vec)
    # Альтернативно можно не использовать word2vec, а просто рандомно инициализировать веса! это матрица W2
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocabulary)
    W2_matrix, _ = get_embedding_matrix(rand_vecs)
    cPickle.dump([data, W_matrix, W2_matrix, word_idx_map, vocabulary], open("mr.p", "wb"))
    print "dataset created!"
    print("--- %s seconds ---" % (time.time() - start_time))