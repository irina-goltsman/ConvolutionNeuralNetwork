# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cPickle
import logging
import time
import datetime
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import sys
sys.path.insert(0, '..')
import CNNTextClassifier

def words_count(text):
    return len(text.split())

def train_and_test_cross_valid(n_epochs=10, n_folds=10, non_static=False, word_embedding="-word2vec"):
    print "loading data...",
    x = cPickle.load(open("mr.p", "rb"))
    data, w2v_matrix, random_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    print "%d samples." % len(data)
    if word_embedding == "-rand":
        print "using: random vectors"
        word_vect = random_matrix
    elif word_embedding == "-word2vec":
        print "using: word2vec vectors"
        word_vect = w2v_matrix
    else:
        raise Warning("No word_vectors!")

    max_l = max(data["text"].apply(words_count))
    print "max length of text = %d" % max_l
    data = add_idx_features(data, word_idx_map, max_l=max_l, k=300, filter_h=5)
    print data["idx_features"][0]

    clf = CNNTextClassifier.CNNTextClassifier(word_vect, learning_rate=0.1, seed=0, L2_reg=0.1,
                                              windows=[4, 5, 6], n_filters=10, k_top=1, activation='tanh',
                                              word_dimension=300, n_epochs=n_epochs, non_static=non_static)

    kf = KFold(len(data), n_folds=n_folds, shuffle=True, random_state=0)
    results = cross_val_score(clf, data["idx_features"], data["label"], cv=kf, n_jobs=1,
                              fit_params={'early_stop': True})
    mean_score = str(np.mean(results))
    #clf.fit(data["idx_features"], data["label"], early_stop=True)

    print "mean score:"
    print mean_score

    new_results_path = "../results/results_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') \
                       + '_mean'
    print "Saving results to '%s'..." % new_results_path
    with open(new_results_path, 'w') as f:
        result_str = [clf.get_params_as_string(), "scores:", str(results), "mean score:", mean_score]
        result_str = '\n'.join(result_str)
        f.write(result_str)


# TODO: вынести в препроцессинг
def add_idx_features(data, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    data_features = pd.Series([[]], index=data.index)
    for i in data.index:
        data_features[i] = get_idx_from_sent(data["text"][i], word_idx_map, max_l, k, filter_h)
    data["idx_features"] = data_features
    return data

# TODO: вынести в препроцессинг
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        # Под нулевым индексом в словаре word_idx_map - пустое слово.
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


if __name__ == "__main__":
    start_time = time.time()
    train_and_test_cross_valid(n_epochs=10, n_folds=10, non_static=False)
    print("--- %s seconds ---" % (time.time() - start_time))