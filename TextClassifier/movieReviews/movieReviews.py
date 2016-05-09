__author__ = 'irina'
# -*- coding: utf-8 -*-

import logging
import time
import datetime
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.cross_validation import KFold

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import sys
sys.path.insert(0, '..')
import CNNTextClassifier


def simple_load_and_test(path_to_model):
    print "Loading data..."
    test_data = pd.read_csv("../data/testData.tsv",
                            header=0, delimiter="\t", quoting=3)
    print "size of test data = %d" % test_data.shape[0]

    max_count = test_data.shape[0]
    print "number of samples = %d" % max_count
    test_data = test_data[0:max_count]

    print "Translating reviews to raw text format..."
    for i in xrange(len(test_data["review"])):
        review_text = BeautifulSoup(test_data["review"][i], "lxml").get_text()
        if review_text != '':
            test_data["review"][i] = review_text
        else:
            print "bad change!"

    classifier = CNNTextClassifier.CNNTextClassifier()
    print "Loading state for classifier..."
    classifier.load(path_to_model)

    print "Prediction..."
    result = classifier.predict(test_data["review"])
    result = np.array(result)
    result = result.flatten(1)
    # Write the test results
    output = pd.DataFrame(data={"id": test_data["id"], "sentiment": result})
    new_state_path = "../output/cnn_word2vec_test" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
    output.to_csv(new_state_path, index=False, quoting=3)


def to_train(path_to_model=None):
    print "Loading data..."
    train = pd.read_csv("../data/labeledTrainData.tsv",
                        header=0, delimiter="\t", quoting=3)

    np.random.seed(seed=0)
    train.reindex(np.random.permutation(train.index))

    print "size of train data = %d" % (train.shape[0])

    max_count = train.shape[0]
    print "number of samples = %d" % max_count
    train = train[0:max_count]

    train_length = int(max_count * 0.90)
    print "Length of train data == %d" % (train_length)
    print "Length of valid data == %d" % (max_count - train_length)

    print "Translating reviews to raw text format..."
    for i in xrange(len(train["review"])):
        review_text = BeautifulSoup(train["review"][i], "lxml").get_text()
        if review_text != '':
            train["review"][i] = review_text
        else:
            print "bad change!"
    classifier = CNNTextClassifier.CNNTextClassifier(learning_rate=0.1, seed=0, L2_reg=0, window=6, n_filters=50,
                                                     k_max=1, activation='iden',
                                                     word_dimension=100,
                                                     model_path="../models/100features_40minwords_10context")
                                                     #model_path="../models/GoogleNews-vectors-negative300.bin")
    if path_to_model:
        print "Loading state for classifier..."
        classifier.load(path_to_model)

    y_train = np.array(train["sentiment"][0:train_length], dtype='int32')
    x_train = np.array(train["review"][0:train_length])

    y_valid = np.array(train["sentiment"][train_length:], dtype='int32')
    x_valid = np.array(train["review"][train_length:])

    print "Fitting a cnn to labeled training data..."
    try:
        classifier.fit(x_train, y_train, x_valid, y_valid, n_epochs=30)
    except:
        new_state_path = "../models/cnn_state_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print "Saving state to '%s'..." % new_state_path
        classifier.save_state(new_state_path)
        raise

    new_state_path = "../models/cnn_state_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print "Saving state to '%s'..." % new_state_path
    classifier.save_state(new_state_path)
    return new_state_path


def train_and_test_cross_folds(max_count=None, n_epochs=15):
    print "Loading data..."
    train = pd.read_csv("../data/labeledTrainData.tsv",
                        header=0, delimiter="\t", quoting=3)

    print "size of train data = %d" % (train.shape[0])

    if not max_count:
        max_count = train.shape[0]
    print "number of samples = %d" % max_count

    train = train[0:max_count]

    print "Translating reviews to raw text format..."
    for i in xrange(max_count):
        review_text = BeautifulSoup(train["review"][i], "lxml").get_text()
        if review_text != '':
            train["review"][i] = review_text
        else:
            print "bad change!"

    n_folds = 10

    kf = KFold(max_count, n_folds=n_folds, shuffle=True, random_state=0)
    results = []
    for num, (train_index, test_index) in enumerate(kf):
        print "num of fold = %d" % num
        X_train = train["review"][train_index].reset_index(drop=True)
        X_test = train["review"][test_index].reset_index(drop=True)
        y_train = train["sentiment"][train_index].reset_index(drop=True)
        y_test = train["sentiment"][test_index].reset_index(drop=True)

        classifier = CNNTextClassifier.CNNTextClassifier(learning_rate=0.1, seed=0, L2_reg=0.1, window=5, n_filters=50,
                                                         k_max=1, activation='iden',
                                                         word_dimension=100,
                                                         model_path="../models/100features_40minwords_10context")

        print "Fitting a cnn to labeled training data..."
        try:
            mean_test_loss, test_score = classifier.fit(X_train, y_train, X_test, y_test, n_epochs=n_epochs)
            results.append((mean_test_loss, test_score))
        except:
            if len(results) > 0:
                new_results_path = "../results/results_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')\
                                 + '_' + str(num)
                print "Saving results to '%s'..." % new_results_path
                with open(new_results_path, 'w') as f:
                    f.write(str(results))
            raise

    results = np.array(results)
    losses = results[:, 0]
    scores = results[:, 1]
    print "-----------------------------------------------------"
    print "------MEAN TEST LOSSES = %f ---- MEAN TEST SCORE = %f-----" % (np.mean(losses), np.mean(scores))
    print "-----------------------------------------------------"
    new_results_path = "../results/results_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') \
                       + '_mean'
    print "Saving results to '%s'..." % new_results_path
    with open(new_results_path, 'w') as f:
        f.write(str((np.mean(losses), np.mean(scores))))


def load_model_and_print_cnn_params(path_to_model):
    classifier = CNNTextClassifier.CNNTextClassifier()
    print "Loading state for classifier..."
    classifier.load(path_to_model)
    params = classifier.get_cnn_params()
    print "params of cnn:"
    print params[-2].eval()
    print params[-1].eval()

if __name__ == '__main__':
    start_time = time.time()
    train_and_test_cross_folds(max_count=300, n_epochs=10)
    #load_model_and_print_cnn_params('../models/cnn_state_2016-05-09-16:39:40_0')
    #new_state_path = to_train()
    #simple_load_and_test('../models/' + new_state_path)
    #simple_load_and_test('../models/cnn_state_2016-05-08-22:13:01')
    print("--- %s seconds ---" % (time.time() - start_time))
