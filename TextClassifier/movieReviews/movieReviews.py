__author__ = 'irina'
# -*- coding: utf-8 -*-

import logging
import time
import datetime
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

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

    classifier = CNNTextClassifier.CNNTextClassifier(model_path="../models/100features_40minwords_10context")
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

    train_length = int(max_count * 0.98)
    print "Length of train data == %d" % (train_length)
    print "Length of valid data == %d" % (max_count - train_length)

    print "Translating reviews to raw text format..."
    for i in xrange(len(train["review"])):
        review_text = BeautifulSoup(train["review"][i], "lxml").get_text()
        if review_text != '':
            train["review"][i] = review_text
        else:
            print "bad change!"

    classifier = CNNTextClassifier.CNNTextClassifier(learning_rate=0.1, seed=0, L2_reg=0.1, window=5, n_filters=15,
                                                     k_max=2,
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

if __name__ == '__main__':
    start_time = time.time()
    to_train()
    simple_load_and_test('../models/cnn_state_2016-05-08-20:09:21')
    print("--- %s seconds ---" % (time.time() - start_time))
