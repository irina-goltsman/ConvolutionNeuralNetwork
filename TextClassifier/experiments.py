# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cPickle
import logging
import time
import datetime
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from data_tools import add_idx_features
from CNNTextClassifier import CNNTextClassifier
from data_tools import get_output_name, words_count, check_all_sentences_have_one_dim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def load_and_print_params(path_to_model):
    clf = CNNTextClassifier()
    print "Loading state for classifier..."
    clf.load(path_to_model)

    print clf.get_params_as_string()
    print "cnn waights values:"
    print clf.get_all_weights_values()


def continue_training(path_to_model, data_file, early_stop, valid_frequency, n_epochs):
    print "loading data...",
    x = cPickle.load(open(data_file, "rb"))
    data, w2v_matrix, random_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    print "%d samples." % len(data)

    max_l = max(data["text"].apply(words_count))
    print "max length of text = %d words" % max_l
    data = add_idx_features(data, word_idx_map, max_l=max_l, filter_h=5)
    print data["idx_features"][0]

    assert check_all_sentences_have_one_dim(data["idx_features"])

    clf = CNNTextClassifier()
    print "Loading state for classifier..."
    clf.load(path_to_model)

    print clf.get_params_as_string()

    # kf = KFold(len(data), n_folds=n_folds, shuffle=True, random_state=0)
    # results = cross_val_score(clf, data["idx_features"], data["label"], cv=kf, n_jobs=1,
    #                           fit_params={'early_stop': early_stop})
    # mean_score = str(np.mean(results))
    try:
        clf.fit(data["idx_features"], data["label"], model_path=path_to_model, early_stop=early_stop,
                valid_frequency=valid_frequency, n_epochs=n_epochs)
    except:
        new_state_path = "./cnn_states/state_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print "Saving state to '%s'..." % new_state_path
        clf.save_state(new_state_path)
        # print clf.get_all_weights_values()
        raise


def look_at_vec_map(data_file):
    print "loading data...",
    x = cPickle.load(open(data_file, "rb"))
    data, w2v_matrix, random_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    print "%d samples." % len(data)
    print w2v_matrix


def train_and_test_cross_valid(data_file, n_epochs, non_static, batch_size, k_top, n_filters, windows,
                               word_embedding, early_stop, valid_frequency, learning_rate, seed):
    print "loading data...",
    x = cPickle.load(open(data_file, "rb"))
    data, w2v_matrix, random_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    print "%d samples." % len(data)
    if word_embedding == "-rand":
        print "using: random vectors"
        word_vect = random_matrix
    elif word_embedding == "-word2vec":
        word_vect = w2v_matrix
        print "using: word2vec vectors"
    else:
        raise Warning("No word_vectors!")
    dim = len(word_vect[0])
    print "word's dim = %d" % dim
    max_l = max(data["text"].apply(words_count))
    print "max length of text = %d words" % max_l
    data = add_idx_features(data, word_idx_map, max_l=max_l, filter_h=5)
    print data["idx_features"][0]

    assert check_all_sentences_have_one_dim(data["idx_features"])
    sentence_len = len(data["idx_features"][0])

    word_dimentions = len(word_vect[0])
    clf = CNNTextClassifier(vocab_size=len(word_vect), word_embedding=word_vect,
                            word_dimension=word_dimentions, sentence_len=sentence_len,
                            windows=windows, n_filters=n_filters, k_top=k_top,
                            batch_size=batch_size, non_static=non_static,
                            learning_rate=learning_rate, n_epochs=n_epochs, seed=seed)

    print clf.get_params_as_string()

    # kf = KFold(len(data), n_folds=n_folds, shuffle=True, random_state=0)
    # results = cross_val_score(clf, data["idx_features"], data["label"], cv=kf, n_jobs=1,
    #                           fit_params={'early_stop': early_stop})
    # mean_score = str(np.mean(results))
    try:
        clf.fit(data["idx_features"], data["label"], early_stop=early_stop, valid_frequency=valid_frequency)
    except:
        new_state_path = "./cnn_states/state_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print "Saving state to '%s'..." % new_state_path
        clf.save_state(new_state_path)
        print clf.get_all_weights_values()
        raise

    # print "mean score:"
    # print mean_score
    #
    # new_results_path = "../results/MR_results_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') \
    #                    + '_mean'
    # print "Saving results to '%s'..." % new_results_path
    # with open(new_results_path, 'w') as f:
    #     result_str = [clf.get_params_as_string(), "scores:", str(results), "mean score:", mean_score]
    #     result_str = '\n'.join(result_str)
    #     f.write(result_str)


if __name__ == "__main__":
    start_time = time.time()
    model_name = "mr_100"
    dataset_name = "mr_kaggle"
    train_and_test_cross_valid(data_file=get_output_name(dataset_name, model_name), n_epochs=40, batch_size=10,
                               non_static=True, early_stop=False, valid_frequency=50, word_embedding="-word2vec",
                               learning_rate=0.1, k_top=4, n_filters=(6, 14), windows=((7,), (5,)), seed=0)
    #
    # load_and_print_params("./cnn_states/state_2016-06-12-19:53:52")
    # continue_training(path_to_model="./cnn_states/state_2016-06-12-23:59:51",
    #                   data_file=get_output_name(dataset_name, model_name),
    #                   early_stop=False, valid_frequency=100, n_epochs=50)
    # look_at_vec_map(data_file=get_output_name(dataset_name, model_name))
    print("--- %s seconds ---" % (time.time() - start_time))
