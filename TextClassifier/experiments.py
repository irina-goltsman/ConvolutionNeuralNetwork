# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cPickle
import logging
import time
import datetime
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from CNNTextClassifier import CNNTextClassifier
import data_tools as dt

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

    max_l = max(data["text"].apply(dt.words_count))
    print "max length of text = %d words" % max_l
    data = dt.add_idx_features(data, word_idx_map, max_l=max_l, filter_h=5)
    print data["idx_features"][0]

    assert dt.check_all_sentences_have_one_dim(data["idx_features"])

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


def train_and_test_cross_valid(data_file, n_epochs, non_static, batch_size, k_top, n_filters, windows, activations,
                               word_embedding, early_stop, valid_frequency, learning_rate, seed, word_dimentions,
                               dropout, L1_regs, n_hidden):
    print "loading data...",
    x = cPickle.load(open(data_file, "rb"))
    data, w2v_matrix, random_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    print "%d samples." % len(data)
    if word_dimentions is not None:
        word_vect = None
    else:
        word_dimentions = len(w2v_matrix[0])
        if word_embedding == "-rand":
            print "using: random vectors"
            word_vect = random_matrix
        elif word_embedding == "-word2vec":
            word_vect = w2v_matrix
            print "using: word2vec vectors"
        else:
            raise Warning("No word_vectors!")

    print "word's dimentions = %d" % word_dimentions
    max_l = max(data["text"].apply(dt.words_count))
    print "max length of text = %d words" % max_l
    data = dt.add_idx_features(data, word_idx_map, max_l=max_l, filter_h=5)

    assert dt.check_all_sentences_have_one_dim(data["idx_features"])
    sentence_len = len(data["idx_features"][0])
    n_out = max(data["label"])+1

    clf = CNNTextClassifier(vocab_size=len(w2v_matrix), word_embedding=word_vect,
                            word_dimension=word_dimentions, sentence_len=sentence_len, n_hidden=n_hidden,
                            windows=windows, n_filters=n_filters, k_top=k_top, activations=activations,
                            batch_size=batch_size, non_static=non_static, dropout=dropout, L1_regs=L1_regs,
                            learning_rate=learning_rate, n_epochs=n_epochs, seed=seed, n_out=n_out)

    print clf.get_params_as_string()

    kf = KFold(len(data), n_folds=10, shuffle=True, random_state=100)
    results = cross_val_score(clf, data["idx_features"], data["label"], cv=kf, n_jobs=1,
                              fit_params={'early_stop': early_stop, 'valid_frequency': valid_frequency})
    mean_score = str(np.mean(results))

    print "mean score:"
    print mean_score

    new_results_path = "../results/results_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') \
                       + '_mean'
    print "Saving results to '%s'..." % new_results_path
    with open(new_results_path, 'w') as f:
        result_str = [clf.get_params_as_string(), "scores:", str(results), "mean score:", mean_score]
        result_str = '\n'.join(result_str)
        f.write(result_str)


def save_model(clf):
    new_state_path = "./cnn_states/state_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print "Saving state to '%s'..." % new_state_path
    clf.save_state(new_state_path)
    # print clf.get_all_weights_values()


def train_and_save_model(clf_name, data_file, n_epochs, non_static, batch_size, k_top, n_filters, windows,
                         activations, word_embedding, early_stop, valid_frequency, learning_rate, seed,
                         word_dimentions, dropout, L1_regs, n_hidden):
    print "loading data...",
    x = cPickle.load(open(data_file, "rb"))
    data, w2v_matrix, random_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    print "%d samples." % len(data)
    if word_dimentions is not None:
        word_vect = None
    else:
        word_dimentions = len(w2v_matrix[0])
        if word_embedding == "-rand":
            print "using: random vectors"
            word_vect = random_matrix
        elif word_embedding == "-word2vec":
            word_vect = w2v_matrix
            print "using: word2vec vectors"
        else:
            raise Warning("No word_vectors!")

    print "word's dimentions = %d" % word_dimentions
    max_l = max(data["text"].apply(dt.words_count))
    print "max length of text = %d words" % max_l
    data = dt.add_idx_features(data, word_idx_map, max_l=max_l, filter_h=5)

    assert dt.check_all_sentences_have_one_dim(data["idx_features"])
    sentence_len = len(data["idx_features"][1])
    n_out = max(data["label"])+1

    clf = CNNTextClassifier(clf_name=clf_name, vocab_size=len(w2v_matrix), word_embedding=word_vect,
                            word_dimension=word_dimentions, sentence_len=sentence_len, n_hidden=n_hidden,
                            windows=windows, n_filters=n_filters, k_top=k_top, activations=activations,
                            batch_size=batch_size, non_static=non_static, dropout=dropout, L1_regs=L1_regs,
                            learning_rate=learning_rate, n_epochs=n_epochs, seed=seed, n_out=n_out)

    print clf.get_params_as_string()

    try:
        clf.fit(data["idx_features"], data["label"], early_stop=early_stop, valid_frequency=valid_frequency)
    except:
        save_model(clf)
        raise
    save_model(clf)


if __name__ == "__main__":
    start_time = time.time()
    model_name = "google_300"
    dataset_name = "twitter"
    # train_and_save_model(clf_name='dcnn', data_file=dt.get_output_name(dataset_name, model_name),
    #                      n_epochs=40, batch_size=20, non_static=True, early_stop=False, valid_frequency=20,
    #                      word_embedding="-word2vec", learning_rate=1.0, k_top=4, n_filters=(6, 14),
    #                      windows=((7,), (5,)), seed=0, word_dimentions=40, activations=('tanh', 'tanh'),
    #                      dropout=0.2, L1_regs=(0.00001, 0.00003, 0.000003, 0.0001), n_hidden=100)
    # #
    train_and_save_model(clf_name='1cnn', data_file=dt.get_output_name(dataset_name, model_name),
                         n_epochs=25, batch_size=50, non_static=True, early_stop=True, valid_frequency=20,
                         word_embedding="-word2vec", learning_rate=0.5, k_top=1, n_filters=(100,),
                         windows=((3, ),), seed=0, word_dimentions=None, activations=('relu',),
                         dropout=0.5, L1_regs=(0.0, 0.00001, 0.00001, 0.00001, 0.0001, 0.0001), n_hidden=100)

    # load_and_print_params("./cnn_states/state_2016-06-12-19:53:52")
    # continue_training(path_to_model="./cnn_states/state_2016-06-12-23:59:51",
    #                   data_file=get_output_name(dataset_name, model_name),
    #                   early_stop=False, valid_frequency=100, n_epochs=50)
    # look_at_vec_map(data_file=get_output_name(dataset_name, model_name))

    print("--- %s seconds ---" % (time.time() - start_time))
