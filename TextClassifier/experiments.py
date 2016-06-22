# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cPickle
import logging
import time
import datetime
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from lasagne.updates import adadelta, adam, adagrad
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
    data, w2v_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3]
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
    data, w2v_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3]
    print "data loaded!"
    print "%d samples." % len(data)
    print w2v_matrix


def train_and_test_cross_valid(data_file, n_epochs, non_static, batch_size, k_top, n_filters, windows, activations,
                               early_stop, valid_frequency, seed, word_dimentions,
                               dropout, l1_regs, n_hidden, update_finction):
    print "loading data...",
    x = cPickle.load(open(data_file, "rb"))
    data, w2v_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3]
    print "data loaded!"
    print "%d samples." % len(data)
    if word_dimentions is None:
        word_vect = w2v_matrix
        word_dimentions = len(word_vect[0])
    else:
        word_vect = None

    print "word's dimentions = %d" % word_dimentions
    counts = data["text"].apply(dt.words_count)
    max_l = max(counts)
    print "max length of text = %d words" % max_l
    print "min lenght of text = %d words" % min(counts)
    data = dt.add_idx_features(data, word_idx_map, max_l=max_l, filter_h=5)

    assert dt.check_all_sentences_have_one_dim(data["idx_features"])
    sentence_len = len(data["idx_features"][0])
    n_out = max(data["label"])+1

    clf = CNNTextClassifier(vocab_size=len(w2v_matrix), word_embedding=word_vect,
                            word_dimension=word_dimentions, sentence_len=sentence_len, n_hidden=n_hidden,
                            windows=windows, n_filters=n_filters, k_top=k_top, activations=activations,
                            batch_size=batch_size, non_static=non_static, dropout=dropout, l1_regs=l1_regs,
                            seed=seed, n_out=n_out)

    print clf.get_params_as_string()

    kf = KFold(len(data), n_folds=10, shuffle=True, random_state=100)
    results = cross_val_score(clf, data["idx_features"], data["label"], cv=kf, n_jobs=1,
                              fit_params={'early_stop': early_stop, 'valid_frequency': valid_frequency,
                                          'n_epochs': n_epochs, 'update_function': update_finction})
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


def train_and_save_model(clf_name, data_file, n_epochs, non_static, batch_size, k_top, n_filters,
                         windows, activations, early_stop, valid_frequency, seed,
                         word_dimentions, dropout, l1_regs, n_hidden, update_finction):
    print "loading data...",
    x = cPickle.load(open(data_file, "rb"))
    try:
        data, w2v_matrix, word_idx_map, vocab = x[0], x[1], x[2], x[3]
    except IndexError:
        data = x[0]
        assert word_dimentions is not None
    print "data loaded!"
    print "%d samples." % len(data)
    if word_dimentions is None:
        word_vect = w2v_matrix
        word_dimentions = len(word_vect[0])
    else:
        word_vect = None

    print data["text"][1]

    print "word's dimentions = %d" % word_dimentions
    counts = data["text"].apply(dt.words_count)
    max_l = max(counts)
    print "max length of text = %d words" % max_l
    print "min lenght of text = %d words" % min(counts)
    data = dt.add_idx_features(data, word_idx_map, max_l=max_l, filter_h=5)

    assert dt.check_all_sentences_have_one_dim(data["idx_features"])
    sentence_len = len(data["idx_features"][1])
    n_out = max(data["label"])+1

    clf = CNNTextClassifier(clf_name=clf_name, vocab_size=len(w2v_matrix), word_embedding=word_vect,
                            word_dimension=word_dimentions, sentence_len=sentence_len, n_hidden=n_hidden,
                            windows=windows, n_filters=n_filters, k_top=k_top, activations=activations,
                            batch_size=batch_size, non_static=non_static, dropout=dropout, l1_regs=l1_regs,
                            seed=seed, n_out=n_out)

    print clf.get_params_as_string()

    try:
        clf.fit(data["idx_features"], data["label"], early_stop=early_stop, valid_frequency=valid_frequency,
                n_epochs=n_epochs, update_function=update_finction)
    except:
        # save_model(clf)
        raise
    save_model(clf)


def test_on_binary_sentiment(data_path, clf_name, n_epochs, batch_size, non_static, early_stop,
                             k_top, n_filters, windows, seed, word_dimentions, activations, dropout,
                             valid_frequency, update_finction, l1_regs=tuple(), l2_regs=tuple()):
    # 6920 - тренировочная выборка, 872 - валидац., 1821 - тестовая, vocabulary size = 15448
    # тренировочная выборка, однако, в 23 раза больше - Kalchbrener разбил выборку на
    x_train_ids, y_train, train_lens = dt.read_and_sort_matlab_data(data_path+"train.txt",
                                                                           data_path+"train_lbl.txt")
    x_valid_ids, y_valid, valid_lens = dt.read_and_sort_matlab_data(data_path + "valid.txt",
                                                                     data_path + "valid_lbl.txt")
    x_test_ids, y_test, test_lens = dt.read_and_sort_matlab_data(data_path + "test.txt",
                                                                        data_path + "test_lbl.txt")

    assert dt.check_all_sentences_have_one_dim(x_train_ids)
    sentence_len = len(x_train_ids[1])
    n_out = max(y_train)+1
    clf = CNNTextClassifier(clf_name=clf_name, vocab_size=15449, word_embedding=None,
                            word_dimension=word_dimentions, sentence_len=None,
                            windows=windows, n_filters=n_filters, k_top=k_top, activations=activations,
                            batch_size=batch_size, non_static=non_static, dropout=dropout,
                            l1_regs=l1_regs, l2_regs=l2_regs,seed=seed, n_out=n_out)

    print clf.get_params_as_string()
    try:
        clf.fit(x_train_ids, y_train, x_valid_ids, y_valid, x_test_ids, y_test,
                train_lens=train_lens, valid_lens=valid_lens, test_lens=test_lens,
                early_stop=early_stop, valid_frequency=valid_frequency,
                n_epochs=n_epochs, update_function=update_finction)
    except:
        # save_model(clf)
        raise
    save_model(clf)

# avaliable_datasets = ("twitter", "mr_kaggle", "polarity", "20_news")
# available_models = ("mr_100", "google_300")

if __name__ == "__main__":
    start_time = time.time()

    # test_on_binary_sentiment(data_path='./data/binarySentiment/', clf_name='dcnn',
    #                          n_epochs=50, batch_size=4, non_static=True, early_stop=False,
    #                          k_top=4, n_filters=(6, 14), windows=((7,), (5,)), seed=0, word_dimentions=48,
    #                          activations=('tanh', 'tanh'), dropout=0.5, valid_frequency=20,
    #                          l2_regs=(0.0001 / 2, 0.00003 / 2, 0.000003 / 2, 0.0001 / 2),
    #                          update_finction=adadelta)

    # test_on_binary_sentiment(data_path='./data/binarySentiment/', clf_name='1cnn',
    #                          n_epochs=100, batch_size=50, non_static=True, early_stop=True, valid_frequency=50,
    #                          k_top=1, n_filters=(100,), windows=((3, 4),), seed=0,
    #                          word_dimentions=30, activations=('relu',), dropout=0.2,
    #                          l1_regs=(0.00001, 0.00003, 0.000003, 0.0001),
    #                          update_finction=adam)

    print("--- %s seconds ---" % (time.time() - start_time))

    max_size = None
    model_name = "mr_100"
    dataset_name = "amazon"
    # train_and_save_model(clf_name='dcnn', data_file=dt.get_output_name(dataset_name, model_name),
    #                      n_epochs=40, batch_size=50, non_static=True, early_stop=False,
    #                      k_top=4, n_filters=(20, 20), windows=((7,), (5,)), seed=0, word_dimentions=40,
    #                      activations=('iden', 'relu'), dropout=0.5, valid_frequency=20,
    #                      L1_regs=(0.00001, 0.00003, 0.000003, 0.0001), n_hidden=100)
    # #
    train_and_save_model(clf_name='1cnn', data_file=dt.get_output_name(dataset_name, model_name, max_size),
                         n_epochs=3, batch_size=50, non_static=True, early_stop=True, valid_frequency=20,
                         k_top=1, n_filters=(100,), windows=((3, 4),), seed=0, update_finction=adam,
                         word_dimentions=40, activations=('relu',), dropout=0.2,
                         l1_regs=(0.00001, 0.00001, 0.00001, 0.0001, 0.0001), n_hidden=100)

    # load_and_print_params("./cnn_states/state_2016-06-12-19:53:52")
    # continue_training(path_to_model="./cnn_states/state_2016-06-18-20:03:13",
    #                   data_file=dt.get_output_name(dataset_name, model_name),
    #                   early_stop=False, valid_frequency=5, n_epochs=50)
    # look_at_vec_map(data_file=get_output_name(dataset_name, model_name))

    print("--- %s seconds ---" % (time.time() - start_time))
