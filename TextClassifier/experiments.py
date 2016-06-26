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


def load_and_cut_data(data_file, big_dataset, need_word_idx_map=True):
    print "loading data...",
    # Немного говнокода из-за разной предобработки датасетов с embedding моделью и без:
    with open(data_file, 'rb') as pickle_file:
        x = cPickle.load(pickle_file)
    try:
        data, w2v_matrix, word_idx_map = x[0], x[1], x[2]
    except IndexError:
        data, word_idx_map, w2v_matrix = x[0], x[1], None
    print "data loaded!"

    print "%d samples." % len(data)

    data["counts"] = data["text"].apply(dt.words_count)
    if big_dataset:
        max_l = data["counts"].quantile(0.95)
        print "Dataset will be cut. Text's 0.95 quantile = %d words" % max_l
    else:
        max_l = max(data["counts"])

    print "max length of text = %d words" % max(data["counts"])
    print "min lenght of text = %d words" % min(data["counts"])
    print "idx features creation..."
    data = dt.add_idx_features(data, word_idx_map, filter_h=5, max_l=max_l)
    print "idx features creation finished"

    assert dt.check_all_sentences_have_one_dim(data["idx_features"])

    if need_word_idx_map:
        return data, word_idx_map, w2v_matrix
    else:
        return data


# TODO: проверь функцию.
def save_model(clf, saveto):
    new_state_path = saveto + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print "Saving state to '%s'..." % new_state_path
    clf.save_state(new_state_path)


def save_history(clf, fit_params, dataset_name, history_saveto):
    print "best_valid_score: %f, clf.best_iter_num:%f" % \
          (clf.best_valid_score, clf.best_iter_num)
    new_res_path = '_'.join((history_saveto, dataset_name, clf.params['clf'],
                             datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))

    best = np.array([clf.best_valid_score, clf.best_iter_num])
    np.savez(new_res_path, clf.get_params, clf.get_arch_params, fit_params, best,
             np.asarray(clf.history_val_err), np.asarray(clf.history_train_err))


def train_and_save_model(data_file, big_dataset, fit_params, dataset_name,
                         clf_params=None, architecture_params=None,
                         path_to_model=None, model_saveto=None, history_saveto=None):
    if path_to_model:
        data = load_and_cut_data(data_file, big_dataset, need_word_idx_map=False)

        clf = CNNTextClassifier()
        print "Loading state for classifier..."
        clf.load(path_to_model)
    else:
        assert clf_params is not None
        assert architecture_params is not None
        data, word_idx_map, w2v_matrix = load_and_cut_data(data_file, big_dataset)

        if clf_params['word_dim'] is None:
            clf_params['word_embedding'] = w2v_matrix
            clf_params['word_dim'] = len(w2v_matrix[0])
        else:
            clf_params['word_embedding'] = None

        clf_params['vocab_size'] = len(word_idx_map) + 1
        clf_params['sent_len'] = len(data["idx_features"][1])
        clf_params['n_out'] = max(data["label"]) + 1

        clf = CNNTextClassifier(clf_params, architecture_params)

    print "clf params:"
    print clf.get_clf_init_params()
    print "arch params:"
    print clf.get_arch_params()
    print "fit params:"
    print fit_params

    try:
        clf.fit(x_train=data["idx_features"], y_train=data["label"],
                fit_params=fit_params, path_to_model=path_to_model)
    except:
        if history_saveto:
            print "Exception! Wait for history save... "
            save_history(clf, fit_params,
                         dataset_name=dataset_name, history_saveto=history_saveto)
            print "Done"
        raise

    if model_saveto:
        save_model(clf, model_saveto)
    if history_saveto:
        save_history(clf, dataset_name=dataset_name, history_saveto=history_saveto)


# available_models = ("mr_100", "google_300")
big_dataset = {'polarity': False, 'mr_kaggle': True, 'dbpedia': True,
               '20_news': True, 'twitter': True, 'amazon': True}

if __name__ == "__main__":
    start_time = time.time()

    max_size = None
    model_name = None
    dataset_name = "twitter"

    clf_params = {
                   'clf': 'lstm',
                   'word_dim': 50,  # Если None, и при этом model_name не None,
                                           # то используется размерность переданной embedding матрицы
                   'seed': 0,
                   'batch_size': 50, # может быть None, тогда передать в fit_params
                   'non_static': True, # если True - изменяет веса embedding матрицы в процессе обучения
                 }

    # Параметры, относящиеся непосредственно к архитектуре:
    # В зависимости от модели могут отличаться
    architecture_params = { # смотри networks для этих параметров
                           'n_hidden': 100,
                           'dropout': 0.5,
                          }

    fit_params = {
                   'n_epochs': 15,
                   'valid_freq': 50,
                   'train_score_freq': 1000,
                   'valid_proportion': 0.1,
                   'early_stop': False,

                   'l1_regs': list(),
                   'l2_regs': list(),
                   'update_func': adam,
                 }

    train_and_save_model(data_file=dt.get_output_name(dataset_name, model_name, max_size),
                         big_dataset=big_dataset[dataset_name],
                         clf_params=clf_params, fit_params=fit_params,
                         architecture_params=architecture_params,
                         dataset_name=dataset_name,
                         history_saveto=None, #"./results/res",
                         model_saveto=None, #"./cnn_states/state_"
                         )

    print("--- %s seconds ---" % (time.time() - start_time))
