# -*- coding: utf-8 -*-
import numpy as np
import cPickle
import logging
import time
import data_tools as dt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def load_only_data(data_file):
    print "loading data from %s..." % data_file
    x = cPickle.load(open(data_file, "rb"))
    data = x[0]
    rng = np.random.RandomState(0)
    data.reindex(rng.permutation(data.index))
    return data


def write_results(dataset_name, clf_name, gs_clf, outfile='./report.txt'):
    result_str = list()
    result_str.append('\n')
    result_str.append('dataset name: %s' % dataset_name)
    result_str.append('classifier name: %s' % clf_name)
    result_str.append('best params:')
    result_str.append(str(gs_clf.best_params_))
    result_str.append('best score = %f' % gs_clf.best_score_)
    result_str = '\n'.join(result_str)
    print result_str
    with open(outfile, 'a') as out_f:
        out_f.write(result_str)


def train_and_test_model_cross_valid(data_file, clf, clf_params):
    data = load_only_data(data_file)

    print 'clf name = %s' % clf.__class__.__name__
    # LogisticRegression; MultinomialNB; LinearSVC; SGDClassifier
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('clf', clf)])
    # TODO: можно добавить проход по параметрам предобработки:
    # 'vect__ngram_range': [(1, 2)],
    # 'vect__stop_words': ('english', None),
    # 'vect__min_df': np.arange(1, 10),
    # 'tfidf__sublinear_tf': (True, False),

    gs_clf = GridSearchCV(text_clf, clf_params, n_jobs=2, cv=10, refit=False, verbose=3)
    gs_clf.fit(data["text"], data["label"])
    write_results(data_file, clf.__class__.__name__, gs_clf)


def train_and_test_models_cross_valid(model_name, dataset_names, classifiers):
    SGDClassifier_params = {
        'clf__alpha': np.arange(1e-5, 2e-4, 1e-5),
        'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
                      'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'),
        'clf__penalty': ('none', 'l2', 'l1', 'elasticnet'),
        'clf__fit_intercept': (False, True)
        # 'clf__learning_rate': ('constant', 'optimal', 'invscaling')
    }

    MultinomialNB_params = {
        'clf__alpha': np.arange(0.1, 2.0, 0.3),
        'clf__fit_prior': (True, False),
    }

    LogisticRegression_params = {
        # 'clf__penalty': ('l2', 'l1'),
        # 'clf__fit_intercept': (False, True),
        'clf__solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag'),
        # 'clf__dual': (True, False),
        'clf__C': np.arange(1.0, 2.0, 0.2)
        # 'clf__multi_class': ('ovr', 'multinomial')
    }

    clf_with_params = {SGDClassifier: SGDClassifier_params,
                       MultinomialNB: MultinomialNB_params,
                       LogisticRegression: LogisticRegression_params}

    for dataset_name in dataset_names:
        if dataset_name in avaliable_datasets:
            print "dataset '%s' is processing..." % dataset_name
        else:
            print "Error: dataset '%s' not avaliable" % dataset_name
            print "List of avaliable datasets: " + str(avaliable_datasets)
            print "Next task will be started.."
            continue

        if model_name in available_models:
            print "model name = %s" % model_name
        else:
            print "Error: model '%s' not avaliable" % model_name
            print "List of avaliable models: " + str(available_models)
            print "Next task will be started.."
            continue

        data_file = dt.get_output_name(dataset_name, model_name)
        for clf in classifiers:
            try:
                params = clf_with_params[clf]
            except KeyError:
                params=[]
            try:
                train_and_test_model_cross_valid(data_file, clf(), params)
            except:
                print "Exception was catched! Next task will be started.."
                continue


avaliable_datasets = ("twitter", "mr_kaggle", "polarity", "20_news")
available_models = ("mr_100", "google_300")

if __name__ == "__main__":
    start_time = time.time()
    train_and_test_models_cross_valid(model_name = "google_300", dataset_names =("twitter",),
                                      classifiers=(LogisticRegression,))
    print("--- %s seconds ---" % (time.time() - start_time))
