# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
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
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


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


def train_and_test_model_cross_valid(data_file, clf, clf_params, n_jobs, n_folds):
    print "loading data from %s..." % data_file
    data = cPickle.load(open(data_file, "rb"))[0]
    assert isinstance(data, pd.DataFrame)
    rng = np.random.RandomState(0)
    data.reindex(rng.permutation(data.index))

    print 'clf name = %s' % clf.__class__.__name__
    # LogisticRegression; MultinomialNB; LinearSVC; SGDClassifier
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('clf', clf)])
    # TODO: можно добавить проход по параметрам предобработки:
    # 'vect__ngram_range': [(1, 2)],
    # 'vect__stop_words': ('english', None),
    # 'vect__min_df': np.arange(1, 10),
    # 'tfidf__sublinear_tf': (True, False),

    gs_clf = GridSearchCV(text_clf, clf_params, n_jobs=n_jobs, cv=n_folds, refit=False, verbose=3)
    gs_clf.fit(data["text"], data["label"])
    write_results(data_file, clf.__class__.__name__, gs_clf)


def train_and_test_models_cross_valid(data_files, clf_names, n_jobs=2, n_folds=10):
    SGDClassifier_params = {
        'clf__alpha': np.arange(2e-4, 2e-3, 2e-4),
        'clf__loss': ('hinge', 'squared_hinge',
                      # 'perceptron', 'modified_huber', 'squared_loss'
                      # 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
                      ),
        'clf__penalty': ('none', 'l2', 'l1', 'elasticnet'),
        'clf__fit_intercept': (True,) # False
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

    classifiers = {'SGDClassifier':      (SGDClassifier, SGDClassifier_params),
                   'MultinomialNB':      (MultinomialNB, MultinomialNB_params),
                   'LogisticRegression': (LogisticRegression, LogisticRegression_params)}

    for data_file in data_files:
        for clf_name in clf_names:
            clf, params = classifiers[clf_name]
            try:
                train_and_test_model_cross_valid(data_file, clf(), params, n_jobs, n_folds)
            except:
                # print "Exception was catched! Next task will be started.."
                # continue
                raise


class Cleaner:
    def fit(self, _, __):
        # Заглушка для Pipeline
        return self

    def transform(self, data):
        return map(dt.clean_str, data)


def train_20_news(n_jobs, n_folds):
    from sklearn.datasets import fetch_20newsgroups
    train = fetch_20newsgroups(subset='train', shuffle=False, random_state=100,
                               remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset='test', shuffle=False, random_state=100,
                              remove=('headers', 'footers', 'quotes'))

    x_train = map(dt.clean_str, train.data)
    x_test = map(dt.clean_str, test.data)

    text_clf = Pipeline([
                         # ('clean', Cleaner()),
                         ('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('clf', SGDClassifier(fit_intercept=True, random_state=0))
                         ])

    SGDClassifier_params = {
        'clf__alpha': np.arange(4e-5, 2e-3, 2e-5),
        'clf__loss': ('squared_loss', 'hinge', 'squared_hinge'),
        'clf__penalty': ('l2', 'elasticnet'),
    }

    gs_clf = GridSearchCV(text_clf, SGDClassifier_params, n_jobs=n_jobs, cv=n_folds, refit=True, verbose=3)
    gs_clf.fit(x_train, train.target)

    result_str = list()
    result_str.append('\n')
    result_str.append('best params:')
    result_str.append(str(gs_clf.best_params_))
    result_str.append('best score = %f' % gs_clf.best_score_)
    result_str = '\n'.join(result_str)
    print result_str

    print "test score = " % gs_clf.score(x_test, test.target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline models grid test.')
    parser.add_argument("--data_files", nargs='+', type=str, default=None, help="List of preprocessed data files.")
    parser.add_argument("--clf", nargs='+', type=str, default=('LogisticRegression', 'SGDClassifier', 'MultinomialNB'),
                        help="Name of classifier. Possible values are 'LogisticRegression', 'MultinomialNB', "
                             "'SGDClassifier'")
    parser.add_argument("--n_jobs", type=int, default=2, help="Number of threads to be run. Default 2.")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds for cross-validation. Default 2.")
    args = vars(parser.parse_args())
    data_files = args['data_files']

    if data_files == ['20_news',]:
        train_20_news(n_jobs=args['n_jobs'], n_folds=args['n_folds'])
    else:
        if data_files is None:
            dataset_names = ("20_news", "twitter", "polarity", "mr_kaggle")
            data_files = [dt.get_output_name(dataset_name) for dataset_name in dataset_names]

        print "classifiers:"
        print args['clf']
        print "data_files:"
        print data_files

        start_time = time.time()
        train_and_test_models_cross_valid(clf_names=args['clf'], data_files=data_files,
                                          n_jobs=args['n_jobs'], n_folds=args['n_folds'])
        print("--- %s seconds ---" % (time.time() - start_time))
