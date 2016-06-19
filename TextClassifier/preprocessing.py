# -*- coding: utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import data_tools as dt
from sklearn.datasets import fetch_20newsgroups
import argparse


def load_polarity_data(data_path, max_size=None):
    data_files = [data_path + "rt-polarity.pos", data_path + "rt-polarity.neg"]
    data = []
    for label in [0, 1]:
        with open(data_files[label], "rb") as f:
            for line in f:
                data.append([line.strip(), label])
    data = pd.DataFrame(data, columns=["text", "label"])
    data = data.reindex(np.random.permutation(data.index))
    if max_size is not None:
        data = data[0:max_size]
    return data


def translation(text):
    review_text = BeautifulSoup(text, "lxml").get_text()
    if review_text != '':
        return review_text
    else:
        print "bad change!"
        return text


def load_reviews_data(data_path, max_size=None):
    data = pd.read_csv(data_path, header=0, delimiter="\t", quoting=3, nrows=max_size,
                       index_col=0)
    data.columns = ["label", "text"]

    print "Translating reviews to raw text format..."
    data["text"] = data["text"].apply(translation)
    return data


def load_twitter_data(data_path, max_size=None):
    data = pd.read_csv(data_path, sep=',', nrows=max_size, index_col=0, usecols=[0, 1, 3], error_bad_lines=False)
    data.columns = ["label", "text"]
    return data


def load_20_news_data(data_path=None, max_size=None):
    newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=100,
                                    remove=('headers', 'footers', 'quotes'))
    data = pd.DataFrame({"text" : newsgroups.data, "label" : newsgroups.target})
    if max_size is not None:
        data = data[0:max_size]
    return data


# TODO:
def load_dbpedia_data(data_path, max_size=None):
    data = pd.read_csv(data_path, sep=',', nrows=max_size, usecols=[2])[4:]
    print data


def load_binary_sentiment(data_path, max_size=None):
    # 6920 - тренировочная выборка, 872 - валидац., 1821 - тестовая
    # тренировочная выборка, однако, в 23 раза больше - Kalchbrener разбил выборку на
    train_x_indexes, train_y, train_lengths = dt.read_and_sort_matlab_data(data_path+"train.txt",
                                                                           data_path+"train_lbl.txt")
    dev_x_indexes, dev_y, dev_lengths = dt.read_and_sort_matlab_data(data_path + "valid.txt",
                                                                     data_path + "valid_lbl.txt")
    test_x_indexes, test_y, test_lengths = dt.read_and_sort_matlab_data(data_path + "test.txt",
                                                                        data_path + "test_lbl.txt")
    print train_x_indexes.shape
    print dev_x_indexes.shape
    print test_x_indexes.shape
    # for i in xrange(1, 57):
    #     print train_lengths.count(i)
    # for i in xrange(15441):
    #     print "sentence: '%s', label: %d" % (str(train_x_indexes[i][0]), train_y[i])


def examine_dataset(data_path, load_function):
    data = load_function(data_path)
    print "data loaded"
    print "number of sentences: " + str(len(data))
    print "example of data: " + data["text"][1]
    print "cleaning..."
    data["cleared_text"] = data["text"].apply(dt.clean_str)
    print "cleaning finished"
    print "example of cleared data: " + data["cleared_text"][1]
    print "vocab list creation..."
    vocabulary = dt.make_vocab_list(data["text"])
    print "vocab size: " + str(len(vocabulary))

    data["length"] = data["text"].apply(dt.words_count)
    print "max length of text = %d words" % max(data['length'])
    print "min lenght of text = %d words" % min(data['length'])
    print data.describe(percentiles=[.25, .5, .75, .8, .9, .95, .99])


models = {"mr_100": "./models/100features_40minwords_10context",
          "google_300": "./models/GoogleNews-vectors-negative300.bin"}

data_files = {"twitter": "./data/tweets/Sentiment Analysis Dataset.csv",
              "mr_kaggle": "./data/MR_kaggle/labeledTrainData.tsv",
              "polarity": "./data/rt-polaritydata/",
              "20_news": None,
              "dbpedia": "./data/DBpedia/Satellite.csv",
              "bin_sent": "./data/binarySentiment/"}

loaders = {"twitter": load_twitter_data,
           "mr_kaggle": load_reviews_data,
           "polarity": load_polarity_data,
           "20_news": load_20_news_data,
           "dbpedia": load_dbpedia_data,
           "bin_sent": load_binary_sentiment}


# --data_path=./data/tweets/Sentiment\ Analysis\ Dataset.csv --dataset_name=twitter
# --model_path=../../hdfs/GoogleNews-vectors-negative300.bin
# --output_path=../../hdfs/preprocessed_data/twitter_google_300
if __name__ == "__main__":
    max_size = None
    model_name = None
    parser = argparse.ArgumentParser(description='Preprocess given dataset.')
    parser.add_argument("--max_size", type=int, default=max_size, help='Max number of rows should be processed.')
    parser.add_argument("--dataset_name", type=str, default="mr_kaggle", help='Dataset short name.')
    parser.add_argument("--model_path", type=str, default=models[model_name] if model_name is not None else None,
                        help='Path to word embedding model.')
    parser.add_argument("--data_path", type=str, default=None, help='Path to dataset.')
    parser.add_argument("--output_path", type=str, default=None, help='Full path to output file.')
    parser.add_argument("--examine", type=bool, default=False, help='Full path to output file.')
    args = vars(parser.parse_args())

    if args['data_path'] is None:
        args['data_path'] = data_files[args['dataset_name']]

    load_binary_sentiment(args['data_path'])

    if args['examine']:
        print args
        examine_dataset(args['data_path'], load_function=loaders[args['dataset_name']])

    else:
        if args['output_path'] is None:
            args['output_path'] = dt.get_output_name(args['dataset_name'], model_name, max_size)
        print args
        dt.preprocess_dataset(model_path=args['model_path'], data_path=args['data_path'],
                              load_function=loaders[args['dataset_name']],
                              output=args['output_path'], max_size=args['max_size'])

