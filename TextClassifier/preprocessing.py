# -*- coding: utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import data_tools as dt
from sklearn.datasets import fetch_20newsgroups
import argparse


def load_polarity_data(data_files, max_size=None):
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
    data = pd.read_csv(data_path, sep=',', nrows=max_size, index_col=0, usecols=[0, 1, 3])
    data.columns = ["label", "text"]
    return data


def load_20_news_data(data_path=None, max_size=None):
    newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=100,
                                    remove=('headers', 'footers', 'quotes'))
    data = pd.DataFrame({"text" : newsgroups.data, "label" : newsgroups.target})
    if max_size is not None:
        data = data[0:max_size]
    return data


def load_DBpedia_data(data_path, max_size=None):
    data = pd.read_csv(data_path, sep=',', nrows=max_size, usecols=[2])[4:]
    print data


def examine_dataset(data_path, load_function):
    data = load_function(data_path)
    print "data loaded"
    print "number of sentences: " + str(len(data))
    print "cleaning..."
    data["text"] = data["text"].apply(dt.clean_str)
    print "cleaning finished"
    print "vocab list creation..."
    vocabulary = dt.make_vocab_list(data["text"])
    print "vocab size: " + str(len(vocabulary))

    data["length"] = data["text"].apply(dt.words_count)
    print "max length of text = %d words" % max(data['length'])
    print "min lenght of text = %d words" % min(data['length'])
    print "mean lenght of text = %d words" % np.mean(data['length'])
    print data.describe(percentiles=[.25, .5, .75, .8, .9, .95, .99])

models = {"mr_100": "./models/100features_40minwords_10context",
          "google_300": "./models/GoogleNews-vectors-negative300.bin"}

data_files = {"twitter": "./data/tweets/Sentiment Analysis Dataset.csv",
              "mr_kaggle": "./data/MR_kaggle/labeledTrainData.tsv",
              "polarity": ["./data/rt-polaritydata/rt-polarity.pos",
                           "./data/rt-polaritydata/rt-polarity.neg"],
              "20_news": None,
              "DBpedia": "./data/DBpedia/Satellite.csv"}

loaders = {"twitter": load_twitter_data,
           "mr_kaggle": load_reviews_data,
           "polarity": load_polarity_data,
           "20_news": load_20_news_data,
           "DBpedia": load_DBpedia_data}

# --data_path=./data/tweets/Sentiment\ Analysis\ Dataset.csv --dataset_name=twitter
# --model_path=../../hdfs/GoogleNews-vectors-negative300.bin
# --output_path=../../hdfs/preprocessed_data/twitter_google_300
if __name__ == "__main__":
    max_size = None
    model_name = "google_300"
    dataset_name = "twitter"
    parser = argparse.ArgumentParser(description='Preprocess given dataset.')
    parser.add_argument("--max_size", type=int, default=max_size, help='Max number of rows should be processed.')
    parser.add_argument("--dataset_name", type=str, default=dataset_name, help='Dataset short name.')
    parser.add_argument("--model_path", type=str, default=models[model_name], help='Path to word embedding model.')
    parser.add_argument("--data_path", type=str, default=data_files[dataset_name], help='Path to dataset.')
    parser.add_argument("--output_path", type=str, default=dt.get_output_name(dataset_name, model_name, max_size),
                        help='Full path to output file.')
    args = vars(parser.parse_args())
    print args

    # dt.preprocess_dataset(model_path=args['model_path'], data_path=args['data_path'],
    #                       load_function=loaders[args['dataset_name']],
    #                       output=args['output_path'], max_size=args['max_size'])

    examine_dataset(args['data_path'], load_function=loaders[args['dataset_name']])
