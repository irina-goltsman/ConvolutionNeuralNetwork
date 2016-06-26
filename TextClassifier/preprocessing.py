# -*- coding: utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import data_tools as dt
from sklearn.datasets import fetch_20newsgroups
import argparse
import json


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


def load_amazon(data_path, max_size=None):
    x_data = []
    y_data = []
    with open(data_path, 'r') as f:
        count = 0
        for line in f:
            row = json.loads(line)
            text = row['summary'] + '\n' + row['reviewText']
            x_data.append(text)
            y_data.append(row['overall'])
            count += 1
            if count == max_size:
                break
    data = pd.DataFrame({"text": x_data, "label": y_data})
    return data


def load_dbpedia_data(data_path, max_size=None):
    limit = 100000
    if max_size is None:
        max_size = limit
    classes = ('ArchitecturalStructure', 'ChemicalSubstance', 'Event', 'MeanOfTransportation',
               'NaturalPlace', 'Organisation', 'Artist', 'Athlete', 'Species', 'Work')
    dataset = []
    shift = 4
    for id, class_name in enumerate(classes):
        path = data_path + class_name + '.csv'
        # TODO: возможно стоило перемешать и брать случайные из класса.
        data = pd.read_csv(path, sep=',', nrows=max_size/len(classes) * 2 + shift, usecols=[2],
                           na_values='NULL')[shift:]
        data.dropna(inplace=True)
        data.columns = ["text",]
        data = data[data["text"].apply(dt.words_count) > 7]
        np.random.seed(0)
        data = data.apply(np.random.permutation)
        data = data[0:max_size/len(classes)]
        print len(data)
        data['label'] = id
        dataset.append(data)
    return pd.concat(dataset, ignore_index=True)


def examine_dataset(data_path, load_function, max_size):
    data = load_function(data_path, max_size)
    print "data loaded"
    print "number of texts: " + str(len(data))
    print "example of data: " + data["text"][1]
    print "label: %d" % data["label"][1]
    # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # data["text"].apply(sent_detector.tokenize)

    print "cleaning..."
    data["cleared_text"] = data["text"].apply(dt.clean_str)
    print "cleaning finished"
    print "example of cleared data: " + data["cleared_text"][1]
    vocabulary = dt.build_full_dict(data["cleared_text"])
    print "last 10 values of vocabulary dict:"
    print vocabulary.items()[-10:]
    word_idx_map = dt.build_word_idx_map(vocabulary)
    print "vocab size: " + str(len(vocabulary))
    # print "max vocabulary value: %d" % max(word_idx_map.values())

    data["length"] = data["cleared_text"].apply(dt.words_count)
    data["cleared_length"] = data["cleared_text"].apply(dt.words_count)
    print "max length of text = %d words" % max(data['length'])
    print "min lenght of text = %d words" % min(data['length'])
    print "max length of cleared text = %d words" % max(data['cleared_length'])
    print "min lenght of cleared text = %d words" % min(data['cleared_length'])
    print data.describe(percentiles=[.25, .5, .75, .8, .9, .95, .99])
    print "sorted by len data:"
    print data.sort_values(by="length")

models = {"mr_100": "./models/100features_40minwords_10context",
          "google_300": "./models/GoogleNews-vectors-negative300.bin"}

data_files = {"twitter": "./data/tweets/Sentiment Analysis Dataset.csv",
              "mr_kaggle": "./data/MR_kaggle/labeledTrainData.tsv",
              "polarity": "./data/rt-polaritydata/",
              "20_news": None,
              "dbpedia": "./data/DBpedia/",
              "bin_sent": "./data/binarySentiment/",
              "amazon": "./data/amazon/reviews_Beauty.json"}

loaders = {"twitter": load_twitter_data,
           "mr_kaggle": load_reviews_data,
           "polarity": load_polarity_data,
           "20_news": load_20_news_data,
           "dbpedia": load_dbpedia_data,
           "amazon": load_amazon}


# --data_path=./data/tweets/Sentiment\ Analysis\ Dataset.csv --dataset_name=twitter
# --model_path=../../hdfs/GoogleNews-vectors-negative300.bin
# --output_path=../../hdfs/preprocessed_data/twitter_google_300
if __name__ == "__main__":
    max_size = 100000
    model_name = None
    parser = argparse.ArgumentParser(description='Preprocess given dataset.')
    parser.add_argument("--max_size", type=int, default=max_size, help='Max number of rows should be processed.')
    parser.add_argument("--dataset_name", type=str, default="dbpedia", help='Dataset short name.')
    parser.add_argument("--model_path", type=str, default=models[model_name] if model_name is not None else None,
                        help='Path to word embedding model.')
    parser.add_argument("--data_path", type=str, default=None, help='Path to dataset.')
    parser.add_argument("--output_path", type=str, default=None, help='Full path to output file.')
    parser.add_argument("--examine", type=bool, default=False,
                        help='Set True to show the full info (without saving)')
    args = vars(parser.parse_args())

    if args['data_path'] is None:
        args['data_path'] = data_files[args['dataset_name']]

    # load_binary_sentiment(args['data_path'])

    if args['examine']:
        print args
        examine_dataset(args['data_path'], load_function=loaders[args['dataset_name']], max_size=args['max_size'])

    else:
        if args['output_path'] is None:
            args['output_path'] = dt.get_output_name(args['dataset_name'], model_name, args['max_size'])
        print args
        dt.preprocess_dataset(model_path=args['model_path'], data_path=args['data_path'],
                              load_function=loaders[args['dataset_name']],
                              output=args['output_path'], max_size=args['max_size'])

