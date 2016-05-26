# -*- coding: utf-8 -*-
import pandas as pd
import sys
sys.path.insert(0, '..')
import data_preprocessing as dp
from bs4 import BeautifulSoup


def translation(text):
    review_text = BeautifulSoup(text, "lxml").get_text()
    if review_text != '':
        return review_text
    else:
        print "bad change!"
        return text


def load_reviews_data(data_path, max_size=None):
    data = pd.read_csv(data_path, header=0, delimiter="\t", quoting=3)
    data = data.drop("id", 1)
    data.columns = ["label", "text"]
    if max_size is not None:
        data = data[0:max_size]

    print "Translating reviews to raw text format..."
    data["text"] = data["text"].apply(translation)
    return data


if __name__ == "__main__":
    model_path = "../models/GoogleNews-vectors-negative300.bin"
    data_path = "../data/labeledTrainData.tsv"
    output = "mr_prepared_data"
    dp.preprocess_dataset(model_path, data_path, load_reviews_data, output, max_size=500)