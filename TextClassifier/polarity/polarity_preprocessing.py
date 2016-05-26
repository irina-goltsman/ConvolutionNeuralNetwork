# -*- coding: utf-8 -*-
import pandas as pd
import sys
sys.path.insert(0, '..')
import data_preprocessing as dp
import numpy as np


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


if __name__ == "__main__":
    model_path = "../models/GoogleNews-vectors-negative300.bin"
    data_files = ["../data/rt-polaritydata/rt-polarity.pos", "../data/rt-polaritydata/rt-polarity.neg"]
    output = "mr.p"
    dp.preprocess_dataset(model_path, data_files, load_polarity_data, output)