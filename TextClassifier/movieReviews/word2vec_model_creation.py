__author__ = 'irina'
# -*- coding: utf-8 -*-

import re
import logging
import datetime

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords # Import the stop word list

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return " ".join( meaningful_words )


def text_to_wordlist(text, remove_stopwords=False):
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 5. Return a list of words
    return words


def make_feature_matrix(words, model):
    # feature_matrix = np.zeros((len(words), num_features), dtype="float32")
    feature_matrix = []
    # counter = 0.
    for word in words:
        if word in model.vocab:
            feature_matrix.append(list(model[word]))
            # feature_matrix[counter] = model[word]
            # counter += 1
        # else:
        #    print 'word', word, 'is not in a model\n'
    feature_matrix = np.array(feature_matrix)
    return feature_matrix


def review_to_wordlist( review, remove_stopwords=False ):
    """
    Function to convert a document to a sequence of words,
    optionally removing stop words.  Returns a list of words.
    """
    # Remove HTML
    review_text = BeautifulSoup(review).get_text()
    return text_to_wordlist(review_text, remove_stopwords)


# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords += 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def main():
    print "Loading data..."
    train = pd.read_csv("../data/labeledTrainData.tsv",
                        header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv( "../data/unlabeledTrainData.tsv", header=0,
                                   delimiter="\t", quoting=3 )

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []
    print("Parsing sentences from training set")
    for i, review in enumerate(train["review"]):
        sentences += review_to_sentences(review.decode("utf8"), tokenizer)

    print("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review.decode("utf8"), tokenizer)

    print len(sentences)
    print sentences[0]
    print sentences[1]

    # Set values for various parameters
    num_features = 100    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 8       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training model...")
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    #model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "../models/100features_40minwords_10context" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model.save(model_name)


if __name__ == '__main__':
    nltk.download()  # Download stop words
    main()
