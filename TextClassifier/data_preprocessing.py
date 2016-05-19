__author__ = 'irina-goltsman'
# -*- coding: utf-8 -*-

import numpy as np
import re
from nltk.corpus import stopwords


def text_to_word_list(text, remove_stopwords):
    #TODO: можно попробовать включить 0-9(),!?
    text_words = re.sub("[^a-zA-Z]", " ", text)
    words = text_words.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

# может вернуть None!
def make_feature_matrix(words, model, strategy):
    '''
    :param words: list of words
    :param model: word embedding model
    :param strategy:  'pass', 'random' or 'zero'
    :return: feature_matrix
    '''
    assert len(words) > 0
    # feature_matrix = np.zeros((len(words), num_features), dtype="float32")
    feature_matrix = []
    for word in words:
        if word in model.vocab:
            feature_matrix.append(model[word])
        else:
            if strategy == 'pass':
                continue
            elif strategy == 'random':
                # TODO: тут что-то очень плохо, визуализируй веса!
                feature_matrix.append(np.random.uniform(low=-1.0, high=1.0, size=model.layer1_size))
            elif strategy == 'zero':
                feature_matrix.append(np.zeros(model.layer1_size))
            else:
                raise AttributeError("No %s strategy. Strategy can be 'pass', 'random' or 'zero'.")

    if len(feature_matrix) == 0:
        return None
    feature_matrix = np.asarray(feature_matrix)
    if len(feature_matrix) == 1:
        feature_matrix = feature_matrix.reshape(1, feature_matrix.shape[0])
    return feature_matrix

# может вернуть None!
def text_to_matrix(text, model, strategy, remove_stopwords):
    '''
    :param text: string
    :param model: word embedding model
    :param strategy: 'pass', 'random' or 'zero'
    :return: text matrix
    '''
    words = text_to_word_list(text, remove_stopwords)
    if len(words) == 0:
        return None
    matrix = make_feature_matrix(words, model, strategy)
    return matrix


def feature_selection(text_data, word_embedding_model, min_len=None,
                      strategy='pass', remove_stopwords=False):
    '''
    :param text_data: iterable array of texts
    :param word_embedding_model: модель векторного представления слов
    :param min_len: минимальная длина  конечного представления текста
    :param strategy: стратегия для заполнения предложений до необходимого размера
                     доступные стратегии:
                     'pass' - не записывать предложение, с размером меньше заданного
                     'random' - заполнение слов случайными значениями
                     'zero' - заполнение слов нулями
    :param remove_stopwords: если True - то производить удаление стоп-слов.
    :return:
    '''
    text_data_as_matrix = []
    for i, text in enumerate(text_data):
        if not isinstance(text, str) and not isinstance(text, np.unicode):
            print type(text)
            raise AttributeError("feature selection error: not string format")
        # text_to_matrix может вернуть None!
        matrix = text_to_matrix(text, word_embedding_model, strategy, remove_stopwords)
        if matrix is None:
            print "Warning: {0}: '{1}' hasn't meaningful words!".format(i, text)
            #TODO: что делать в этом случае?
            continue
        elif (min_len is not None) and (matrix.shape[0] < min_len):
            print "Warning: {0} sentence's length ({1}) is less then min_len ({2})" \
                  .format(i, matrix.shape[0], min_len)
            print "sentence: {}".format(text)
            n_rest = min_len - matrix.shape[0]
            if strategy == 'pass':
                continue
            elif strategy == 'random':
                rest = np.random.uniform(low=-1.0, high=1.0, size=(n_rest, matrix.shape[1]))
                matrix = np.append(matrix, rest, axis=0)
            elif strategy == 'zero':
                rest = np.zeros((n_rest, matrix.shape[1]))
                matrix = np.append(matrix, rest, axis=0)
            else:
                raise AttributeError("No %s strategy. Strategy can be 'pass', 'random' or 'zero'.")
        text_data_as_matrix.append(matrix)
    return np.asarray(text_data_as_matrix)