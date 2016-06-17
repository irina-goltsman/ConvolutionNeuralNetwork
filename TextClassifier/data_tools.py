# -*- coding: utf-8 -*-
__author__ = 'irina-goltsman'

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import time
import cPickle
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def get_bag_of_words(data, min_df=1):
    vectorizer = CountVectorizer(min_df=min_df)
    return vectorizer.fit_transform(data)


def get_tfidf(data_counts):
    tfidf_transformer = TfidfTransformer()
    return tfidf_transformer.fit_transform(data_counts)


def check_all_sentences_have_one_dim(X):
    sentence_len = len(X[1])
    for sentence in X:
        if len(sentence) != sentence_len:
            return False
    return True


def words_count(text):
    return len(text.split())


def get_output_name(dataset_name, model_name, max_size=None, output_folder="./prepocessed_data"):
    output = output_folder + '/' +dataset_name + "_" + model_name
    if max_size is not None:
        output += "_" + str(max_size)
    return output


def make_vocab_list(data, min_df=1):
    '''
    :param data: массив текстов
    :param min_df: минимальное количество вхождения слов для его учёта
    :return: словарь, для каждого слова подсчитано число его вхождений в датасете
    '''
    vectorizer = CountVectorizer(min_df=min_df)
    vectorizer.fit(data)
    return vectorizer.vocabulary_


def load_bin_vec(fname, vocab):
    """
    Нет смысла сохранять всю модель Word2Vec, сохраню только те слова, которые встречаются в датасете.
    Возвращает словарь, для каждого слова в нём содержится его векторное представление
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def load_w2v(model_path, vocab):
    try:
        model = Word2Vec.load_word2vec_format(model_path, binary=True)
    except UnicodeDecodeError:
        model = Word2Vec.load(model_path)
    dim = model.layer1_size
    word_vecs = {}
    for word in vocab:
        if word in model:
            word_vecs[word] = model[word]
    return word_vecs, dim


def add_unknown_words(model, vocab, dim, min_df=1, v_min=-0.25, v_max=0.25):
    """
        Расширяем модель неизвестными словами, рандомно докидывая вектора от v_min до v_max
    """
    for word in vocab:
        if word not in model and vocab[word] >= min_df:
            model[word] = np.random.uniform(v_min, v_max, dim)


def get_embedding_matrix(word_vecs, dim):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    :param word_vecs: словарь, который для каждого слова возвращает векторное представление
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W_matrix = np.zeros(shape=(vocab_size + 1, dim), dtype='float32')
    # Это нужно для заполнения нулями предложений до необходимой длины
    W_matrix[0] = np.zeros(dim, dtype='float32')
    i = 1
    for word in word_vecs:
        W_matrix[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W_matrix, word_idx_map


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def preprocess_dataset(model_path, data_path, load_function, output="prepared_data", max_size=None):
    start_time = time.time()
    print "loading data...\n",
    data = load_function(data_path, max_size)
    print "data loaded!"
    print "number of sentences: " + str(len(data))
    data["text"] = data["text"].apply(clean_str)
    vocabulary = make_vocab_list(data["text"])
    print "vocab size: " + str(len(vocabulary))
    print "Word embedding model is loading from %s." % model_path
    word_vec, dim = load_w2v(model_path, vocabulary)
    print "Word embedding model has been loaded."
    print "Word dimensions = %d" % dim
    print "num words already in word2vec: " + str(len(word_vec))

    # Расширяем word embedding модель неизвестными словами, рандомно докидывая вектора -0.25 до 0.25
    add_unknown_words(word_vec, vocabulary, dim)
    # W - матрица всех слов из модели word2vec и word_idx_map - словарь, по слову можно узнать id, чтобы вызвать W[id]
    W_matrix, word_idx_map = get_embedding_matrix(word_vec, dim)
    # Альтернативно можно не использовать word2vec, а просто рандомно инициализировать веса! это матрица W2
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocabulary, dim)
    W2_matrix, _ = get_embedding_matrix(rand_vecs, dim)
    cPickle.dump([data, W_matrix, W2_matrix, word_idx_map, vocabulary], open(output, "wb"))
    print "dataset preprocessed and saved as '%s'" % output
    print("--- %s seconds ---" % (time.time() - start_time))



def add_idx_features(data, word_idx_map, max_l=51, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    data_features = pd.Series([[]], index=data.index)
    for i in data.index:
        data_features[i] = get_idx_from_sent(data["text"][i], word_idx_map, max_l, filter_h)
    data["idx_features"] = data_features
    return data


def get_idx_from_sent(sent, word_idx_map, max_l=51, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        # Под нулевым индексом в словаре word_idx_map - пустое слово.
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def text_to_word_list(text, remove_stopwords):
    text_words = clean_str(text)
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
                # 0.25 и -0.25 - магические числа из кода Кима
                feature_matrix.append(np.random.uniform(low=-0.25, high=0.25, size=model.layer1_size))
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
                # 0.25 и -0.25 - магические числа из кода Кима
                rest = np.random.uniform(low=-0.25, high=0.25, size=(n_rest, matrix.shape[1]))
                matrix = np.append(matrix, rest, axis=0)
            elif strategy == 'zero':
                rest = np.zeros((n_rest, matrix.shape[1]))
                matrix = np.append(matrix, rest, axis=0)
            else:
                raise AttributeError("No %s strategy. Strategy can be 'pass', 'random' or 'zero'.")
        text_data_as_matrix.append(matrix)
    return np.asarray(text_data_as_matrix)
