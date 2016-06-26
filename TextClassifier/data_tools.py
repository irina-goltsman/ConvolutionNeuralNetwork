# -*- coding: utf-8 -*-
__author__ = 'irina-goltsman'

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import time
import cPickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def prepare_batch(seqs, labels, maxlen=None):
    """
    Вызывается для каждого минибатча.
    Создаёт матрицу для минибатча, заполняя нулями вектора до длины наибольшего предложения в батче
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
    return x, labels


# From theano tutorial
def load_data(path, n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset directory
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    '''
    path = path + '.pkl'
    with open(path, 'rb') as f:
        train_set = cPickle.load(f)
        test_set = cPickle.load(f)

    if maxlen is not None:
        if isinstance(maxlen, float):
            lengths = map(len, train_set)
            maxlen = np.percentile(lengths, maxlen * 100)
            pass
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) >= maxlen:
                l = maxlen - 1
                x = x[0:l]
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test

def read_and_sort_matlab_data(x_file, y_file, padding_value=15448):
    sorted_dict = {}
    x_data = []
    i=0
    file = open(x_file,"r")
    for line in file:
        words = line.split(",")
        result = []
        length=None
        for word in words:
            word_i = int(word)
            if word_i == padding_value and length==None:
                length = len(result)
            result.append(word_i)
        x_data.append(result)

        if length==None:
            length=len(result)

        if length in sorted_dict:
            sorted_dict[length].append(i)
        else:
            sorted_dict[length]=[i]
        i+=1

    file.close()

    file = open(y_file,"r")
    y_data = []
    for line in file:
        words = line.split(",")
        y_data.append(int(words[0])-1)
    file.close()

    new_train_list = []
    new_label_list = []
    lengths = []
    for length, indexes in sorted_dict.items():
        for index in indexes:
            new_train_list.append(x_data[index])
            new_label_list.append(y_data[index])
            lengths.append(length)

    return np.asarray(new_train_list, dtype=np.int32), np.asarray(new_label_list, dtype=np.int32), lengths


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


def get_output_name(dataset_name, model_name=None, max_size=None, output_folder="./preprocessed_data"):
    output = output_folder + '/' + dataset_name
    if model_name is not None:
        output += "_" + model_name
    if max_size is not None:
        output += "_" + str(max_size)
    return output


def load_bin_vec(fname, vocab):
    """
    Нет смысла сохранять всю модель Word2Vec, сохраню только те слова, которые встречаются в датасете.
    Возвращает словарь, для каждого слова в нём содержится его векторное представление, а также размерность слов
    """
    word_vecs = {}
    dim = None
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
                if dim is None:
                    dim = len(word_vecs[word])
            else:
                f.read(binary_len)
    return word_vecs, dim


def load_w2v(model_path, vocab):
    from gensim.models import Word2Vec
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
    string = re.sub(r"[^A-Za-z0-9(),:!?\'\`\^\.]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def build_full_dict(data):
    print 'Building dictionary..',
    wordcount = dict()
    for text in data:
        words = text.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1
    return wordcount


def build_word_idx_map(wordcount):
    counts = wordcount.values()
    unique_words = wordcount.keys()

    # Наиболее популярные слова будут иметь наименьший индекс
    sorted_idx = np.argsort(counts)[::-1]

    word_idx_map = dict()
    for idx, ss in enumerate(sorted_idx):
        # Оставлю символ 0 для padding
        word_idx_map[unique_words[ss]] = idx + 1

    print '%d total words, %d unique words' % (int(np.sum(counts)), len(unique_words))
    return word_idx_map


def preprocess_dataset(data_path, load_function, output="prepared_data", model_path=None, max_size=None):
    start_time = time.time()
    print "loading data...\n",
    data = load_function(data_path, max_size)
    print "data loaded!"
    print "number of samples: " + str(len(data))
    data["text"] = data["text"].apply(clean_str)
    vocabulary = build_full_dict(data["text"])
    print "vocabulary size = %d" % len(vocabulary)

    if model_path is None:
        # TODO: обрезать словарь? выкидывать стоп-слова?
        word_idx_map = build_word_idx_map(vocabulary)
        with open(output, "wb") as f:
            cPickle.dump([data, word_idx_map], f)
        print "dataset without embedding model preprocessed and saved as '%s'" % output
        print("--- %s seconds ---" % (time.time() - start_time))
        return

    print "Word embedding model is loading from %s." % model_path
    word_vec, dim = load_w2v(model_path, vocabulary)
    # word_vec, dim = load_bin_vec(model_path, vocabulary)
    print "Word embedding model has been loaded."
    print "Word dimensions = %d" % dim
    print "num words already in word2vec: " + str(len(word_vec))

    # Расширяем word embedding модель неизвестными словами, рандомно докидывая вектора -0.25 до 0.25
    add_unknown_words(word_vec, vocabulary, dim)
    # W - матрица всех слов из модели word2vec и word_idx_map - словарь, по слову можно узнать id, чтобы вызвать W[id]
    W_matrix, word_idx_map = get_embedding_matrix(word_vec, dim)
    cPickle.dump([data, W_matrix, word_idx_map, vocabulary], open(output, "wb"))
    print "dataset with embedding model preprocessed and saved as '%s'" % output
    print("--- %s seconds ---" % (time.time() - start_time))


def add_idx_features(data, word_idx_map, filter_h=1, max_l=None):
    """
    Transforms sentences into a 2-d matrix.
    """
    data_features = pd.Series([[]], index=data.index)
    for i in data.index:
        data_features[i] = get_idx_from_sent(data["text"][i], word_idx_map, filter_h, max_l)
    data["idx_features"] = data_features
    return data


def get_idx_from_sent(sent, word_idx_map, filter_h=1, max_l=None):
    """
    По умолчанию просто переводит текст в набор индексов, при filter_h > 1 добавляет в начале и в конце нули
    При max_l != None добивает нуждую длину нулями, либо наоборот, обрезает текст
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        # Под нулевым индексом в словаре word_idx_map - пустое слово.
        x.append(0)
    words = sent.split()

    for word in words:
        if (max_l is not None) and (len(x) >= max_l):
            break
        if word in word_idx_map:
            x.append(word_idx_map[word])

    if max_l is None:
        for i in xrange(pad):
            x.append(0)
    else:
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
