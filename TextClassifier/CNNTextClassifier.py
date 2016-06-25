__author__ = 'irina-goltsman'
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator
import cPickle as pickle
import theano
from pandas.core.series import Series
# from sklearn.cross_validation import train_test_split
import time
import warnings
from networks import *
from data_tools import check_all_sentences_have_one_dim
from lasagne.random import set_rng

theano.config.exception_verbosity = 'high'
warnings.filterwarnings("ignore", category=DeprecationWarning)

#TODO: заюзать glove вместо/вместе word2vec

# mode=NanGuardMode - для дебага нанов, поняла что [nvcc] fastmath=True - вызывает nan!

# TODO: ================================6 подумать над реализацией character-level сети - 30мин
# TODO: ================================сделать character-level сеть - 2ч ( переоценить время после пункта 6 )

# TODO: передавать вместе с датасетом реальную длину предложения и заменить k-max-pooling на dynamic k-max-pooling
# TODO: распарсить dbpedia, проверить на бейзлайнах

class CNNTextClassifier(BaseEstimator):
    def __init__(self, clf_name='1cnn', vocab_size=None, word_dimension=100, word_embedding=None, non_static=True,
                 batch_size=100, sentence_len=None, n_out=2, k_top=1, seed=0,
                 windows=((5, 6), (4,)), n_filters=(10, 25), n_hidden=10, activations=('tanh', 'tanh'),
                 dropout=0.5, l1_regs = list(), l2_regs=list()):
        """
        :param clf_name: имя сети, доступно: '1cnn', 'dcnn'
        :param vocab_size: размер словаря
        :param word_dimension: размерность слов
        :type word_embedding: matrix 2d or None
        :param word_embedding: матрица, i-ая строка содержит векторное представление i-го слова
                               если None - то матрица заполняется случайными значениями
        :param non_static: если True, то веса word_embedding могут менять в процессе обучения
        :param batch_size: размер пакета
        :param sentence_len: длина входного предложения
        :param n_out: количество классов для классификации
        :param windows: размеры окон для обработки близких друг к другу слов
        :param n_filters: число фильтров для каждого вида окна
        :param n_hidden: число нейронов в скрытом слое
        :type activations: tuple of string, варианты: 'tanh', 'sigmoid', 'relu', 'cappedrelu', 'iden'
        :param activations: виды активационных функций
        :param dropout: параметр для dropout слоя
        :param L1_regs: параметры для регуляризации
        :type k_top: int (>=1)
        :param k_top: параметр для итогового k-max-pooling слоя
        :param learning_rate: темп обучения
        :param n_epochs: количество эпох обучения
        :param seed: начальное значение для генератора случайных чисел
        """
        self._estimator_type = "classifier"
        self.clf_name = clf_name
        self.vocab_size = vocab_size
        self.word_dimension = word_dimension
        self.word_embedding = word_embedding
        self.non_static = non_static
        self.batch_size = batch_size
        self.sentence_len = sentence_len
        self.n_out = n_out
        self.windows = windows
        self.n_filters = n_filters
        self.n_hidden = n_hidden
        self.activations = activations
        self.dropout = dropout
        self.l1_regs = l1_regs
        self.l2_regs = l2_regs
        self.is_ready = False
        self.is_ready_to_train = False
        self.k_top = k_top
        self.seed = seed

    @staticmethod
    def parse_activation(name):
        if name == 'tanh':
            return T.tanh
        elif name == 'sigmoid':
            return T.nnet.sigmoid
        elif name == 'relu':
            return T.nnet.relu
        elif name == 'cappedrelu':
            # TODO: что за магическое число 6?
            return lambda x: T.minimum(x * (x > 0), 6)
        elif name == 'iden':
            return lambda x: x
        else:
            raise NotImplementedError

    @staticmethod
    def get_clf_builder(name):
        clf_builder_dict = {'1cnn': build_1cnn, 'dcnn': build_dcnn, 'lstm': build_lstm}
        try:
            builder = clf_builder_dict[name]
        except KeyError:
            raise NotImplementedError
        return builder

    def ready(self):
        # Матрица входа, размера n_batch * n_sentence
        self.x = T.lmatrix('x')
        # Результирующие классы для каждого предложения в batch
        self.y = T.ivector('y')

        self.rng = np.random.RandomState(self.seed)
        set_rng(self.rng)

        print "CNN building..."
        clf_builder = self.get_clf_builder(self.clf_name)
        self.network = clf_builder(input_var=self.x, batch_size=self.batch_size,
                                   sentence_len=self.sentence_len, vocab_size=self.vocab_size,
                                   word_dimension=self.word_dimension, word_embedding=self.word_embedding,
                                   non_static=self.non_static,
                                   windows=self.windows, k_top=self.k_top, n_filters=self.n_filters,
                                   activations=self.activations, dropout=self.dropout, n_out=self.n_out)

        # ключевое слово deterministic отключает стохастическое поведение, например, убирает dropout
        self.p_y_given_x = lasagne.layers.get_output(self.network, self.x, deterministic=True)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.predict_wrap = theano.function(inputs=[self.x], outputs=self.y_pred, allow_input_downcast=True)
        self.predict_proba_wrap = theano.function(inputs=[self.x], outputs=self.p_y_given_x, allow_input_downcast=True)
        print "CNN building finished"

        self.is_ready = True

    def ready_to_train(self, update_function, **kwargs):
        # zero_vec_tensor = T.vector()
        # # Если non_static, то 0ая компонента матрицы слов могла измениться, а она должна всегда быть нулевым вектором.
        # set_zero = theano.function([zero_vec_tensor], updates=[(self.words, T.set_subtensor(self.words[0, :], zero_vec_tensor))],
        #                            allow_input_downcast=True)

        # Выпишу слои к которым должна применяться L2 регуляризация
        print "Preparing for training..."
        regularizable_layers = []
        for layer in lasagne.layers.get_all_layers(self.network):
            if isinstance(layer, (
                                  CNN.embeddings.SentenceEmbeddingLayer,
                                  CNN.Conv1DLayerSplitted,
                                  lasagne.layers.conv.Conv2DLayer,
                                  lasagne.layers.DenseLayer)):
                regularizable_layers.append(layer)
        print "num of regularizable layers is %d" % len(regularizable_layers)

        train_prediction = lasagne.layers.get_output(self.network, self.x, deterministic=False)
        loss_train = lasagne.objectives.aggregate(lasagne.objectives.categorical_crossentropy(train_prediction, self.y),
                                                  mode='mean')\
                     + lasagne.regularization.regularize_layer_params_weighted(dict(zip(regularizable_layers,
                                                                                        self.l1_regs)),
                                                                               lasagne.regularization.l1)\
                     + lasagne.regularization.regularize_layer_params_weighted(dict(zip(regularizable_layers,
                                                                                        self.l2_regs)),
                                                                               lasagne.regularization.l2)
        all_params = lasagne.layers.get_all_params(self.network)
        updates = update_function(loss_train, all_params, **kwargs)

        self.train_model = theano.function(inputs=[self.x, self.y], outputs=loss_train, updates=updates,
                                           allow_input_downcast=True)

        print "Preparing for training finished"
        self.is_ready_to_train = True

    @staticmethod
    def shared_dataset(X, y, borrow=True):
        """
        Функция представляет датасет в виде shared variables.
        Это позволяет theano скопировать данные на GPU (если код запускается на GPU).
        """
        if isinstance(X, Series):
            X = X.values.tolist()
        assert check_all_sentences_have_one_dim(X)
        shared_x = theano.shared(np.asarray(X), borrow=borrow)
        shared_y = theano.shared(np.asarray(y), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    def score(self, X, y, lengths=None):
        """
        :param X: array-like, shape = [n_samples, n_features]
        :param y: array-like, shape = [n_samples]
        :param lengths: (optional) array-like, shape = [n_samples] - реальные длины текстов
        :return: среднюю точность предсказания
        """
        return T.mean(T.eq(self.predict(X, lengths), y)).eval()

    def loss(self, X, y, lengths=None):
        """
        :param X: array-like, shape = [n_samples, n_features]
        :param y: array-like, shape = [n_samples]
        :param lengths: (optional) array-like, shape = [n_samples] - реальные длины текстов
        :return: кросс-энтропию
        """
        return lasagne.objectives.categorical_crossentropy(self.predict_proba(X, lengths), y)

    @staticmethod
    def extend_to_batch_size(X, y, batch_size, rng, lengths=None):
        """
        Если длина датасета не делится на размер пакета, то происходит
        репликация случайной подвыборки данных для достижения минимально допустимого размера.
        :param X: массив данных
        :param y: значения
        :param batch_size: размер пакета
        :param rng: numpy.random.RandomState
        :return: дополненные данные
        """
        if len(X) % batch_size > 0:
            num_extra_data = batch_size - len(X) % batch_size
            extra_data_indices = rng.permutation(len(X))[:num_extra_data]
            X_extra = [X[i] for i in extra_data_indices]
            y_extra = [y[i] for i in extra_data_indices]
            if lengths is not None:
                lengths_extra = [lengths[i] for i in extra_data_indices]

            if isinstance(X, Series):
                X = X.values.tolist()

            X = np.append(X, X_extra, axis=0)
            y = np.append(y, y_extra, axis=0)
            if lengths is not None:
                lengths = np.append(lengths, lengths_extra, axis=0)

        return X, y, lengths

# TODO: передавать window - максимальное из возможных окон свёртки
    def get_batch(self, X, id, lengths=None, window=4):
        if lengths is not None:
            sent_len = lengths[(id + 1) * self.batch_size - 1] + window
            return X[id * self.batch_size: (id + 1) * self.batch_size, 0:sent_len]
        else:
            return X[id * self.batch_size: (id + 1) * self.batch_size]

    @staticmethod
    def train_test_split(X, y, lengths, test_size, shuffle=True, rng=np.random.RandomState()):
        permutation = rng.permutation(len(X))
        ind_test = permutation[:test_size]
        ind_train = permutation[test_size:]
        if not shuffle:
            ind_test = ind_test.sort()
        if lengths is not None:
            train_lengths = lengths[ind_train]
            test_lengths = lengths[ind_test]
        else:
            train_lengths = None
            test_lengths = None
        return X[ind_train], X[ind_test], y[ind_train], y[ind_test], train_lengths, test_lengths

    # TODO: разбери эту огромную функцию на маленькие читабельные части
    def fit(self, x_train, y_train, x_valid=None, y_valid=None, x_test=None, y_test=None,
            train_lens=None, valid_lens=None, test_lens=None,
            model_path=None, n_epochs=5, validation_part=0.1, valid_frequency=4, early_stop=False,
            update_function=lasagne.updates.adadelta, **kwargs):
        """ Fit model
        :type x_train: 2d массив из int
        :param x_train: входные данные - список из текстов
                        (каждый текст представлен как список номеров слов)
        :type y_train: 1d массив int
        :param y_train: целевые значения для каждого текста
        :param x_valid: можно передать явно валидационную выборку, иначе она будет отщеплена
                        от обучающей,в пропорции validation_part
        :param x_test: ясно, что можно запомнить итерацию, на которой сеть показала наилучший результат
                       на валидационной выборке, переобучить сеть, остановившись на этой итерации
                       и только после этого считать результат на тестовой выборке. Но это долго.
                       Легче сразу передать тестовую выборку и запоминать score на лучших показателях
                       валидационной выборки.
        :param train_lens, valid_lens, test_lens: если не None, то обрезаю пакет по длине последнего
                    предложения (следует использовать для упорядоченного по возрастанию длины входа датасета)
        :type n_epochs: int
        :param n_epochs: количество эпох для обучения
        :type validation_part: float
        :param validation_part: доля тренеровочных данных, которые станут валидационной выборкой
                                (если x_valid передана явно, этот параметр игнорируется)
        :type valid_frequency: int/None
        :param valid_frequency: если не None, то каждые visualization_frequency интераций
                                        будет выводиться результат модели на валидационной выборке
        :type early_stop: bool
        :param early_stop: если True - будет происходить досрочная остановка.
        :param update_function: функция обучения, по умолчанию: lasagne.updates.adadelta
        :param **kwargs: дополнительные параметры для функции обучения
        """
        assert max(y_train) < self.n_out
        assert min(y_train) >= 0
        assert len(x_train) == len(y_train)
        if train_lens is not None:
            assert len(x_train) == len(train_lens)
        if valid_lens is not None:
            assert len(x_valid) == len(valid_lens)
        if test_lens is not None:
            assert len(x_test) == len(test_lens)

        # подготовим CNN
        if not self.is_ready:
            self.ready()

        if not self.is_ready_to_train:
            self.ready_to_train(update_function, **kwargs)

        if isinstance(x_train, Series):
            x_train = x_train.values.tolist()
            x_train = np.asarray(x_train)
        if isinstance(y_train, Series):
            y_train = y_train.values.tolist()
            y_train = np.asarray(y_train)

        x_train, y_train, train_lens = self.extend_to_batch_size(x_train, y_train, self.batch_size,
                                                                 self.rng, train_lens)

        num_batches = len(x_train) / self.batch_size

        if x_valid is None or y_valid is None:
            num_val_batches = int(np.floor(num_batches * validation_part))
            num_train_batches = num_batches - num_val_batches
            x_train, x_valid, y_train, y_valid, train_lens, valid_lens =\
                self.train_test_split(x_train, y_train, train_lens,
                                      test_size=num_val_batches * self.batch_size,
                                      rng=self.rng)
        else:
            num_train_batches = num_batches

        assert len(x_train) % self.batch_size == 0

        print 'x_train.shape = ' + str(x_train.shape)
        print 'x_valid.shape = ' + str(x_valid.shape)
        # x_train, y_train = self.shared_dataset(x_train, y_train)
        # x_valid, y_valid = self.shared_dataset(x_valid, y_valid)

        if model_path is not None:
            self.load(model_path)

        epoch = 0
        best_valid_score, best_iter_num, last_test_score = 0, 0, 0
        # early-stopping parameters
        valid_frequency = min(valid_frequency, num_train_batches - 1)
        print "visualization frequency: %d batches" % valid_frequency
        patience = num_train_batches * 2  # look as this many examples regardless
        patience_increase = 2.0  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is
        stop = False
        while (epoch < n_epochs) and (not stop):
            start_time = time.time()
            epoch += 1
            train_cost = 0
            indices = self.rng.permutation(num_train_batches)
            for cur_idx, real_idx in enumerate(indices):
                iter = (epoch - 1) * num_train_batches + cur_idx
                cur_train_cost = self.train_model(self.get_batch(x_train, real_idx, train_lens),
                                                  self.get_batch(y_train, real_idx))
                train_cost += cur_train_cost

                if iter > 0 and iter % valid_frequency == 0:
                    print "global_iter %d, epoch %d, batch %d, mean train cost = %f" % \
                          (iter, epoch, cur_idx, train_cost / valid_frequency)
                    train_cost = 0
                    valid_score = self.score(x_valid, y_valid, valid_lens)
                    # выведу результаты валидации:
                    print "------------valid score: %f------------" \
                          % (float(valid_score))

                    if early_stop and (valid_score > improvement_threshold * best_valid_score):
                        patience = max(patience, iter * patience_increase)
                        print "new patience = %d" % patience

                    if valid_score > best_valid_score:
                        best_valid_score = valid_score
                        best_iter_num = iter
                        if x_test is not None:
                            last_test_score = self.score(x_test, y_test, test_lens)
                            print "Test score = %f." % last_test_score

                    # TODO: адекватно ли прерываться на середине эпохи?
                    if early_stop and (patience < iter):
                        print "Early stop!"
                        print "patience = %d" % patience
                        print "iter = %d" % iter
                        stop = True
                        break
            # Конец эпохи:
            train_score = self.score(x_train, y_train, train_lens)
            valid_score =  self.score(x_valid, y_valid, valid_lens)
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_iter_num = epoch * num_train_batches - 1
                if x_test is not None:
                    last_test_score = self.score(x_test, y_test, test_lens)
            print "Epoch %d finished. Training time: %.2f secs" % (epoch, time.time()-start_time)
            print "Train score = %f. Valid score = %f." % (train_score, valid_score)
            if x_test is not None:
                print "Last test score = %f." % last_test_score
        print "OPTIMIZATION COMPLETE."
        print "Best valid score: %f" % best_valid_score
        print "Best iter num: %d, best epoch: %d" % (best_iter_num, best_iter_num // num_train_batches + 1)
        if x_test is not None:
            print "Test score of best iter num: %f" % last_test_score

    def predict(self, data, lengths=None):
        """
        Сеть требует вход размера self.batch_size.
        Если необходимо, заполню недостающие значения нулями.
        :param data: массив данных, каждая строка - отдельный текст,
                     требующий предсказания класса
        :param lengths: (опционально) массив реальных длин текстов
        :return массив - i-ый элемент - наиболее вероятный класс для i-го текста
        """
        if isinstance(data, Series):
            data = data.values.tolist()

        assert len(data) > 0
        sentence_len = len(data[0])

        num_batches = len(data) // self.batch_size
        num_rest = len(data) % self.batch_size
        if num_batches > 0:
            predictions = [self.predict_wrap(self.get_batch(data, i, lengths))
                           for i in range(num_batches)]
        else:
            predictions = []
        if num_rest > 0:
            z = np.zeros((self.batch_size, sentence_len))
            z[0:num_rest] = data[num_batches * self.batch_size: num_batches * self.batch_size + num_rest]
            if lengths is not None:
                z = z[:, :lengths[num_batches * self.batch_size + num_rest - 1]]
            predictions.append(self.predict_wrap(z)[0:num_rest])
        return np.hstack(predictions).flatten()

    def predict_proba(self, data, lengths=None):
        """
        Сеть требует вход размера self.batch_size.
        Если необходимо, заполню недостающие значения нулями.
        :param data: массив данных, каждая строка - отдельный текст,
                     требующий предсказания класса
        :param lengths: (опционально) массив реальных длин текстов
        :return матрицу - в i-ой строке вероятноти всех классов для i-го текста
        """
        if isinstance(data, Series):
            data = data.reset_index(drop=True)

        assert len(data) > 0
        sentence_len = len(data[0])

        num_batches = len(data) // self.batch_size
        num_rest = len(data) % self.batch_size
        if num_batches > 0:
            # TODO: проверь, тут точно нужно приводить к list?
            predictions = [list(self.predict_wrap(self.get_batch(data, i, lengths)))
                           for i in range(num_batches)]
        else:
            predictions = []
        if num_rest > 0:
            z = np.zeros((self.batch_size, sentence_len))
            z[0:num_rest] = data[num_batches * self.batch_size: num_batches * self.batch_size + num_rest]
            if lengths is not None:
                z = z[:, :lengths[num_batches * self.batch_size + num_rest - 1]]
            predictions.append(self.predict_wrap(z)[0:num_rest])

        return np.vstack(predictions)

    def __getstate__(self):
        """ Return state sequence."""
        if hasattr(self, 'orig_class'):
            superparams = self.get_params()  # sklearn.BaseEstimator
            oc = self.orig_class
            cc = self.__class__
            self.__class__ = oc
            params = self.get_params()
            for k, v in superparams.iteritems():
                params[k] = v
            self.__class__ = cc
        else:
            params = self.get_params()  # sklearn.BaseEstimator
        if hasattr(self, 'network') and self.is_ready:
            weights = self.get_all_weights_values()
        else:
            weights = []
        state = (params, weights)
        return state

    def __setstate__(self, state):
        """ Set parameters from state sequence. """
        params, weights = state
        # we may have several classes or superclasses
        for k in ['n_comp', 'use_pca', 'feature']:
            if k in params:
                self.set_params(**{k: params[k]})
                params.pop(k)

        # now switch to MetaCNN if necessary
        if hasattr(self, 'orig_class'):
            cc = self.__class__
            oc = self.orig_class
            self.__class__ = oc
            self.set_params(**params)
            if not self.is_ready:
                self.ready()
            self.set_all_weights_values(weights)
            self.__class__ = cc
        else:
            self.set_params(**params)
            if not self.is_ready:
                self.ready()
            self.set_all_weights_values(weights)

    def load(self, path):
        """ Загрузить параметры модели из файла. """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.__setstate__(state)

    def save_state(self, path):
        """ Сохранить параметры модели в файл. """
        with open(path, 'w') as f:
            pickle.dump(self.__getstate__(), f)

    def set_all_weights_values(self, weights):
        if hasattr(self, 'network'):
            if len(weights) > 0:
                lasagne.layers.set_all_param_values(self.network, weights)
            else:
                print "Error in function 'set_all_weights_values': there is no weights"
        else:
            print "Error in function 'set_all_weights_values': there is no network"

    def get_all_weights_values(self):
        if hasattr(self, 'network'):
            return lasagne.layers.get_all_param_values(self.network)
        else:
            print "Error in function 'get_all_weights_values': there is no network yet - call 'ready' before it."

    def get_params_as_string(self):
        result_str = list()
        result_str.append("clf_name = %s" % self.clf_name)
        result_str.append("vocab_size = %d" % self.vocab_size)
        result_str.append("word_dimension = %d" % self.word_dimension)
        result_str.append("non_static = " + str(self.non_static))

        result_str.append("sentence_len = " + str(self.sentence_len))
        result_str.append("n_out = %d" % self.n_out)

        result_str.append("windows = " + str(self.windows))
        result_str.append("n_filters = %s" % str(self.n_filters))
        result_str.append("n_hidden = %d" % self.n_hidden)
        result_str.append("k_top = %d" % self.k_top)
        result_str.append("activation = %s" % str(self.activations))
        result_str.append("dropout = %2f" % self.dropout)
        result_str.append("l1_regs = %s" % str(self.l1_regs))
        result_str.append("l2_regs = %s" % str(self.l2_regs))

        result_str.append("batch_size = %d" % self.batch_size)
        result_str.append("seed = %d" % self.seed)
        return '\n'.join(result_str)
