__author__ = 'irina-goltsman'
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator
import cPickle as pickle
import theano
from pandas.core.series import Series
from sklearn.cross_validation import train_test_split
import time
import warnings
from networks import *
from data_tools import check_all_sentences_have_one_dim
from lasagne.random import set_rng

theano.config.exception_verbosity = 'high'
warnings.filterwarnings("ignore", category=DeprecationWarning)

#TODO: юзать больше карт: from 100 to 600
#TODO: заюзать glove вместо/вместе word2vec
#TODO: 5. добавить dropout-слой - "use a small dropout rate(0.0-0.5) and a large max norm constraint"

#TODO: в датасете MR бывают совсем длинные тексты, возможно выравнивание по длине - не лучшая идея
#TODO: почему-то с word_dim = 100 обучается, а с word_dim = 300 - нет.
# mode=NanGuardMode - для дебага нанов, поняла что [nvcc] fastmath=True - вызывает nan!
# TODO: возможно get_all_params в lasagne делает не то, что я предполагала

# TODO: ================================6 подумать над реализацией character-level сети - 30мин
# TODO: ================================сделать character-level сеть - 2ч ( переоценить время после пункта 6 )
# TODO: 2 проверить, сохраняется ли state? - 5 мин
# TODO: 3 протестить 1cnn на 20news - чисто эмпирически (макс 3 эпохи) - 30мин - 1.5ч
# TODO: 5 протестить 20news на baselines - сравнить - 1ч
# TODO: 7 я правильно понимаю, что 2-х слойная сеть вообще не обучается? - 15мин
# TODO: 8 запустить baseline на twitter_google_300 - оценить примерное качество - 2ч
# TODO: 9 обучить сетку 1cnn на twitter_google_300 - 10ч (ГДЕ ВЗЯТЬ СТОЛЬКО ВРЕМЕНИ???)
# TODO: загрузить, почистить, пересобрать новый датасет - 3ч
# TODO: нечего делать? занят процессор? - рефакторинг кода наше всё
# TODO: убрать стоп слова и протестить на бейзлайнах ещё раз

# На хадупе от яндекса нет gensim и lasagne - хотя с первой проблемо ещё можно справиться... load_bin_vec

class CNNTextClassifier(BaseEstimator):
    def __init__(self, clf_name='1cnn', vocab_size=None, word_dimension=100, word_embedding=None, non_static=True,
                 batch_size=100, sentence_len=None, n_out=2,
                 windows=((5, 6), (4,)), n_filters=(10, 25), n_hidden=10, activations=('tanh', 'tanh'),
                 dropout=0.5, L1_regs=(0.0001 / 2, 0.00003 / 2, 0.000003 / 2, 0.0001 / 2),
                 k_top=1, learning_rate=0.1, n_epochs=3, seed=0):
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
        self.L1_regs = L1_regs
        self.is_ready = False
        self.is_ready_to_train = False
        self.k_top = k_top
        self.learning_rate = learning_rate
        self.n_epochs = int(n_epochs)
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
        clf_builder_dict = {'1cnn': build_1cnn, 'dcnn': build_dcnn}
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
                                   non_static=self.non_static,  # n_hidden=self.n_hidden
                                   windows=self.windows, k_top=self.k_top, n_filters=self.n_filters,
                                   activations=self.activations, dropout=self.dropout, n_out=self.n_out)

        # ключевое слово deterministic отключает стохастическое поведение, например, убирает dropout
        self.p_y_given_x = lasagne.layers.get_output(self.network, self.x, deterministic=True)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.predict_wrap = theano.function(inputs=[self.x], outputs=self.y_pred, allow_input_downcast=True)
        self.predict_proba_wrap = theano.function(inputs=[self.x], outputs=self.p_y_given_x, allow_input_downcast=True)
        print "CNN building finished"

        self.is_ready = True

    def ready_to_train(self, x_train, y_train, x_valid, y_valid):
        # zero_vec_tensor = T.vector()
        # # Если non_static, то 0ая компонента матрицы слов могла измениться, а она должна всегда быть нулевым вектором.
        # set_zero = theano.function([zero_vec_tensor], updates=[(self.words, T.set_subtensor(self.words[0, :], zero_vec_tensor))],
        #                            allow_input_downcast=True)

        # Выпишу слои к которым должна применяться L2 регуляризация
        print "Preparing for training..."
        l1_layers = []
        for layer in lasagne.layers.get_all_layers(self.network):
            if isinstance(layer, (CNN.embeddings.SentenceEmbeddingLayer,
                                  CNN.Conv1DLayerSplitted,
                                  lasagne.layers.conv.Conv2DLayer,
                                  lasagne.layers.DenseLayer)):
                l1_layers.append(layer)
        print "num of l1_layers is %d" % len(l1_layers)

        train_prediction = lasagne.layers.get_output(self.network, self.x, deterministic=False)
        loss_train = lasagne.objectives.aggregate(
            lasagne.objectives.categorical_crossentropy(train_prediction, self.y), mode='mean')\
                     + lasagne.regularization.regularize_layer_params_weighted(dict(zip(l1_layers, self.L1_regs)),
                                                                               lasagne.regularization.l1)

        all_params = lasagne.layers.get_all_params(self.network)
        updates = lasagne.updates.adadelta(loss_train, all_params, self.learning_rate)
        # updates = lasagne.updates.adam(loss_train, all_params)

        # self.loss_eval = lasagne.objectives.categorical_crossentropy(self.p_y_given_x, self.y)
        self.correct_predictions = T.eq(self.y_pred, self.y)

        index = T.lscalar()
        self.train_model = theano.function([index], outputs=loss_train, updates=updates,
                                           givens={
                                               self.x: x_train[index * self.batch_size: (index + 1) * self.batch_size],
                                               self.y: y_train[index * self.batch_size: (index + 1) * self.batch_size]},
                                           allow_input_downcast=True)

        # self.val_loss = theano.function([index], outputs=self.loss_eval,
        #                                 givens={
        #                                     self.x: x_valid[index * self.batch_size: (index + 1) * self.batch_size],
        #                                     self.y: y_valid[index * self.batch_size: (index + 1) * self.batch_size]},
        #                                 allow_input_downcast=True)

        self.val_score = theano.function([index], outputs=self.correct_predictions,
                                         givens={
                                             self.x: x_valid[index * self.batch_size: (index + 1) * self.batch_size],
                                             self.y: y_valid[index * self.batch_size: (index + 1) * self.batch_size]},
                                         allow_input_downcast=True)

        self.train_score = theano.function([index], outputs=self.correct_predictions,
                                           givens={
                                               self.x: x_train[index * self.batch_size: (index + 1) * self.batch_size],
                                               self.y: y_train[index * self.batch_size: (index + 1) * self.batch_size]},
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

    def score(self, X, y):
        """
        :param X: array-like, shape = [n_samples, n_features]
        :param y: array-like, shape = [n_samples]
        :return: среднюю точность предсказания
        """
        return np.mean(self.predict(X) == y)

    @staticmethod
    def extend_to_batch_size(X, y, batch_size, rng):
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

            if isinstance(X, Series):
                X = X.values.tolist()

            X = np.append(X, X_extra, axis=0)
            y = np.append(y, y_extra, axis=0)

        return X, y

    # TODO: разбери эту огромную функцию на маленькие читабельные части
    def fit(self, x_train, y_train, model_path=None, n_epochs=None, validation_part=0.1,
            valid_frequency=4, early_stop=False):
        """ Fit model
        :type x_train: 2d массив из int
        :param x_train: входные данные - список из текстов
                        (каждый текст представлен как список номеров слов)
        :type y_train: 1d массив int
        :param y_train: целевые значения для каждого текста
        :type n_epochs: int/None
        :param n_epochs: used to override self.n_epochs from init.
        :type validation_part: float
        :param validation_part: доля тренеровочных данных, которые станут валидационной выборкой
        :type valid_frequency: int/None
        :param valid_frequency: если не None, то каждые visualization_frequency интераций
                                        будет выводиться результат модели на валидационной выборке
        :type early_stop: bool
        :param early_stop: если True - будет происходить досрочная остановка.
        """
        assert max(y_train) < self.n_out
        assert min(y_train) >= 0
        assert len(x_train) == len(y_train)

        # подготовим CNN
        if not self.is_ready:
            self.ready()

        if isinstance(x_train, Series):
            x_train = x_train.values.tolist()
        if isinstance(y_train, Series):
            y_train = y_train.values.tolist()

        x_train, y_train = self.extend_to_batch_size(x_train, y_train, self.batch_size, self.rng)

        num_batches = len(x_train) / self.batch_size
        num_val_batches = int(np.floor(num_batches * validation_part))
        num_train_batches = num_batches - num_val_batches

        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                              test_size=num_val_batches * self.batch_size,
                                                              random_state=self.rng)
        assert len(x_valid) % self.batch_size == 0

        if isinstance(y_valid, Series):
            y_valid = y_valid.reset_index(drop=True)
        if isinstance(x_valid, Series):
            x_valid = x_valid.reset_index(drop=True)

        x_train, y_train = self.shared_dataset(x_train, y_train)
        x_valid, y_valid = self.shared_dataset(x_valid, y_valid)
        print 'x_train.shape = ' + str(x_train.shape.eval())
        print 'x_valid.shape = ' + str(x_valid.shape.eval())

        if not self.is_ready_to_train:
            self.ready_to_train(x_train, y_train, x_valid, y_valid)

        if model_path is not None:
            self.load(model_path)

        if n_epochs is not None:
            self.n_epochs = int(n_epochs)

        epoch = 0
        best_valid_score, best_iter_num = 0, 0
        # early-stopping parameters
        valid_frequency = min(valid_frequency, num_train_batches - 1)
        print "visualization frequency: %d batches" % valid_frequency
        patience = num_train_batches * 2  # look as this many examples regardless
        patience_increase = 2.0  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is
        stop = False
        while (epoch < self.n_epochs) and (not stop):
            start_time = time.time()
            epoch += 1
            train_cost = 0
            indices = self.rng.permutation(num_train_batches)
            for cur_idx, real_idx in enumerate(indices):
                iter = (epoch - 1) * num_train_batches + cur_idx
                cur_train_cost = self.train_model(real_idx)
                train_cost += cur_train_cost

                if iter > 0 and iter % valid_frequency == 0:
                    print "global_iter %d, epoch %d, batch %d, mean train cost = %f" % \
                          (iter, epoch, cur_idx, train_cost / valid_frequency)
                    train_cost = 0
                    valid_score = np.mean([self.val_score(i) for i in xrange(num_val_batches)])
                    # выведу результаты валидации:
                    print "------------valid score: %f------------" \
                          % (float(valid_score))

                    if early_stop and (valid_score > improvement_threshold * best_valid_score):
                        patience = max(patience, iter * patience_increase)
                        print "new patience = %d" % patience

                    if valid_score > best_valid_score:
                        best_valid_score = valid_score
                        best_iter_num = iter

                    # TODO: адекватно ли прерываться на середине эпохи?
                    if early_stop and (patience < iter):
                        print "Early stop!"
                        print "patience = %d" % patience
                        print "iter = %d" % iter
                        stop = True
                        break
            # Конец эпохи:
            train_score = np.mean([self.train_score(i) for i in xrange(num_train_batches)])
            valid_score = np.mean([self.val_score(i) for i in xrange(num_val_batches)])
            print "Epoch %d finished. Training time: %.2f secs" % (epoch, time.time()-start_time)
            print "Train score = %f. Valid score = %f." % (float(train_score), float(valid_score))

        print "OPTIMIZATION COMPLETE."
        print "Best valid score: %f" % best_valid_score
        print "Best iter num: %d, best epoch: %d" % (best_iter_num, best_iter_num // num_train_batches + 1)

    def predict(self, data):
        """
        Сеть требует вход размера self.batch_size.
        Если необходимо, заполню недостающие значения нулями.
        :param data: массив данных, каждая строка - отдельный текст,
                     требующий предсказания класса
        :return массив - i-ый элемент - наиболее вероятный класс для i-го текста
        """
        if isinstance(data, Series):
            data = data.values.tolist()

        assert len(data) > 0
        sentence_len = len(data[0])

        num_batches = len(data) // self.batch_size
        num_rest = len(data) % self.batch_size
        if num_batches > 0:
            predictions = [self.predict_wrap(data[i * self.batch_size: (i + 1) * self.batch_size])
                           for i in range(num_batches)]
        else:
            predictions = []
        if num_rest > 0:
            z = np.zeros((self.batch_size, sentence_len))
            z[0:num_rest] = data[num_batches * self.batch_size: num_batches * self.batch_size + num_rest]
            predictions.append(self.predict_wrap(z)[0:num_rest])
        return np.hstack(predictions).flatten()

    def predict_proba(self, data):
        """
        Сеть требует вход размера self.batch_size.
        Если необходимо, заполню недостающие значения нулями.
        :param data: массив данных, каждая строка - отдельный текст,
                     требующий предсказания класса
        :return матрицу - в i-ой строке вероятноти всех классов для i-го текста
        """
        if isinstance(data, Series):
            data = data.reset_index(drop=True)

        assert len(data) > 0
        sentence_len = len(data[0])

        num_batches = len(data) // self.batch_size
        num_rest = len(data) % self.batch_size
        if num_batches > 0:
            predictions = [list(self.predict_wrap(data[i * self.batch_size: (i + 1) * self.batch_size]))
                           for i in range(num_batches)]
        else:
            predictions = []
        if num_rest > 0:
            z = np.zeros((self.batch_size, sentence_len))
            z[0:num_rest] = data[num_batches * self.batch_size: num_batches * self.batch_size + num_rest]
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

        result_str.append("sentence_len = %d" % self.sentence_len)
        result_str.append("n_out = %d" % self.n_out)

        result_str.append("windows = " + str(self.windows))
        result_str.append("n_filters = %s" % str(self.n_filters))
        result_str.append("n_hidden = %d" % self.n_hidden)
        result_str.append("k_top = %d" % self.k_top)
        result_str.append("activation = %s" % str(self.activations))
        result_str.append("dropout = %2f" % self.dropout)
        result_str.append("L1_regs = %s" % str(self.L1_regs))

        result_str.append("batch_size = %d" % self.batch_size)
        result_str.append("learning_rate = %2f" % self.learning_rate)
        result_str.append("n_epochs = %d" % self.n_epochs)
        result_str.append("seed = %d" % self.seed)
        return '\n'.join(result_str)
