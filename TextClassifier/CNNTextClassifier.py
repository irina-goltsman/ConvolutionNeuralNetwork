__author__ = 'irina-goltsman'
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator
import cPickle as pickle
import theano
# from theano.tensor.nnet import conv
from pandas.core.series import Series
from sklearn.cross_validation import train_test_split
import time
import warnings
from network import *
from data_tools import check_all_sentences_have_one_dim

theano.config.exception_verbosity = 'high'
warnings.filterwarnings("ignore", category=DeprecationWarning)

#TODO: юзать больше карт: from 100 to 600
#TODO: 4. добавить возможность изменять векторное представление слов в процессе обучения
#TODO: заюзать glove вместо/вместе word2vec
#TODO: попробовать разные активационные функции, в том числе Ident
#TODO: 5. добавить dropout-слой - "use a small dropout rate(0.0-0.5) and a large max norm constraint"
#TODO: лучше юзай 1-max-pooling для предложений
#TODO: ОБУЧАТЬ ПАЧКАМИ - лучше adam optimizations или хотя бы adagrad
#TODO: реализовать 2х слойную модель с k-max-pooling.

#TODO: в датасете MR бывают совсем длинные тексты, возможно выравнивание по длине - не лучшая идея
#TODO: почему-то с word_dim = 100 обучается, а с word_dim = 300 - нет.
# mode=NanGuardMode - для дебага нанов, поняла что [nvcc] fastmath=True - вызывает nan!

'''
class ConvLayerForSentences(object):
    """Свёрточный слой для классификации предложений"""

    def __init__(self, input_data, rng, filter_shape=(10, 1, 5, 100), sentences_shape=None):
        """
        Инициализирует ConvLayerForSentences с общими переменными внутренних параметров.

        :type input_data: theano.tensor.dtensor4
        :param input_data: символичный тензор предложений формата sentences_shape

        :type rng: numpy.random.RandomState
        :param rng: генератор случайных чисел для инициализации весов

        :type filter_shape: tuple или list длины 4
        :param filter_shape: (количество фильтров, количество входных каналов,
                              высота фильтра = окно слов, ширина фильтра = размерность вектора слова)

        :type sentences_shape: tuple или list длины 4
        :param sentences_shape: (количество предложений, количество каналов,
                                 высота изображения = длина предложения,
                                 ширина изображения = размерность вектора слова)

        # Записывает в self.output 4D тензор, размера: (batch size, nfilters, output row, output col)
        """
        if sentences_shape is not None:
            # проверяю совпадение размерности вектора слова
            assert sentences_shape[4] == filter_shape[4]
        self.input = input_data

        W_bound = 0.5
        # каждая карта входных признаков соединена с каждым фильтром,
        # поэтому и такая размерность у матрицы весов
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                          dtype=theano.config.floatX),
                               borrow=True)

        # символическое выражение, выполняющее операцию свёртки с помощью фильтров
        # Возвращает 4D тензор, размера: (batch size, nfilters, output row, output col)
        conv_out = conv.conv2d(
            input=input_data,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=sentences_shape
        )

        # смещения - 1D тензор (вектор) - одно смещение для одного фильтра
        # filter_shape[0] - количество фильтров
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # дабавляем смещения. Изначально это 1D вектор,
        # который мы преобразовываем в тензор (1, n_filters, 1, 1)
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # сохраним параметры этого слоя
        self.params = [self.W, self.b]


class KMaxPoolingLayer(object):
    def __init__(self, input, k):
        """
        Записывет k максимальных значений по оси 3, отвечающей результату одного фильтра
        :param input: символичный тензор размера 4: (batch size, nfilters, output row, output col)
                      (output row = высота - итоговое количество окон предложения для одного фильтра)
        :param k: int, количество максимальных элементов
        """
        # axis=2 так как нам нужна сортировка внутри каждого фильтра
        pooling_args_sorted = T.argsort(input, axis=2)
        args_of_k_max = pooling_args_sorted[:, :, -k:, :]
        # не хочу терять порядок слов, поэтому ещё раз сортирую номера максимумов:
        args_of_k_max_sorted = T.sort(args_of_k_max, axis=2)

        dim0 = T.arange(input.shape[0]).repeat(input.shape[1] * k * input.shape[3])
        dim1 = T.arange(input.shape[1]).repeat(k * input.shape[3]).reshape((1, -1))\
            .repeat(input.shape[0], axis=0).flatten()
        dim2 = args_of_k_max_sorted.flatten()
        dim3 = T.arange(input.shape[3]).reshape((1, -1))\
            .repeat(input.shape[0] * input.shape[1] * k, axis=0).flatten()

        self.output = input[dim0,dim1,dim2,dim3].reshape((input.shape[0], input.shape[1], k, input.shape[3]))


class FullyConnectedLayer(object):
    def __init__(self, input, n_in, n_out, rng, W=None, b=None):
        """
        Скрытый слой. W - матрица весов размерности (n_in, n_out), b - вектор сдвигов (n_out)

        :type input: theano.tensor.lvector
        :param input: символичный вектор размерности (n_in)

        :type n_in: int
        :param n_in: размерность входа

        :type n_out: int
        :param n_out: размерность выхода

        :type rng: np.random.RandomState
        :param rng: генератор случайных чисел

        """
        self.input = input

        # Инициализация матрицы весов
        if W is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                              high=np.sqrt(6. / (n_in + n_out)),
                                              size=(n_in, n_out)),
                                  dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.output = T.dot(input, self.W) + self.b

        # параметры слоя
        self.params = [self.W, self.b]

class CNNForSentences(object):
    """
    Свёрточная сеть с одним свёрточным слоем
    """
    def __init__(self, input, n_out, n_hidden, n_filters, n_chanels, windows, word_dimension,
                 activations=(T.nnet.relu, T.nnet.relu, T.nnet.sigmoid), seed=0, k_max=5, k_top=1):
        """
        :type input: theano.tensor.dtensor4
        :param input: символичный тензор предложений формата:
                      (количество предложений,
                       количество каналов - для первой свёртки обычно 1,
                       высота = длина предложения,
                       ширина = размерность вектора слова)

        :param n_out: количество целевых классов классификации
        :param n_hidden:  число нейронов скрытого полносвязного слоя
        :param n_filters: число фильтров для каждого вида свёртки
        :param n_chanels: число входных каналов
        :type windows: list
        :param windows: размеры окон для фильтров
        :param activations: активационные функции для слоёв
        :param seed: начальное значение для генератора случайных чисел
        """
        rng = np.random.RandomState(seed)

        assert word_dimension == input.shape[3]

        self.layers0 = list()
        layer2_inputs = list()
        for window in windows:
            layer0 = ConvLayerForSentences(input_data=input, rng=rng,
                                           filter_shape=(n_filters, n_chanels, window, word_dimension / 2))
            self.layers0.append(layer0)

            # layer0 записывает в self.output 4D тензор, размера: (batch size, nfilters, output row, output col)
            # если ширина фильтра = word_dimension (ширине изображения), то output col = 1
            layer1_input = activations[0](layer0.output)
            # Для фильтров разной ширины тут оставляем ровно k_max максимальных значений
            self.layer1 = KMaxPoolingLayer(layer1_input, k_max)

            layer2_input = self.layer1.output
            layer2_inputs.append(layer2_input)
        # Собираем все максимальные значения по всем фильтрам
        layer2_input = T.concatenate(layer2_inputs, axis=1)

        # TODO: скорее всего так указывать размерность некорректно
        self.layer2 = ConvLayerForSentences(input_data=layer2_input, rng=rng,
                                            filter_shape=(n_filters, layer2_input.shape[1],
                                                          k_max/2, layer2_input.shape[3]))

        layer3_input = activations[1](self.layer2.output)

        self.layer3 = KMaxPoolingLayer(layer3_input, k_top)

        layer4_input = self.layer3.output.reshape((batch_size, -1))
        # После этого слоя осталось ровно n_filters * k * len(windows) элементов
        self.layer4 = FullyConnectedLayer(input=layer4_input, n_in=n_filters * k_max * len(windows),
                                          n_out=n_hidden, rng=rng)

        # TODO: до или после полносвязного слоя добавь dropout слой

        softmax_input = activations[1](self.layer4.output)

        # Выход сети - вероятности всех классов
        self.p_y_given_x = T.nnet.softmax(softmax_input)

        # TODO: тут зафигачь регуляризацию везде где можно - передавай
        # # CNN regularization
        # self.L1 = self.layer3.L1
        # self.L2_sqr = self.layer3.L2_sqr

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer4.params + self.layer2.params
        # TODO: тут не уверена в порядке
        for layer0 in self.layers0:
            self.params += layer0.params

        # self.y_pred = self.layer3.y_pred

        # TODO: T.nnet.categorical_crossentropy(predictions, targets)
        # self.loss = lambda y: self.layer3.negative_log_likelihood(y)
'''


class CNNTextClassifier(BaseEstimator):
    def __init__(self, vocab_size, word_dimension=100, word_embedding=None, non_static=True,
                 batch_size=100, sentence_len=None, n_out=2,
                 windows=((5, 6), (4,)), n_filters=(10, 25), n_hidden=10, activations=('tanh', 'tanh'),
                 dropout=0.5, L2_regs=(0.0001/2, 0.00003/2, 0.000003/2, 0.0001/2),
                 k_top=1, learning_rate=0.1, n_epochs=3, seed=0):
        """
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
        :param L2_regs: параметры для регуляризации
        :type k_top: int (>=1)
        :param k_top: параметр для итогового k-max-pooling слоя
        :param learning_rate: темп обучения
        :param n_epochs: количество эпох обучения
        :param seed: начальное значение для генератора случайных чисел
        """
        self._estimator_type = "classifier"
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
        self.L2_regs = L2_regs
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

    def ready(self):
        # Матрица входа, размера n_batch * n_sentence
        self.x = T.lmatrix('x')
        # Результирующие классы для каждого предложения в batch
        self.y = T.ivector('y')

        print "CNN building..."
        self.network = build_cnn_for_texts(input_var=self.x, batch_size=self.batch_size,
                                           sentence_len=self.sentence_len, vocab_size=self.vocab_size,
                                           word_dimension=self.word_dimension, word_embedding=self.word_embedding,
                                           non_static=self.non_static,
                                           windows=self.windows, k_top=self.k_top, n_filters=self.n_filters,
                                           activations=self.activations, dropout=self.dropout, n_out=self.n_out)

        # ключевое слово deterministic отключает стохастическое поведение
        # например, убирает dropout
        self.p_y_given_x = lasagne.layers.get_output(self.network, self.x, deterministic=True)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.predict_wrap = theano.function(inputs=[self.x], outputs=self.y_pred)
        self.predict_proba_wrap = theano.function(inputs=[self.x], outputs=self.p_y_given_x)
        print "CNN building finished"

        self.is_ready = True

    def ready_to_train(self, x_train, y_train, x_valid, y_valid):
        # zero_vec_tensor = T.vector()
        # # Если non_static, то 0ая компонента матрицы слов могла измениться, а она должна всегда быть нулевым вектором.
        # set_zero = theano.function([zero_vec_tensor], updates=[(self.words, T.set_subtensor(self.words[0, :], zero_vec_tensor))],
        #                            allow_input_downcast=True)

        # Выпишу слои к которым должна применяться L2 регуляризация
        print "Preparing for training..."
        l2_layers = []
        for layer in lasagne.layers.get_all_layers(self.network):
            if isinstance(layer, (CNN.embeddings.SentenceEmbeddingLayer,
                                  lasagne.layers.conv.Conv1DLayer,
                                  lasagne.layers.DenseLayer)):
                l2_layers.append(layer)

        train_prediction = lasagne.layers.get_output(self.network, self.x)
        loss_train = lasagne.objectives.aggregate(
            lasagne.objectives.categorical_crossentropy(train_prediction, self.y), mode='mean')\
                     + lasagne.regularization.regularize_layer_params_weighted(dict(zip(l2_layers, self.L2_regs)),
                                                                               lasagne.regularization.l2)

        all_params = lasagne.layers.get_all_params(self.network)
        updates = lasagne.updates.adagrad(loss_train, all_params, self.learning_rate)

        self.loss_eval = lasagne.objectives.categorical_crossentropy(self.p_y_given_x, self.y)
        self.correct_predictions = T.eq(self.y_pred, self.y)

        index = T.lscalar()
        self.train_model = theano.function([index], outputs=loss_train, updates=updates,
                                           givens={
                                               self.x: x_train[index * self.batch_size: (index + 1) * self.batch_size],
                                               self.y: y_train[index * self.batch_size: (index + 1) * self.batch_size]},
                                           allow_input_downcast=True)

        self.val_loss = theano.function([index], outputs=self.loss_eval,
                                        givens={
                                            self.x: x_valid[index * self.batch_size: (index + 1) * self.batch_size],
                                            self.y: y_valid[index * self.batch_size: (index + 1) * self.batch_size]},
                                        allow_input_downcast=True)

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

            X_new = np.append(X, X_extra, axis=0)
            y_new = np.append(y, y_extra, axis=0)
        else:
            X_new = X
            y_new = y
        return X_new, y_new

    # TODO: разбери эту огромную функцию на маленькие читабельные части
    def fit(self, x_train, y_train, n_epochs=None, validation_part=0.1,
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

        rng = np.random.RandomState(self.seed)
        x_train, y_train = self.extend_to_batch_size(x_train, y_train, self.batch_size, rng)

        num_batches = len(x_train) / self.batch_size
        num_val_batches = int(np.floor(num_batches * validation_part))
        num_train_batches = num_batches - num_val_batches

        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                              test_size=num_val_batches * self.batch_size,
                                                              random_state=rng)
        assert len(x_valid) % self.batch_size == 0

        if isinstance(y_valid, Series):
            y_valid = y_valid.reset_index(drop=True)
        if isinstance(x_valid, Series):
            x_valid = x_valid.reset_index(drop=True)

        # подготовим CNN
        if not self.is_ready:
            self.ready()

        x_train, y_train = self.shared_dataset(x_train, y_train)
        x_valid, y_valid = self.shared_dataset(x_valid, y_valid)
        print 'x_train.shape = ' + str(x_train.shape.eval())
        print 'x_valid.shape = ' + str(x_valid.shape.eval())

        if not self.is_ready_to_train:
            self.ready_to_train(x_train, y_train, x_valid, y_valid)

        if n_epochs is not None:
            self.n_epochs = int(n_epochs)

        epoch = 0
        best_valid_loss, best_valid_score, best_iter_num = np.inf, 0, 0
        # early-stopping parameters
        # TODO: разберись каким выбрать этот параметр
        valid_frequency = min(valid_frequency, num_train_batches - 1)
        print "visualization frequency: %d batches" % valid_frequency
        patience = num_train_batches * 2  # look as this many examples regardless
        patience_increase = 1.5  # wait this much longer when a new best is found
        improvement_threshold = 0.9  # a relative improvement of this much is
        stop = False
        while (epoch < self.n_epochs) and (not stop):
            start_time = time.time()
            epoch += 1
            train_cost = 0
            indices = rng.permutation(num_train_batches)
            for cur_idx, real_idx in enumerate(indices):
                iter = (epoch - 1) * num_train_batches + cur_idx
                train_cost += self.train_model(real_idx)

                if iter % valid_frequency == 0:
                    valid_loss = np.mean([self.val_loss(i) for i in xrange(num_val_batches)])
                    valid_score = np.mean([self.val_score(i) for i in xrange(num_val_batches)])
                    # выведу результаты:
                    print "Training: cur_idx = %d, real_idx = %d, mean train cost = %f" % \
                          (cur_idx, real_idx, train_cost / (cur_idx + 1))
                    print "global_iter %d, epoch %d, batch %d: mean valid loss: %f, valid score: %f" \
                          % (iter, epoch, cur_idx, float(valid_loss), float(valid_score))

                    if early_stop and (valid_loss * improvement_threshold < best_valid_loss):
                        patience = max(patience, iter * patience_increase)
                        print "new patience = %d" % patience

                    if valid_loss < best_valid_loss or valid_score > best_valid_score:
                        best_valid_loss = valid_loss
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
            print "train_score = %f" % train_score
            valid_score = np.mean([self.val_score(i) for i in xrange(num_val_batches)])
            print "Epoch %d finished. Training time: %.2f secs" % (epoch, time.time()-start_time)
            print "Train score = %f. Valid score = %f." % (float(train_score), float(valid_score))

        print "OPTIMIZATION COMPLETE."
        print "Best valid loss: %f" % best_valid_loss
        print "Best iter num: %d, best epoch: %d" % (best_iter_num, best_iter_num // num_train_batches)

    def predict(self, data):
        """
        Сеть требует вход размера self.batch_size.
        Если необходимо, заполню недостающие значения нулями.
        :param data: массив данных, каждая строка - отдельный текст,
                     требующий предсказания класса
        :return массив - i-ый элемент - наиболее вероятный класс для i-го текста
        """
        if isinstance(data, Series):
            data = data.reset_index(drop=True)

        assert len(data) > 0
        sentence_len = len(data[0])

        num_batches = len(data) // self.batch_size
        num_rest = len(data) % self.batch_size
        if num_batches > 0:
            predictions = [list(self.predict_proba_wrap(data[i * self.batch_size: (i + 1) * self.batch_size]))
                           for i in range(num_batches)]
        else:
            predictions = []
        if num_rest > 0:
            z = np.zeros((self.batch_size, sentence_len))
            z[0:num_rest] = data[num_batches * self.batch_size: num_batches * self.batch_size + num_rest]
            predictions.append(self.predict_proba_wrap(z)[0:num_rest])

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
        if hasattr(self, 'network'):
            weights = self.get_all_param_values()
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
            self.ready()
            self.set_all_param_values(weights)
            self.__class__ = cc
        else:
            self.set_params(**params)
            self.ready()
            self.set_all_param_values(weights)

    def load(self, path):
        """ Загрузить параметры модели из файла. """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.__setstate__(state)

    def save_state(self, path):
        """ Сохранить параметры модели в файл. """
        with open(path, 'w') as f:
            pickle.dump(self.__getstate__(), f)

    def set_all_param_values(self, weights):
        if hasattr(self, 'network'):
            if len(weights) > 0:
                lasagne.layers.set_all_param_values(self.network, weights)
            else:
                print "Error in function 'set_all_param_values': there is no weights"
        else:
            print "Error in function 'set_all_param_values': there is no network"

    def get_all_param_values(self):
        if hasattr(self, 'network'):
            return lasagne.layers.get_all_param_values(self.network)
        else:
            print "Error in function 'get_all_param_values': there is no network yet - call 'ready' before it."

    def get_params_as_string(self):
        result_str = list()
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
        result_str.append("L2_regs = %s" % str(self.L2_regs))

        result_str.append("batch_size = %d" % self.batch_size)
        result_str.append("learning_rate = %2f" % self.learning_rate)
        result_str.append("n_epochs = %d" % self.n_epochs)
        result_str.append("seed = %d" % self.seed)
        return '\n'.join(result_str)
