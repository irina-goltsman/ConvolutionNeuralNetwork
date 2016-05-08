# -*- coding: utf-8 -*-
import numpy
import re
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator
import cPickle as pickle
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from gensim.models import Word2Vec

theano.config.exception_verbosity = 'high'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ConvLayerForSentences(object):
    """Свёрточный слой для классификации предложений"""

    def __init__(self, rng, input_data, filter_shape=(10, 1, 5, 100),
                 sentences_shape=None, activation=T.tanh):
        """
        Инициализирует ConvLayerForSentences с общими переменными внутренних параметров.

        :type rng: numpy.random.RandomState
        :param rng: генератор случайных чисел для инициализации весов

        :type input_data: theano.tensor.dtensor4
        :param input_data: символичный тензор предложений формата sentences_shape

        :type filter_shape: tuple или list длины 4
        :param filter_shape: (количество фильтров, количество входных каналов (для первой свёртки = 1),
                              высота фильтра = окно слов, ширина фильтра = размерность вектора слова)

        # NOTE: вероятно не нужно это указывать, если только каналов > 1, то тут нужно заморачиваться
        :type sentences_shape: tuple или list длины 4
        :param sentences_shape: (количество предложений = 1(всегда), количество каналов - обычно 1,
                                 высота изображения = длина предложения,
                                 ширина изображения = размерность вектора слова)

        :param activation: активационная функция

        # Записывает в self.output 4D тензор, размера: (batch size (1), nfilters, output row, output col)
        """
        if sentences_shape is not None:
            # проверяю совпадение размерности вектора слова
            assert sentences_shape[4] == filter_shape[4]
        self.input = input_data

        W_bound = 0.5
        # каждая карта входных признаков соединена с каждым фильтром,
        # поэтому и такая размерность у матрицы весов
        self.W = theano.shared(
            numpy.asarray(rng.uniform(low=-W_bound, high=W_bound,
                                      size=filter_shape),
                          dtype=theano.config.floatX),
            borrow=True
        )

        # символическое выражение, выполняющее операцию свёртки с помощью фильтров
        # Возвращает 4D тензор, размера: (batch size (1), nfilters, output row, output col)
        conv_out = conv.conv2d(
            input=input_data,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=sentences_shape
        )

        # смещения - 1D тензор (вектор) - одно смещение для одного фильтра
        # filter_shape[0] - количество фильтров
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # дабавляем смещения. Изначально это 1D вектор,
        # который мы преобразовываем в тензор (1, n_filters, 1, 1)
        self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # сохраним параметры этого слоя
        self.params = [self.W, self.b]


class MaxOverTimePoolingLayer(object):
    def __init__(self, pooling_input):
        """
        Записывет максимальные значения по оси, отвечающей результату одного фильтра
        Один максимум для одного фильтра - для простейшего варианта - ширина фильтра =
        размерности слова
        :type pooling_input: символичный тензор размера 2:
                            (количество входных карт признаков(число фильтров),
                             output row * output col(1) * batch size (1)
                             (output row = высота - итоговое количество окон предложения для одного фильтра))
        :param pooling_input: вход для пулинга
        """
        max_args = T.argmax(pooling_input, axis=1)
        n_filters = pooling_input.shape[0]
        self.output = pooling_input[T.arange(n_filters), max_args]
        self.output = self.output.reshape((n_filters, 1))


class KMaxPoolingLayer(object):
    def __init__(self, pooling_input, k):
        """
        Записывает в self.output k максимальных значений входа
        :param pooling_input: символичный тензор размера 2:
                              (количество входных карт признаков(число фильтров),
                              output row * output col(1) * batch size (1)
                              (output row = высота - итоговое количество окон предложения для одного фильтра))
        :param k: int, количество максимальных элементов
        В self.output записывается тензор размера 2: (количество входных карт признаков(число фильтров),
                                                      k максимальных значений по каждому фильтру без потери порядка)
        """
        # axis=1 так как нам нужна сортировка внутри каждого фильтра (т.е внутри строк каждой матрицы)
        pooling_args_sorted = T.argsort(pooling_input, axis=1)
        args_of_k_max = pooling_args_sorted[:, -k:]
        # не хочу терять порядок слов, поэтому ещё раз сортирую номера максимумов:
        args_of_k_max_sorted = T.sort(args_of_k_max, axis=1)

        n_filters = pooling_input.shape[0]

        ii = T.repeat(T.arange(n_filters), k)
        jj = args_of_k_max_sorted.flatten()

        self.output = pooling_input[ii, jj].reshape((n_filters, k))


class FullyConnectedLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        Скрытый слой. W - матрица весов размерности (n_in, n_out), b - вектор сдвигов (n_out)

        :type rng: np.random.RandomState
        :param rng: генератор рандомных чисел

        :type input: theano.tensor.lvector
        :param input: символичный вектор размерности (n_in)

        :type n_in: int
        :param n_in: размерность входа

        :type n_out: int
        :param n_out: размерность выхода

        :type activation: theano.Op or function
        :param activation: функция активации
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        if activation is None:
            self.output = lin_output
        else:
            self.output =  activation(lin_output)
        # параметры слоя
        self.params = [self.W, self.b]


class SoftmaxLayer(object):
    def __init__(self, input, n_in, n_out):
        """ Инициализации параметров логистической регрессии

        :type input: theano.tensor.TensorType
        :param input: символичный вектор, описывающий вход

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which the datapoints lie

        :type n_out: int
        :param n_out: размерость выхода - количество целевых значений для классификации
        """
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W', borrow=True)
        # Инициализация вектора сдвига
        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # Вычисление вектора принадлежности к каждому классу
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b).flatten(1)

        # Вычисление наиболее вероятного класса
        self.y_pred = T.argmax(self.p_y_given_x)

        # parameters of the model
        self.params = [self.W, self.b]

        # L1 norm ; регуляризатор
        self.L1 = 0
        self.L1 += abs(self.W.sum())

        # square of L2 norm ; ещё один регуляризатор
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()

    def negative_log_likelihood(self, y):
        # self.p_y_given_x - вектор принадлежности к каждому классу
        return -T.log(self.p_y_given_x)[y]

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

def Iden(x):
    y = x
    return(y)


class CNNForSentences(object):
    """
    Свёрточная сеть с одним свёрточным слоем
    """
    def __init__(self, input, n_out, n_hidden,
                 n_filters, n_chanels, window, word_dimension, activation=T.tanh, seed=0, k_max=1):
        """
        :type input: theano.tensor.dtensor4
        :param input: символичный тензор предложений формата:
                      (количество предложений = 1(всегда), количество каналов - обычно 1,
                       высота = длина предложения,
                       ширина = размерность ветора слова)

        :param n_out: количество целевых классов классификации
        :param n_hidden:  число нейронов скрытого полносвязного слоя
        :param n_filters: число фильтров свёртки
        :param n_chanels: число входных каналов
        :param window: размер окна для фильтров
        :param activation: активационная функция
        :param seed: начальное значение для генератора случайных чисел
        """
        self.softmax = T.nnet.softmax

        rng = numpy.random.RandomState(seed)
        # assert word_dimension == input.shape[3]

        self.layer0 = ConvLayerForSentences(rng, input_data=input, filter_shape=(n_filters, n_chanels,
                                                                                 window,
                                                                                 word_dimension))
        # Записывает в self.output 4D тензор, размера: (batch size (1), nfilters, output row, output col(1))
        # output col == 1, т.к. ширина фильтра = word_dimension (ширине изображения)
        # Меняю форму, на: (nfilters, output row, output col, batch size (1))
        layer1_input = self.layer0.output.dimshuffle(1, 2, 0, 3)
        # Меняю форму на: ( nfilters, output row * output col(1) * batch size(1) )
        layer1_input = layer1_input.flatten(2)

        if k_max == 1:
            self.layer1 = MaxOverTimePoolingLayer(layer1_input)
        else:
            self.layer1 = KMaxPoolingLayer(layer1_input, k_max)
        layer2_input = self.layer1.output.flatten(1)
        # После этого слоя осталось ровно n_filters * k элементов
        self.layer2 = FullyConnectedLayer(rng, layer2_input, n_in=n_filters * k_max,
                                          n_out=n_hidden, activation=activation)

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = SoftmaxLayer(input=self.layer2.output, n_in=n_hidden, n_out=n_out)

        # CNN regularization
        self.L1 = self.layer3.L1
        self.L2_sqr = self.layer3.L2_sqr

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + self.layer2.params + self.layer0.params

        self.y_pred = self.layer3.y_pred
        self.p_y_given_x = self.layer3.p_y_given_x

        self.loss = lambda y: self.layer3.negative_log_likelihood(y)


def text_to_word_list(text, remove_stopwords=False):
    text_words = re.sub("[^a-zA-Z]", " ", text)
    words = text_words.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return words


def make_feature_matrix(words, model):
    assert len(words) > 0
    # feature_matrix = np.zeros((len(words), num_features), dtype="float32")
    feature_matrix = []
    # counter = 0.
    for word in words:
        if word in model.vocab:
            feature_matrix.append(list(model[word]))
            # feature_matrix[counter] = model[word]
            # counter += 1
        else:
            # TODO: зачем я это делаю?
            feature_matrix.append([0 for i in xrange(model.layer1_size)])
    assert len(feature_matrix) > 0
    feature_matrix = numpy.asarray(feature_matrix)
    if len(feature_matrix) == 1:
        feature_matrix = feature_matrix.reshape(1, feature_matrix.shape[0])
    return feature_matrix


# может вернуть None!
def text_to_matrix(text, model):
    words = text_to_word_list(text, remove_stopwords=False)
    if len(words) == 0:
        return None
    matrix = make_feature_matrix(words, model)
    return matrix


class CNNTextClassifier(BaseEstimator):

    def __init__(self, learning_rate=0.1, n_epochs=3, activation='tanh', window=5,
                 n_hidden=10, n_filters=25, pooling_type='max_overtime',
                 L1_reg=0.00, L2_reg=0.00, n_out=2, word_dimension=100,
                 seed=0, model_path=None, k_max=1):
        """
        :param learning_rate: темп обучения
        :param n_epochs: количество эпох обучения
        :type activation: string, варианты: 'tanh', 'sigmoid', 'relu', 'cappedrelu'
        :param activation: вид активационной функции
        :param window: размер "окна" для обработки близких друг к другу слов
        :param n_hidden: число нейронов в скрытом слое
        :param n_filters: число фильтров
        :param pooling_type: тип пулинга, пока что доступен только max_overtime пулинг
        :param L1_reg: параметр для регуляризации
        :param L2_reg: параметр для регуляризации
        :param n_out: количество классов для классификации
        :param word_dimension: размерность слов
        :param seed: начальное значение для генератора случайных чисел
        :type model_path: string / None
        :param model_path: путь к сохранённой модели word2vec, если путь не указан, используется
                        стандартная предобученная модель
        """
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.n_epochs = int(n_epochs)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.n_out = n_out
        self.n_filters = n_filters
        self.pooling_type = pooling_type
        self.window = window
        self.word_dimension = word_dimension
        self.seed = seed
        self.is_ready = False
        self.model_path = model_path
        self.k_max = k_max

        print "Model word2vec is loading from %s." % self.model_path
        try:
            self.model = Word2Vec.load_word2vec_format(self.model_path, binary=True)
        except UnicodeDecodeError:
            self.model = Word2Vec.load(self.model_path)
        print "Model word2vec was loaded."
        assert self.model.layer1_size == self.word_dimension

    def ready(self):
        """
        this function is called from "fit"
        """
        #input
        self.x = T.tensor4('x')
        #output (a label)
        self.y = T.lscalar('y')

        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError

        self.cnn = CNNForSentences(input=self.x,
                       n_out=self.n_out, activation=activation,
                       n_hidden=self.n_hidden, n_filters=self.n_filters, n_chanels=1, window=self.window,
                       word_dimension=self.word_dimension, seed=self.seed, k_max=self.k_max)

        #self.cnn.predict expects one input.
        #we wrap those functions and pad as necessary in 'def predict' and 'def predict_proba'
        self.predict_wrap = theano.function(inputs=[self.x], outputs=self.cnn.y_pred)
        self.predict_proba_wrap = theano.function(inputs=[self.x], outputs=self.cnn.p_y_given_x)
        self.is_ready = True

    def score(self, x_data, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        z : float

        """
        return numpy.mean(self.predict(x_data) == y)

    def fit(self, x_train, y_train, x_test=None, y_test=None, n_epochs=None):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.
        :type x_train: list(string) или numpy.array(string)
        :param x_train: входные данные - список из текстов
        :type y_train: list(int)
        :param y_train: целевые значения для каждого текста

        :type n_epochs: int/None
        :param n_epochs: used to override self.n_epochs from init.
        """
        assert max(y_train) < self.n_out
        assert min(y_train) >= 0
        assert len(x_train) == len(y_train)
        print "Feature selection..."
        x_train_matrix = self.__feature_selection(x_train)
        if x_test is not None:
            assert(y_test is not None)
            assert len(x_test) == len(y_test)
            interactive = True
            x_test_matrix = self.__feature_selection(x_test)
        else:
            interactive = False
        print "Feature selection finished"

        # подготовим CNN
        if not self.is_ready:
            self.ready()

        self.compute_error = theano.function(inputs=[self.x, self.y],
                                             outputs=self.cnn.loss(self.y))

        cost = self.cnn.loss(self.y) + self.L1_reg * self.cnn.L1 + self.L2_reg * self.cnn.L2_sqr

        # Создаём список градиентов для всех параметров модели
        grads = T.grad(cost, self.cnn.params)

        # train_model это функция, которая обновляет параметры модели с помощью SGD
        # Так как модель имеет много парамметров, было бы утомтельным вручную создавать правила
        # обновления для каждой модели, поэтому нужен updates list для автоматического
        # прохождения по парам (params[i], grads[i])
        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(self.cnn.params, grads)
        ]

        self.train_model = theano.function([self.x, self.y], cost, updates=updates)

        if n_epochs is None:
            n_epochs = self.n_epochs

        n_train_samples = x_train_matrix.shape[0]
        if interactive:
            n_test_samples = x_test_matrix.shape[0]

        rng = numpy.random.RandomState(self.seed)
        visualization_frequency = min(2000, n_train_samples - 1)
        epoch = 0
        while epoch < n_epochs:
            epoch += 1
            # compute loss on training set
            print "start epoch %d: this TRAIN 500 SCORE: %f"\
                  % (epoch, float(self.score(x_train[0:500], y_train[0:500])))

            indices = rng.permutation(n_train_samples)
            for idx, real_idx in enumerate(indices):
                # Если матрица пустая - тут пропускаю
                if x_train_matrix[real_idx] is None:
                    continue
                x_current_input = x_train_matrix[real_idx].reshape(1, 1, x_train_matrix[real_idx].shape[0],
                                                              x_train_matrix[real_idx].shape[1])
                cost_ij = self.train_model(x_current_input, y_train[real_idx])

                if idx % visualization_frequency == 0 and idx > 0:
                    # print "train cost_ij = ", cost_ij
                    if interactive:
                        test_losses = [self.compute_error(x_test_matrix[i].reshape(1, 1,
                                                                                   x_test_matrix[i]
                                                                                   .shape[0],
                                                          x_test_matrix[i].shape[1]), y_test[i])
                                       for i in xrange(n_test_samples) if x_test_matrix[i] is not None]
                        this_test_loss = numpy.mean(test_losses)
                        print "epoch %d, review %d: this test losses(score): %f, this TEST MEAN " \
                              "SCORE: %f" % (epoch, idx, float(this_test_loss),
                                             float(self.score(x_test, y_test)))

                    else:
                        # compute loss on training set
                        train_losses = [self.compute_error(x_train_matrix[i].reshape(1, 1,
                                                                                     x_train_matrix[i].shape[0],
                                                           x_train_matrix[i].shape[1]), y_train[i])
                                        for i in xrange(n_train_samples)]
                        this_train_loss = numpy.mean(train_losses)
                        print self.score(x_train, y_train)
                        print "cost_ij = ", cost_ij
                        print "epoch %d, review %d: this train losses: %f"\
                              % (epoch, real_idx, float(this_train_loss))

        print "Fitting was finished. Test score:"
        print self.score(x_test, y_test)

    def predict(self, data):
        if isinstance(data[0], str) or isinstance(data[0], unicode):
            matrix_data = self.__feature_selection(data)
        else:
            print type(data[0])
            matrix_data = data
        if isinstance(matrix_data, list) or isinstance(matrix_data, numpy.matrix):
            matrix_data = numpy.array(matrix_data)
        return [self.predict_wrap(matrix_data[i].reshape(1, 1, matrix_data[i].shape[0],
                                  matrix_data[i].shape[1])) for i in xrange(matrix_data.shape[0])]

    def predict_proba(self, data):
        if isinstance(data[0], str):
            matrix_data = self.__feature_selection(data)
        else:
            matrix_data = data
        if isinstance(matrix_data, list) or isinstance(matrix_data, numpy.matrix):
            matrix_data = numpy.array(matrix_data)
        return [self.predict_proba_wrap(matrix_data[i].reshape(1, 1, matrix_data[i].shape[0],
                                        matrix_data[i].shape[1])) for i in xrange(matrix_data.shape[0])]

    def __getstate__(self):
        """ Return state sequence."""

        #check if we're using ubc_AI.classifier wrapper,
        #adding it's params to the state
        if hasattr(self, 'orig_class'):
            superparams = self.get_params()
            #now switch to orig. class (MetaCNN)
            oc = self.orig_class
            cc = self.__class__
            self.__class__ = oc
            params = self.get_params()
            for k, v in superparams.iteritems():
                params[k] = v
            self.__class__ = cc
        else:
            params = self.get_params()  #sklearn.BaseEstimator
        if hasattr(self, 'cnn'):
            weights = [p.get_value() for p in self.cnn.params]
        else:
            weights = []
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)
        if hasattr(self, 'cnn'):
            for param in self.cnn.params:
                param.set_value(i.next())
        else:
            print "Error in function _set_weights: there is no cnn"

    def __setstate__(self, state):
        """ Set parameters from state sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
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
            if len(weights) > 0:
                self._set_weights(weights)
            else:
               print "Error in function __setstate__: there is no weights"
            self.__class__ = cc
        else:
            self.set_params(**params)
            self.ready()
            self._set_weights(weights)

    def load(self, path):
        """ Load model parameters from path. """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.__setstate__(state)

    def save_state(self, path):
        with open(path, 'w') as f:
            pickle.dump(self.__getstate__(), f)

    def __feature_selection(self, text_data):
        text_data_as_matrix = []
        for i, text in enumerate(text_data):
            if not isinstance(text, str) and not isinstance(text, numpy.unicode):
                print type(text)
                raise AttributeError("feature selection error: not string format")
            # text_to_matrix может вернуть None!
            matrix = text_to_matrix(text, self.model)
            if matrix is None:
                print "Warning: {0}: '{1}' hasn't meaningful words!".format(i, text)
            elif matrix.shape[0] < self.window + self.k_max - 1:
                print "Warning: {0} sentence's length ({1}) is less then window + k_max - 1 ({2})"\
                    .format(i, matrix.shape[0], self.window + self.k_max - 1)
                print "sentence: {}".format(text)
                continue
            text_data_as_matrix.append(matrix)
        return numpy.asarray(text_data_as_matrix)

    def get_cnn_params(self):
        if hasattr(self, 'cnn'):
            return self.cnn.params
        else:
            print "Error in function _set_weights: there is no cnn"
