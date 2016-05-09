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

#TODO: юзать больше карт: from 100 to 600
#TODO: 4. добавить возможность изменять векторное представление слов в процессе обучения
#TODO: 2. реализовать линейную модель для сравнения качества классификации
#TODO: заюзать glove вместо/вместе word2vec
#TODO: попробовать разные активационные функции, в том числе Ident
#TODO: 5. добавить dropout-слой - "use a small dropout rate(0.0-0.5) and a large max norm constraint"
#TODO: лучше юзай 1-max-pooling для предложений
#TODO: попробовать юзать adaboost и обучать-таки пачками
#TODO: реализовать 2х слойную модель с k-max-pooling.
#TODO: 3. добавить функцию заполнения параметров модели рандомными значениями

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


class CNNForSentences(object):
    """
    Свёрточная сеть с одним свёрточным слоем
    """
    def __init__(self, input, n_out, n_hidden, n_filters, n_chanels, windows, word_dimension,
                 activation=T.tanh, seed=0, k_max=1):
        """
        :type input: theano.tensor.dtensor4
        :param input: символичный тензор предложений формата:
                      (количество предложений = 1(всегда), количество каналов - обычно 1,
                       высота = длина предложения,
                       ширина = размерность ветора слова)

        :param n_out: количество целевых классов классификации
        :param n_hidden:  число нейронов скрытого полносвязного слоя
        :param n_filters: число фильтров для каждого вида свёртки
        :param n_chanels: число входных каналов
        :type windows: list
        :param windows: размеры окон для фильтров
        :param activation: активационная функция
        :param seed: начальное значение для генератора случайных чисел
        """
        self.softmax = T.nnet.softmax

        rng = numpy.random.RandomState(seed)
        # assert word_dimension == input.shape[3]

        self.layers0 = list()
        layer2_inputs = list()
        for window in windows:
            layer0 = ConvLayerForSentences(rng, input_data=input, filter_shape=(n_filters, n_chanels,
                                                                                     window, word_dimension),
                                                activation=activation)
            # Записывает в self.output 4D тензор, размера: (batch size (1), nfilters, output row, output col(1))
            # output col == 1, т.к. ширина фильтра = word_dimension (ширине изображения)
            # Меняю форму, на: (nfilters, output row, output col, batch size (1))
            layer1_input = layer0.output.dimshuffle(1, 2, 0, 3)
            self.layers0.append(layer0)
            # Меняю форму на: ( nfilters, output row * batch size(1) * output col(1) )
            layer1_input = layer1_input.flatten(2)

            if k_max == 1:
                self.layer1 = MaxOverTimePoolingLayer(layer1_input)
            else:
                self.layer1 = KMaxPoolingLayer(layer1_input, k_max)
            layer2_input = self.layer1.output.flatten(1)
            layer2_inputs.append(layer2_input)
        # TODO: если буду юзать мини-батчи, то тут сложнее
        layer2_input = T.concatenate(layer2_inputs)
        # После этого слоя осталось ровно n_filters * k элементов
        self.layer2 = FullyConnectedLayer(rng, layer2_input, n_in=n_filters * k_max * len(windows),
                                          n_out=n_hidden, activation=activation)

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = SoftmaxLayer(input=self.layer2.output, n_in=n_hidden, n_out=n_out)

        # CNN regularization
        self.L1 = self.layer3.L1
        self.L2_sqr = self.layer3.L2_sqr

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + self.layer2.params
        # TODO: проверь, возможно есть смысл записать параметры наоборот
        for layer0 in self.layers0:
            self.params += layer0.params

        self.y_pred = self.layer3.y_pred
        self.p_y_given_x = self.layer3.p_y_given_x

        self.loss = lambda y: self.layer3.negative_log_likelihood(y)


def text_to_word_list(text, remove_stopwords=False):
    #TODO: можно попробовать включить 0-9(),!?
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
    for word in words:
        if word in model.vocab:
            feature_matrix.append(list(model[word]))
        else:
            # TODO: заменить на рандомную инициализацию
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

    def __init__(self, learning_rate=0.1, n_epochs=3, activation='tanh', windows=[5],
                 n_hidden=10, n_filters=25,
                 L1_reg=0.00, L2_reg=0.00, n_out=2, seed=0, k_max=1, word_dimension=100,
                 model_path="../models/100features_40minwords_10context"):
        """
        :param learning_rate: темп обучения
        :param n_epochs: количество эпох обучения
        :type activation: string, варианты: 'tanh', 'sigmoid', 'relu', 'cappedrelu', 'iden'
        :param activation: вид активационной функции
        :param windows: размеры окон для обработки близких друг к другу слов
        :param n_hidden: число нейронов в скрытом слое
        :param n_filters: число фильтров для каждого вида окна
        :param L1_reg: параметр для регуляризации
        :param L2_reg: параметр для регуляризации
        :param n_out: количество классов для классификации
        :param word_dimension: размерность слов
        :param seed: начальное значение для генератора случайных чисел
        :type model_path: string / None
        :param model_path: путь к сохранённой модели векторного представления слов
        :type k_max: int (>=1)
        :param k_max: при k==1 используется max-overtime-pooling, иначе k-max-pooling
        """
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.n_epochs = int(n_epochs)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.n_out = n_out
        self.n_filters = n_filters
        self.windows = windows
        self.word_dimension = word_dimension
        self.seed = seed
        self.is_ready = False
        self.model_path = model_path
        self.k_max = k_max

        print "Word embedding model is loading from %s." % self.model_path
        try:
            self.model = Word2Vec.load_word2vec_format(self.model_path, binary=True)
        except UnicodeDecodeError:
            self.model = Word2Vec.load(self.model_path)
        print "Word embedding model has been loaded."
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
            activation = lambda x: T.maximum(0.0, x)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        elif self.activation == 'iden':
            activation = lambda x: x
        else:
            raise NotImplementedError

        self.cnn = CNNForSentences(input=self.x, n_out=self.n_out, activation=activation, n_hidden=self.n_hidden,
                                   n_filters=self.n_filters, n_chanels=1, windows=self.windows,
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

    def fit(self, x_train, y_train, x_test=None, y_test=None, n_epochs=None, validation_part=0.1,
            visualization_frequency=5000):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.
        :type x_train: list(string) или numpy.array(string)
        :param x_train: входные данные - список из текстов
        :type y_train: list(int)
        :param y_train: целевые значения для каждого текста

        :type n_epochs: int/None
        :param n_epochs: used to override self.n_epochs from init.
        :type validation_part: float
        :param validation_part: доля тренеровочных данных, которые станут валидационной выборкой
        :type visualization_frequency: int/None
        :param visualization_frequency: если не None, то каждые visualization_frequency интераций
                                        будет выводиться результат модели на валидационной выборке
        """
        assert max(y_train) < self.n_out
        assert min(y_train) >= 0
        assert len(x_train) == len(y_train)

        n_valid_samples = int(len(x_train) * validation_part)
        x_valid = x_train[-n_valid_samples:]
        y_valid = y_train[-n_valid_samples:]
        x_train = x_train[0:n_valid_samples]
        y_train = y_train[0:n_valid_samples]

        print "Feature selection..."
        x_train_matrix = self.__feature_selection(x_train)
        x_valid_matrix = self.__feature_selection(x_valid)
        n_train_samples = x_train_matrix.shape[0]
        n_valid_samples = x_valid_matrix.shape[0]

        if x_test is not None:
            assert len(x_test) == len(y_test)
            x_test_matrix = self.__feature_selection(x_test)
            n_test_samples = x_test_matrix.shape[0]

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

        if n_epochs is not None:
            self.n_epochs = int(n_epochs)

        rng = numpy.random.RandomState(self.seed)
        visualization_frequency = min(visualization_frequency, n_train_samples - 1)
        epoch = 0
        best_valid_score, best_epoch_num = 0, 0
        mean_test_loss, test_score = None, None
        while epoch < self.n_epochs:
            epoch += 1
            print "start epoch %d: this valid score: %f" % (epoch, float(self.score(x_valid_matrix, y_valid)))

            indices = rng.permutation(n_train_samples)
            for cur_idx, real_idx in enumerate(indices):
                # Если матрица пустая - тут пропускаю
                if x_train_matrix[real_idx] is None:
                    continue
                x_current_input = x_train_matrix[real_idx].reshape(1, 1, x_train_matrix[real_idx].shape[0],
                                                              x_train_matrix[real_idx].shape[1])
                train_cost = self.train_model(x_current_input, y_train[real_idx])

                if cur_idx % visualization_frequency == 0 and cur_idx > 0:
                    valid_losses = [self.compute_error(X.reshape(1, 1, X.shape[0], X.shape[1]), y)
                                   for X, y in zip(x_valid_matrix, y_valid) if X is not None]
                    mean_valid_loss = numpy.mean(valid_losses)
                    print "epoch %d, review %d: this mean valid losses: %f, this mean valid score: %f" \
                          % (epoch, cur_idx, float(mean_valid_loss), self.score(x_valid_matrix, y_valid))
                    print "current train cost: %f" % train_cost
            current_valid_score = float(self.score(x_valid_matrix, y_valid))
            print "end of epoch %d: this valid score: %f" % (epoch, current_valid_score)
            if current_valid_score > best_valid_score:
                print "NEW BEST VALID SCORE: %f" % current_valid_score
                best_valid_score = current_valid_score
                best_epoch_num = epoch
                if x_test is not None:
                    test_losses = [self.compute_error(X.reshape(1, 1, X.shape[0], X.shape[1]), y)
                                    for X, y in zip(x_test_matrix, y_test) if X is not None]
                    mean_test_loss = numpy.mean(test_losses)
                    test_score =  self.score(x_test_matrix, y_test)
                    print "epoch %d: this mean TEST LOSSES: %f, this MEAN TEST SCORE: %f" \
                          % (epoch, float(mean_test_loss), test_score)

        print "Fitting was finished."
        print "Train score: %f" % self.score(x_train_matrix, y_train)
        print "Valid score: %f" % self.score(x_valid_matrix, y_valid)
        print "Best valid score: %f" % best_valid_score
        print "Best number of epoch: %d" % best_epoch_num
        if x_test is not None:
            print "TEST LOSSES of best valid score: %f" % mean_test_loss
            print "TEST SCORE of best valid score: %f" % test_score
            return (mean_test_loss, test_score)

    def predict(self, data):
        if isinstance(data[0], str) or isinstance(data[0], unicode):
            matrix_data = self.__feature_selection(data)
        else:
            matrix_data = data
        if isinstance(matrix_data, list) or isinstance(matrix_data, numpy.matrix):
            matrix_data = numpy.array(matrix_data)
        return [self.predict_wrap(X.reshape(1, 1, X.shape[0], X.shape[1])) for X in matrix_data]

    def predict_proba(self, data):
        if isinstance(data[0], str):
            matrix_data = self.__feature_selection(data)
        else:
            matrix_data = data
        if isinstance(matrix_data, list) or isinstance(matrix_data, numpy.matrix):
            matrix_data = numpy.array(matrix_data)
        return [self.predict_proba_wrap(X.reshape(1, 1, X.shape[0], X.shape[1])) for X in matrix_data]

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
            elif matrix.shape[0] < max(self.windows) + self.k_max - 1:
                print "Warning: {0} sentence's length ({1}) is less then max window + k_max - 1 ({2})"\
                    .format(i, matrix.shape[0], max(self.windows) + self.k_max - 1)
                print "sentence: {}".format(text)
                continue
            text_data_as_matrix.append(matrix)
        return numpy.asarray(text_data_as_matrix)

    def get_cnn_params(self):
        if hasattr(self, 'cnn'):
            return self.cnn.params
        else:
            print "Error in function _set_weights: there is no cnn"

    def get_params_as_string(self):
        result_str = list()
        result_str.append("learning_rate = %2f" % self.learning_rate)
        result_str.append("n_hidden = %d" % self.n_hidden)
        result_str.append("n_epochs = %d" % self.n_epochs)
        result_str.append("L1_reg = %2f" % self.L1_reg)
        result_str.append("L2_reg = %2f" % self.L2_reg)
        result_str.append("activation = %s" % self.activation)
        result_str.append("n_out = %d" % self.n_out)
        result_str.append("n_filters = %d" % self.n_filters)
        result_str.append("windows = " + str(self.windows))
        result_str.append("word_dimension = %d" % self.word_dimension)
        result_str.append("seed = %d" % self.seed)
        result_str.append("model_path = %s" % self.model_path)
        result_str.append("k_max = %d" % self.k_max)
        return '\n'.join(result_str)
