# -*- coding: utf-8 -*-
from lasagne import *
from lasagne.layers import Layer
import lasagne.utils
import theano.tensor as T


class Conv1DLayerSplitted(Layer):
    """
        Свёртка, описанная в статье Kalchbrenner.
        Вход делится по колонкам (по частям векторов слов)
        Свёртка производится вдоль предложения, независимо для каждой компоненты векторов слов.
        Но при этом учитываются все предыдущие карты признаков.
    """
    def __init__(self, incoming, num_filters, filter_hight, border_mode="valid",
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(Conv1DLayerSplitted, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_hight = filter_hight
        self.border_mode = border_mode

        self.num_input_channels = self.input_shape[1]
        self.embedding_size = self.input_shape[3]

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            biases_shape = (self.num_filters, self.embedding_size)
            # TODO: разумно ли это?
            self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

    def get_W_shape(self):
        return (self.num_filters, self.num_input_channels, self.filter_hight, self.embedding_size)

    def get_output_shape_for(self, input_shape):
        output_length = lasagne.layers.conv.conv_output_length(input_shape[2],
                                                               self.filter_hight,
                                                               stride=1,
                                                               pad=self.border_mode)

        return (input_shape[0], self.num_filters, output_length, self.embedding_size)

    def get_output_for(self, input, **kwargs):
        if self.border_mode in ['valid', 'full']:
            input_shape_col = (self.input_shape[0], self.input_shape[1], self.input_shape[2], 1)
            new_input = input
            filter_shape = self.get_W_shape()
            filter_shape_col = (filter_shape[0], filter_shape[1], filter_shape[2], 1)

            conveds = []
            for i in range(self.embedding_size):
                conveds.append(T.nnet.conv.conv2d(new_input[:, :, :, i].dimshuffle(0, 1, 2, 'x'),
                                                  self.W[:, :, :, i].dimshuffle(0, 1, 2, 'x'),
                                                  image_shape=input_shape_col,
                                                  filter_shape=filter_shape_col,
                                                  border_mode=self.border_mode))

            conved = T.concatenate(conveds, axis=3)

        elif self.border_mode == 'same':
            raise NotImplementedError("Not implemented yet ")
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)

        if self.b is not None:
            conved = conved + self.b.dimshuffle('x', 0, 'x', 1)

        return self.nonlinearity(conved)
