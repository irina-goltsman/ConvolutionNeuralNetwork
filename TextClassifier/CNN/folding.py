# -*- coding: utf-8 -*-
import theano.tensor as T
from lasagne.layers.base import Layer


class FoldingLayer(Layer):
    """
        В статье Kalchbrenner описан folding следующим образом: каждые 2 колонки сливаются в 1.
        В целях увеличения эффективности, лучше использовать reshape функцию, которая сливает каждую x и x+n/2 колонку.
        """
    def __init__(self,incoming,**kwargs):
        super(FoldingLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], input_shape[3]/2

    # TODO: разберись, правда ли можно юзать reshape функцию
    def get_output_for(self, input, **kwargs):
        long_shape = (self.input_shape[0], self.input_shape[1], -1, 2)
        long_cols = T.reshape(input, long_shape)
        # суммирую 2 колонки
        summed = T.sum(long_cols, axis=3, keepdims=True)
        # reshape them back
        folded_output = T.reshape(summed, (self.input_shape[0], self.input_shape[1], -1, self.input_shape[3]/2))

        return folded_output