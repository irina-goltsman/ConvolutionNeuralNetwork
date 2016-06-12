# -*- coding: utf-8 -*-
import theano.tensor as T
from lasagne.layers import Layer
from lasagne import init


class SentenceEmbeddingLayer(Layer):
    def __init__(self, incoming, vocab_size, word_dimension, word_embedding,
                 non_static=True, **kwargs):
        super(SentenceEmbeddingLayer, self).__init__(incoming, **kwargs)

        self.vocab_size = vocab_size
        self.word_dimension = word_dimension

        if word_embedding is None:
            word_embedding = init.Normal()

        self.W = self.add_param(word_embedding, (vocab_size, word_dimension), name="Words",
                                trainable=non_static, regularizable=non_static)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1, input_shape[1], self.word_dimension)

    def get_output_for(self, input, **kwargs):
        return self.W[T.cast(input.flatten(), dtype="int32")].reshape((input.shape[0], 1,
                                                                       input.shape[1],
                                                                       self.word_dimension))