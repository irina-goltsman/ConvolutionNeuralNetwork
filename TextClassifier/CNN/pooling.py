# -*- coding: utf-8 -*-
import theano.tensor as T
from lasagne.layers.base import Layer


class KMaxPoolLayer(Layer):

    def __init__(self, incoming, k, **kwargs):
        super(KMaxPoolLayer, self).__init__(incoming, **kwargs)
        self.k = k

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.k, input_shape[3])

    def get_output_for(self, input, **kwargs):
        return self.kmaxpooling(input, self.k)


    def kmaxpooling(self,input,k):
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

        return input[dim0, dim1, dim2, dim3].reshape((input.shape[0], input.shape[1], k, input.shape[3]))


