# -*- coding: utf-8 -*-
import lasagne
import theano.tensor as T
import CNN


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


def build_1cnn(input_var, batch_size, sentence_len, vocab_size, word_dimension, word_embedding,
               non_static, windows, n_filters, activations, k_top, dropout, n_hidden, n_out):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, sentence_len),
        input_var=input_var
    )

    l_embedding = CNN.embeddings.SentenceEmbeddingLayer(
        l_in,
        vocab_size=vocab_size,
        word_dimension=word_dimension,
        word_embedding=word_embedding,
        non_static=non_static
    )
    assert len(windows) >= 1
    assert isinstance(windows[0], tuple)

    layers = list()
    for window in windows[0]:
        l_conv = lasagne.layers.conv.Conv2DLayer(
            l_embedding,
            n_filters[0],
            filter_size=(window, word_dimension),
            nonlinearity=lasagne.nonlinearities.linear
        )
        # Для фильтров разной ширины тут оставляем ровно k максимальных значений
        l_pool = CNN.pooling.KMaxPoolLayer(l_conv, k=k_top)
        l_nonlinear = lasagne.layers.NonlinearityLayer(l_pool, nonlinearity=parse_activation(activations[0]))
        layers.append(l_nonlinear)

    l_concat1 = lasagne.layers.ConcatLayer(layers, axis=1)

    l_dropout2 = lasagne.layers.DropoutLayer(l_concat1, p=dropout)

    l_dense = lasagne.layers.DenseLayer(
        l_dropout2,
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.sigmoid
    )

    l_out = lasagne.layers.DenseLayer(
        l_dense,
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    return l_out


def build_dcnn(input_var, batch_size, sentence_len, vocab_size, word_dimension, word_embedding,
               non_static, windows, n_filters, activations, k_top, dropout, n_out):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, sentence_len),
        input_var=input_var
    )

    l_embedding = CNN.embeddings.SentenceEmbeddingLayer(
        l_in,
        vocab_size=vocab_size,
        word_dimension=word_dimension,
        word_embedding=word_embedding,
        non_static=non_static
    )
    assert len(windows) == 2
    assert isinstance(windows[0], tuple)
    assert isinstance(windows[1], tuple)

    l_conv1 = CNN.Conv1DLayerSplitted(
        l_embedding,
        n_filters[0],
        filter_hight=windows[0][0],
        nonlinearity=lasagne.nonlinearities.linear
    )
    l_fold1 = CNN.folding.FoldingLayer(l_conv1)
    l_pool1= CNN.pooling.KMaxPoolLayer(l_fold1, k=sentence_len / 2)
    l_nonlinear1 = lasagne.layers.NonlinearityLayer(l_pool1, nonlinearity=parse_activation(activations[0]))

    l_conv2 = CNN.Conv1DLayerSplitted(
        l_nonlinear1,
        n_filters[1],
        filter_hight=windows[1][0],
        nonlinearity=lasagne.nonlinearities.linear
    )
    l_fold2 = CNN.folding.FoldingLayer(l_conv2)
    l_pool2 = CNN.pooling.KMaxPoolLayer(l_fold2, k=k_top)
    l_nonlinear2 = lasagne.layers.NonlinearityLayer(l_pool2, nonlinearity=parse_activation(activations[1]))

    l_dropout2 = lasagne.layers.DropoutLayer(l_nonlinear2, p=dropout)
    l_out = lasagne.layers.DenseLayer(
        l_dropout2,
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    return l_out
