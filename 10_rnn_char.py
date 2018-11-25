#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
from collections import OrderedDict
from common.datasets import TEXT
from common.util import one_hot, simple_grad_clipping
from common.functions import softmax, cross_entropy
import sys
import os
"""
Reference:
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""


class SimpleRNN(object):
    def __init__(self, weights_xh, weights_hh, bias_h, weights_hy, bias_y):
        # init para
        self.x = None
        self.t = None

        self.Wxh = weights_xh
        self.Whh = weights_hh
        self.bh = bias_h
        self.Why = weights_hy
        self.by = bias_y

        self.h_size = self.Wxh.shape[-1]

        self.h = OrderedDict()
        self.y = OrderedDict()
        self.p = OrderedDict()

        self.d_Wxh = None
        self.d_Whh = None
        self.d_bh = None
        self.d_Why = None
        self.d_by = None

        self.seq_length = None
        self.batch_size = None
        self.vocab_size = None

        # self.params = {}  # init parameters
        # self.__init_params()
        logging.info('M@{}, C@{}, F@{}, W_xh shape: {}, W_hh shape: {}'.format(__name__,
                                                                               self.__class__.__name__,
                                                                               sys._getframe().f_code.co_name,
                                                                               self.Wxh.shape,
                                                                               self.Whh.shape))
        logging.info('M@{}, C@{}, F@{}, b_h shape: {}'.format(__name__,
                                                              self.__class__.__name__,
                                                              sys._getframe().f_code.co_name,
                                                              self.bh.shape))

    def __str__(self):
        if hasattr(self.x, 'shape'):
            batch_size = self.x.shape[0]
        else:
            batch_size = '?'
        (x_feature_count, h_feature_count) = self.Wxh.shape
        x_shape = (batch_size, x_feature_count)
        h_shape = (batch_size, h_feature_count)
        ret_str = "RNN core layer: {} dot {} + {} dot {} + {} => {}".format(x_shape, self.Wxh.shape,
                                                                            h_shape, self.Whh.shape,
                                                                            self.bh.shape,
                                                                            h_shape)
        return ret_str

    def update_data(self, x_batch, t_batch):
        assert x_batch.ndim == 3
        self.x = x_batch
        self.t = t_batch

    def forward(self):
        self.seq_length, self.batch_size, self.vocab_size = self.x.shape  # seq_length * batch_size * vocab_size
        if not self.h:  # when self.h is empty
            self.h[-1] = np.zeros((self.batch_size, self.h_size))
        else:
            self.h[-1] = np.copy(self.h[self.seq_length - 1])
        loss = 0
        for idx in range(self.seq_length):
            self.h[idx] = np.tanh(np.dot(self.x[idx], self.Wxh) + np.dot(self.h[idx-1], self.Whh) + self.bh)
            self.y[idx] = np.dot(self.h[idx], self.Why) + self.by
            self.p[idx] = softmax(self.y[idx])
            loss += cross_entropy(self.p[idx], self.t[idx])
        return loss

    def backward(self, bp=1):
        self.d_Why = 0
        self.d_by = 0
        self.d_Wxh = 0
        self.d_Whh = 0
        self.d_bh = 0

        d_h_next = np.zeros_like(self.h[0])
        for idx in reversed(range(self.seq_length)):
            # dL/dy
            d_y = self.p[idx] - one_hot(self.t[idx], class_num=self.vocab_size)  # gradient of cross-entropy: d_y = y - t

            # dL/dWhy dL/dby
            self.d_Why += np.dot(self.h[idx].T, d_y)  # gradient of Why
            self.d_by += d_y  # gradient of by

            # dL/dh
            if idx == self.seq_length - 1:
                d_h = np.dot(d_y, self.Why.T)
            else:
                d_h = np.dot(d_y, self.Why.T) + d_h_next

            # dL/dh(before nonlinear ops @ tanh)
            d_h_raw = (1 - self.h[idx]**2) * d_h

            self.d_Wxh += np.dot(self.x[idx].T, d_h_raw)
            self.d_Whh += np.dot(self.h[idx-1].T, d_h_raw)
            self.d_bh += d_h_raw  # gradient of bh

            d_h_next = np.dot(d_h_raw, self.Whh.T)

            for grads in [self.d_Wxh, self.d_Whh, self.d_Why, self.d_bh, self.d_by]:
                simple_grad_clipping(grads)  # clip to mitigate exploding gradients
        self.d_by = np.sum(self.d_by, axis=0)
        self.d_bh = np.sum(self.d_bh, axis=0)
        return None  # no lower level layer

    def update_params(self, learning_rate=1e-3):
        self.Wxh -= learning_rate * self.d_Wxh
        self.Whh -= learning_rate * self.d_Whh
        self.bh -= learning_rate * self.d_bh
        self.Why -= learning_rate * self.d_Why
        self.by -= learning_rate * self.d_by

    def init_rnn_state(self, batch_size, num_hiddens):
        return np.zeros(shape=(batch_size, num_hiddens))

    def predict(self, prefix, output_num):
        # inputs 和 outputs 皆为 num_steps 个形状为（batch_size，vocab_size）的矩阵。
        H = self.init_rnn_state(1, self.h_size)
        outputs = []
        for X in prefix:
            H = np.tanh(np.dot(X, self.Wxh) + np.dot(H, self.Whh) + self.bh)
            Y = np.dot(H, self.Why) + self.by
            outputs.append(Y)
        for t in range(output_num):
            X = np.array(outputs[-1])
            H = np.tanh(np.dot(X, self.Wxh) + np.dot(H, self.Whh) + self.bh)
            Y = np.dot(H, self.Why) + self.by
            outputs.append(Y)

        outputs = np.array(outputs)
        outputs = np.argmax(outputs, axis=2)
        return outputs, H


if __name__ == '__main__':
    if os.name == 'nt':
        file_root_path = r"datasets\\text"
    elif os.name == 'posix':
        file_root_path = r"datasets/text"
    else:
        raise Exception("Invalid os!", os.name)
    text = TEXT(file_root_path)
    text_data = text.load()
    vocab_size = len(text.corpus_chars)
    print(text_data[:10])
    print(text.idx_to_char(text_data[:10]))

    # hyper-parameters
    batch_size = 10
    hidden_size = 100  # size of hidden layer of neurons
    seq_length = 5  # number of steps to unroll the RNN for
    learning_rate = 1e-2

    # x, t = text.get_one_batch_random(text_data, batch_size=batch_size, steps_num=seq_length)
    g = text.data_iter_consecutive(text_data, batch_size=batch_size, steps_num=seq_length)
    x, t = next(g)
    x = one_hot(x, class_num=vocab_size)
    print("x.shape:", x.shape)
    print("t.shape:", t.shape)

    # weights init
    weight_init_std = 0.01
    params = OrderedDict()

    params['weights_xh'] = weight_init_std * np.random.randn(vocab_size, hidden_size)  #
    params['weights_hh'] = weight_init_std * np.random.randn(hidden_size, hidden_size)  #
    params['bias_h'] = np.zeros(hidden_size)  #
    params['weights_hy'] = weight_init_std * np.random.randn(hidden_size, vocab_size)  #
    params['bias_y'] = np.zeros(vocab_size)  #

    network = SimpleRNN(**params)
    max_iteration = 10000
    epoch = 100
    loss_list = []
    for i in range(max_iteration):
        network.update_data(x, t)
        loss = network.forward()
        loss_list.append(loss)
        network.backward()
        network.update_params()
        if i % epoch == 0:
            print("loss:", loss)

    pre = "分开"
    pre_list = []
    for cha in pre:
        pre_list.append(text.char_to_idx_dict[cha])
    pre = np.array(pre_list).reshape(-1, 1)
    pre = one_hot(pre, class_num=vocab_size)
    pre = pre.reshape(2, 1, -1)
    print('pre.shape:', pre.shape)

    output, _ = network.predict(prefix=pre, output_num=10)
    print('output.shape:', output.shape)

    out_list =[]
    for o in output.flatten():
        out_list.append(text.idx_to_char_dict[o])
    print(out_list)

    """
    # hyper-parameters
    batch_size = 2  # batch size
    vocab_size = 128  # size of full vocabulary dataset
    hidden_size = 100  # size of hidden layer of neurons
    seq_length = 5  # number of steps to unroll the RNN for
    learning_rate = 1e-1

    # model parameters
    Wxh = np.random.randn(vocab_size, hidden_size) * 0.01  # input to hidden
    Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
    bh = np.zeros((1, hidden_size))  # hidden bias

    Why = np.random.randn(hidden_size, vocab_size) * 0.01  # hidden to output
    by = np.zeros((1, vocab_size))  # output bias

    rnn = RNNcore(weights_xh=Wxh, weights_hh=Whh, bias_h=bh, weights_hy=Why, bias_y=by)
    batch_x = one_hot(np.random.choice(vocab_size, size=(seq_length, batch_size)), class_num=vocab_size)
    print(batch_x.shape)
    y = rnn.forward(batch_x)
    for key, value in y.items():
        print(key, value.shape)
    """
