#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
from collections import OrderedDict
from common.datasets import TEXT
from common.util import one_hot
import sys
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""


class RNNcore(object):
    def __init__(self, weights_xh, weights_hh, bias_h, weights_hy, bias_y):
        # init para
        self.x = None

        self.Wxh = weights_xh
        self.Whh = weights_hh
        self.bh = bias_h
        self.Why = weights_hy
        self.by = bias_y

        self.h = OrderedDict()
        self.y = OrderedDict()

        self.d_Wxh = None
        self.d_Whh = None
        self.d_bh = None
        self.d_Why = None
        self.d_by = None

        self.h[-1] = None
        self.h_size = self.Wxh.shape[-1]
        # self.h[-1] = np.zeros(1, 100)

        # self.d_y = OrderedDict()
        self.d_h = OrderedDict()

        self.step_num = None
        self.batch_size = None
        self.class_num = None
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

    def forward(self, x_batch):
        assert x_batch.ndim == 3
        self.x = x_batch
        self.step_num, self.batch_size, self.class_num = self.x.shape  # step x batch x class

        self.h[-1] = np.zeros((self.batch_size, self.h_size))
        for idx, x_step in enumerate(self.x):
            self.h[idx] = np.tanh(np.dot(x_step, self.Wxh) + np.dot(self.h[idx-1], self.Whh) + self.bh)
            self.y[idx] = np.dot(self.h[idx], self.Why) + self.by

        return self.y

    def backward(self, d_y_bp):
        self.d_Why = 0
        self.d_Wxh = 0
        self.d_Whh = 0
        for idx in range(self.step_num):
            tmp = np.dot(self.h[idx].T, d_y_bp[idx])
            self.d_Why += tmp
        for idx in range(self.step_num)[::-1]:
            if idx == self.step_num:
                self.d_h[idx] = np.dot(d_y_bp[idx], self.Why.T)
            else:
                self.d_h[idx] = np.dot(d_y_bp[idx], self.Why.T) + np.dot(self.d_h[idx+1], self.Whh.T)
        for idx, x_step in enumerate(self.x):
            self.d_Wxh += np.dot(x_step.T, self.d_h[idx])
            self.d_Whh += np.dot(self.h[idx-1].T, self.d_h[idx])

        return None  # no lower level layer

if __name__ == '__main__':
    text = TEXT(r"datasets/text")
    text_data = text.load()
    vocab_num = len(text.corpus_chars)
    print(text_data[:10])
    print(text.idx_to_char(text_data[:10]))
    x, y = text.get_one_batch_random(text_data, batch_size=2, steps_num=5)
    x = one_hot(x, class_num=vocab_num)
    print("x.shape", x.shape)

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
