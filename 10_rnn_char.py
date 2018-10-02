#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
from collections import OrderedDict
from common import functions
import sys
from common.util import one_hot
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

        h_size = self.Wxh.shape[-1]
        self.h[-1] = np.zeros((1, h_size))
        # self.h[-1] = np.zeros(1, 100)

        # self.d_y = OrderedDict()
        self.d_h = OrderedDict()

        self.step_num = None

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
        ret_str = "Affine layer: {} dot {} + {} dot {} + {} => {}".format(x_shape, self.Wxh.shape,
                                                                          h_shape, self.Whh.shape,
                                                                          self.bh.shape,
                                                                          h_shape)
        return ret_str

    def forward(self, x_batch):
        assert self.x.ndim == 3
        self.x = x_batch
        self.step_num = self.x.shape[0]
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


# hyper-parameters
batch_size = 2
step_num = 3
# vocab_size = 2581
vocab_size = 10
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(vocab_size, hidden_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
bh = np.zeros((hidden_size, 1))  # hidden bias

Why = np.random.randn(hidden_size, vocab_size) * 0.01  # hidden to output
by = np.zeros((vocab_size, 1))  # output bias

rnn = RNNcore(weights_xh=Wxh, weights_hh=Whh, bias_h=bh, weights_hy=Why, bias_y=by)
batch_x = one_hot(np.arange(batch_size*step_num), class_num=10)
print(batch_x)

