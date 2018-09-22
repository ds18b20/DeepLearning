#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
from common import functions
from common.util import one_hot, im2col, col2im
import sys
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""

# data I/O
with open(r'data/text/jaychou_lyrics.txt', 'r', encoding='utf-8') as f:
    data = f.read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
print(chars[:10])
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

print(char_to_ix['湾'])
print(ix_to_char[0])

# hyper-parameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias


class AffineWithHidden(object):
    def __init__(self, weights_hx, weights_hh, bias_h):
        self.Wxh = weights_hx
        self.Whh = weights_hh
        self.bh = bias_h

        self.d_Wxh = None
        self.d_Whh = None
        self.d_bh = None

        self.x = None
        self.y = None

        logging.info(
            'M@{}, C@{}, F@{}, W_hx shape: {}, W_hh shape: {}'.format(__name__,
                                                                      self.__class__.__name__,
                                                                      sys._getframe().f_code.co_name,
                                                                      self.Wxh.shape,
                                                                      self.Whh.shape))
        logging.info(
            'M@{}, C@{}, F@{}, b_h shape: {}'.format(__name__,
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
        self.x = x_batch
        self.y = np.dot(self.x, self.W) + self.b

        return self.y

    def backward(self, d_y):
        self.d_x = np.dot(d_y, self.W.T)
        self.d_W = np.dot(self.x.T, d_y)
        self.d_b = np.sum(d_y, axis=0)
        self.d_x = self.d_x.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）

        return self.d_x

