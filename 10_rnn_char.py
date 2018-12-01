#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
from collections import OrderedDict
from common.datasets import TEXT
from common.util import one_hot, simple_grad_clipping
from common.functions import softmax, cross_entropy
from common.optimizer import SGD, Adam
import sys
import os
"""
Reference:
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""


class SimpleRNN(object):
    def __init__(self,
                 seq_length,
                 batch_size,
                 vocab_size,
                 hidden_size,
                 output_size,
                 weight_init_std=0.01):
        # structure init
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        # weights init
        self.params = {}
        self.params['Wxh'] = weight_init_std * np.random.randn(vocab_size, hidden_size)  #
        self.params['Whh'] = weight_init_std * np.random.randn(hidden_size, hidden_size)  #
        self.params['bh'] = np.zeros(hidden_size)  #
        self.params['Why'] = weight_init_std * np.random.randn(hidden_size, output_size)  #
        self.params['by'] = np.zeros(output_size)  #

        self.Wxh = self.params['Wxh']
        self.Whh = self.params['Whh']
        self.bh = self.params['bh']
        self.Why = self.params['Why']
        self.by = self.params['by']

        # process value
        self.h = OrderedDict()
        self.y = OrderedDict()
        self.p = OrderedDict()

        self.d_Wxh = None
        self.d_Whh = None
        self.d_bh = None
        self.d_Why = None
        self.d_by = None

        self.x = None
        self.t = None

        str_base = 'M@{}, C@{}, F@{}, W_xh shape: {}, W_hh shape: {}, b_h shape: {}'
        logging.info(str_base.format(__name__,
                                     self.__class__.__name__,
                                     sys._getframe().f_code.co_name,
                                     self.Wxh.shape,
                                     self.Whh.shape,
                                     self.bh.shape))
        str_base = 'M@{}, C@{}, F@{}, W_hy shape: {}, b_y shape: {}'
        logging.info(str_base.format(__name__,
                                     self.__class__.__name__,
                                     sys._getframe().f_code.co_name,
                                     self.Why.shape,
                                     self.by.shape))

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
        assert t_batch.ndim == 3
        self.x = x_batch
        self.t = t_batch

    def forward(self):
        if not self.h:  # when self.h is empty
            self.h[-1] = np.zeros((self.batch_size, self.hidden_size))
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

        # calculate gradients
        grads = {}
        grads["Wxh"], grads["Whh"], grads["bh"] = self.d_Wxh, self.d_Whh, self.d_bh
        grads["Why"], grads["by"] = self.d_Why, self.d_by

        return grads  #

    def update_params(self, learning_rate=1e-3):
        self.Wxh -= learning_rate * self.d_Wxh
        self.Whh -= learning_rate * self.d_Whh
        self.bh -= learning_rate * self.d_bh
        self.Why -= learning_rate * self.d_Why
        self.by -= learning_rate * self.d_by

    def init_rnn_state(self, batch_size, num_hiddens):
        return np.zeros(shape=(batch_size, num_hiddens))

    def predict(self, prefix, output_num):
        # prefix 和 outputs 皆为 num_steps 个形状为（batch_size，vocab_size）的矩阵。
        H = self.init_rnn_state(1, self.hidden_size)
        outputs = []
        for X in prefix:
            H = np.tanh(np.dot(X, self.Wxh) + np.dot(H, self.Whh) + self.bh)
            # Y = np.dot(H, self.Why) + self.by
            # outputs.append(Y)
        for t in range(output_num):
            X = prefix[-1]
            H = np.tanh(np.dot(X, self.Wxh) + np.dot(H, self.Whh) + self.bh)
            Y = np.dot(H, self.Why) + self.by
            outputs.append(Y)

        outputs = np.array(outputs)
        outputs = np.argmax(outputs, axis=2)
        return outputs, H

    def sample(self, h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        x = np.zeros((1, self.vocab_size))
        x[0][seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(x, self.Wxh) + np.dot(h, self.Whh) + self.bh)
            y = np.dot(h, self.Why) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((1, self.vocab_size))
            x[0][ix] = 1
            ixes.append(ix)
        return ixes

if __name__ == '__main__':
    if os.name == 'nt':
        file_root_path = r"datasets\\text"
    elif os.name == 'posix':
        file_root_path = r"datasets/text"
    else:
        raise Exception("Invalid os!", os.name)
    # text = TEXT(file_root_path, filename="jaychou_lyrics")
    text = TEXT(file_root_path, filename="shakespeare_input")
    text_data = text.load(convert=False)
    print("text.corpus_chars:", text.corpus_chars)
    vocab_size = len(text.corpus_chars)
    print('vocab_size:', vocab_size)
    print(text_data[:10])
    print(text.idx_to_char(text_data[:10]))

    # hyper-parameters
    seq_length = 25  # number of steps to unroll the RNN for
    batch_size = 1
    hidden_size = 100  # size of hidden layer of neurons
    learning_rate = 1e-2

    # x, t = text.get_one_batch_random(text_data, batch_size=batch_size, steps_num=seq_length)
    g = text.data_iter_consecutive(text_data, batch_size=batch_size, steps_num=seq_length)
    x, t = next(g)
    print("sample before one_hot x.shape:", x.shape)
    x = one_hot(x, class_num=vocab_size, squeeze=False)
    print("sample x.shape:", x.shape)
    print("sample t.shape:", t.shape)

    network = SimpleRNN(seq_length, batch_size, vocab_size, hidden_size, output_size=vocab_size, weight_init_std=0.01)

    optimizer = Adam()
    max_iteration = 20000
    epoch = 200
    loss_list = []
    for i in range(max_iteration):
        # update x, t
        x, t = next(g)
        x_num = np.copy(x)
        x = one_hot(x, class_num=vocab_size, squeeze=False)
        network.update_data(x, t)
        # calculate loss
        loss = network.forward()
        loss_list.append(loss)
        # calculate gradients
        grads = network.backward()
        # network.update_params()
        optimizer.update(network.params, grads)
        if i % epoch == 0:
            print('x:', ''.join([text.idx_to_char_dict[i] for i in x_num.flatten()]))
            print('t:', ''.join([text.idx_to_char_dict[i] for i in t.flatten()]))
            print("{} / {} loss: {}".format(i, max_iteration, loss))

            sample_ix = network.sample(network.h[seq_length-1], x_num[0], 200)
            txt = ''.join(text.idx_to_char_dict[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt,))


    """
    # pre = "分开"
    pre = "This is true"
    pre_list = []
    for cha in pre:
        pre_list.append(text.char_to_idx_dict[cha])
    pre = np.array(pre_list).reshape(-1, 1)
    pre = one_hot(pre, class_num=vocab_size)
    print('pre.shape:', pre.shape)
    pre = pre.reshape(len(pre), 1, -1)
    print('pre.shape:', pre.shape)

    output, _ = network.predict(prefix=pre, output_num=100)
    print('output.shape:', output.shape)

    out_list =[]
    for o in output.flatten():
        out_list.append(text.idx_to_char_dict[o])
    out_str = ''.join(out_list)
    print(out_str)
    """
