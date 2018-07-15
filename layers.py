#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import datasets
import numpy as np
import functions


class Affine(object):
    def __init__(self, weights, bias):
        self.W = weights
        self.b = bias
        self.d_W = None
        self.d_b = None

        self.x = None
        self.y = None

    def __str__(self):
        if type(self.x) == np.ndarray:
            batch_size = self.x.shape[0]
        else:
            batch_size = '?'
        (x_feature_count, y_feature_count) = self.W.shape
        x_shape = (batch_size, x_feature_count)
        y_shape = (batch_size, y_feature_count)
        ret_str = "Affine layer: {} dot {} + {} => {}".format(x_shape, self.W.shape, self.b.shape, y_shape)
        return ret_str

    def forward(self, x_batch):
        self.x = x_batch.copy()
        self.y = np.dot(self.x, self.W) + self.b
        return self.y

    def backward(self, d_y):
        d_x = np.dot(d_y, self.W.T)
        self.d_W = np.dot(self.x.T, d_y)
        self.d_b = np.sum(d_y, axis=0)
        return d_x


class Relu(object):
    def __init__(self):
        self.x = None
        self.y = None

        self.d_x = None

    def __str__(self):
        if type(self.x) == np.ndarray:
            x_shape = self.x.shape
        else:
            x_shape = ('?', '?')
        if type(self.y) == np.ndarray:
            y_shape = self.y.shape
        else:
            y_shape = ('?', '?')

        return "Relu layer: {} => {}".format(x_shape, y_shape)

    def forward(self, x_batch):
        self.x = x_batch.copy()
        self.y = np.maximum(self.x, 0)
        return self.y

    def backward(self, d_y):
        idx = (self.x <= 0)
        tmp = d_y.copy()
        tmp[idx] = 0
        self.d_x = tmp
        return self.d_x


class SoftmaxCrossEntropy(object):
    def __init__(self):
        self.x = None
        self.t = None

        self.y = None
        self.loss = None

        self.d_x = None

    def __str__(self):
        if type(self.x) == np.ndarray:
            x_shape = self.x.shape
        else:
            x_shape = ('?', '?')
        if type(self.y) == np.ndarray:
            y_shape = self.y.shape
        else:
            y_shape = ('?', '?')
        if type(self.loss) == np.float64:
            loss_shape = self.loss.shape
        else:
            loss_shape = ('?', '?')
        return "Softmax Cross Entropy layer: {} => {} => {}".format(x_shape, y_shape, loss_shape)

    def forward(self, x_batch, t_batch):
        self.x = x_batch.copy()
        self.t = t_batch
        self.y = functions.softmax(self.x)
        self.loss = functions.cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, d_y=1):
        assert self.t.ndim == 1
        batch_size = self.y.shape[0]
        self.d_x = self.y - datasets.one_hot(self.t) / batch_size
        return d_y * self.d_x
