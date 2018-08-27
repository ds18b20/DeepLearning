#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
from common import functions
from common.util import one_hot, im2col, col2im
import sys


class Affine(object):
    def __init__(self, weights, bias):
        self.W = weights
        self.b = bias
        self.d_W = None
        self.d_b = None

        self.x = None
        self.y = None
        self.original_x_shape = None

        self.d_x = None
        logging.info('M@{}, C@{}, F@{}, W shape: {}'.format(__name__, self.__class__.__name__, sys._getframe().f_code.co_name, self.W.shape))
        logging.info('M@{}, C@{}, F@{}, b shape: {}'.format(__name__, self.__class__.__name__, sys._getframe().f_code.co_name, self.b.shape))

    def __str__(self):
        if hasattr(self.x, 'shape'):
            batch_size = self.x.shape[0]
        else:
            batch_size = '?'
        (x_feature_count, y_feature_count) = self.W.shape
        x_shape = (batch_size, x_feature_count)
        y_shape = (batch_size, y_feature_count)
        ret_str = "Affine layer: {} dot {} + {} => {}".format(x_shape, self.W.shape, self.b.shape, y_shape)
        return ret_str

    def forward(self, x_batch):
        # テンソル対応
        self.original_x_shape = x_batch.shape
        x = x_batch.reshape(x_batch.shape[0], -1)
        # self.x = x_batch.copy()
        self.x = x
        self.y = np.dot(self.x, self.W) + self.b

        return self.y

    def backward(self, d_y):
        self.d_x = np.dot(d_y, self.W.T)
        self.d_W = np.dot(self.x.T, d_y)
        self.d_b = np.sum(d_y, axis=0)
        self.d_x = self.d_x.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）

        return self.d_x


class Relu(object):
    def __init__(self):
        self.x = None
        self.y = None

        self.d_x = None

    def __str__(self):
        if hasattr(self.x, 'shape'):
            x_shape = self.x.shape
        else:
            x_shape = ('?', '?')
        if hasattr(self.x, 'shape'):
            y_shape = self.y.shape
        else:
            y_shape = ('?', '?')

        return "Relu layer: {} => {}".format(x_shape, y_shape)

    def forward(self, x_batch):
        # self.x = x_batch.copy()
        self.x = x_batch
        self.y = np.maximum(self.x, 0)
        return self.y

    def backward(self, d_y):
        idx = (self.x <= 0)
        tmp = d_y.copy()  # keep d_y not modified, even modification is OK
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
        if hasattr(self.x, 'shape'):
            x_shape = self.x.shape
        else:
            x_shape = ('?', '?')
        if hasattr(self.x, 'shape'):
            y_shape = self.y.shape
        else:
            y_shape = ('?', '?')
        if hasattr(self.x, 'shape'):
            loss_shape = self.loss.shape
        else:
            loss_shape = ('?', '?')
        return "Softmax Cross Entropy layer: {} => {} => {}".format(x_shape, y_shape, loss_shape)

    def forward(self, x_batch, t_batch):
        # self.x = x_batch.copy()
        self.x = x_batch
        self.t = t_batch
        self.y = functions.softmax(self.x)
        self.loss = functions.cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, d_y=1):
        assert self.t.ndim == 1
        batch_size = self.y.shape[0]
        # 此处错误导致梯度无法正常下降
        self.d_x = (self.y - one_hot(self.t)) / batch_size  # fix here: (y - t) / batch
        return d_y * self.d_x


class Convolution:
    def __init__(self, weights, bias, stride=1, pad=0):
        self.W = weights
        self.b = bias
        self.pad = pad
        self.stride = stride
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.d_W = None
        self.d_b = None

        logging.info('M@{}, C@{}, F@{}, W shape: {}'.format(__name__, self.__class__.__name__, sys._getframe().f_code.co_name, self.W.shape))
        logging.info('M@{}, C@{}, F@{}, b shape: {}'.format(__name__, self.__class__.__name__, sys._getframe().f_code.co_name, self.b.shape))

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, d_y):
        FN, C, FH, FW = self.W.shape
        d_y = d_y.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.d_b = np.sum(d_y, axis=0)
        self.d_W = np.dot(self.col.T, d_y)
        self.d_W = self.d_W.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(d_y, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, d_y):
        d_y = d_y.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((d_y.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = d_y.flatten()
        dmax = dmax.reshape(d_y.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx