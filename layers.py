#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np

import datasets
import functions
import util

class Affine(object):
    def __init__(self, weights, bias):
        self.W = weights
        self.b = bias
        self.d_W = None
        self.d_b = None

        self.x = None
        self.y = None
        
        self.d_x = None

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
        # self.x = x_batch.copy()
        self.x = x_batch
        self.y = np.dot(self.x, self.W) + self.b
        return self.y

    def backward(self, d_y):
        self.d_x = np.dot(d_y, self.W.T)
        self.d_W = np.dot(self.x.T, d_y)
        self.d_b = np.sum(d_y, axis=0)
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
        self.d_x = (self.y - datasets.one_hot(self.t)) / batch_size  # fix here: (y - t) / batch
        return d_y * self.d_x


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

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

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
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

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
