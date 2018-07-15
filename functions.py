#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np


def softmax(array):
    tmp = array.copy()
    tmp -= tmp.max(axis=1, keepdims=True)  # max of array in axis 1
    exp = np.exp(tmp)  # exp(matrix)
    return exp / np.sum(exp, axis=1, keepdims=True)  # exp / sum of each row


def cross_entropy(y_hat, y):
    delta = 1e-6  # in case of log(0)
    row_count = y_hat.shape[0]
    index_row = range(row_count)
    index_column = y
    picked = y_hat[index_row, index_column] + delta  # select element by y
    return np.sum(-np.log(picked)) / row_count  # sum(-t * ln(y)) / row_count


def accuracy(y_hat: np.array, y: np.array):
    tmp = y_hat.argmax(axis=1) == y  # type: np.ndarray
    return np.mean(tmp)
