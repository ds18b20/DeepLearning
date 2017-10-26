#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np


def sigmoid(weighted_input: float):
    return np.longfloat(1.0 / (1.0 + np.exp(-weighted_input)))


# Sigmoid激活函数类
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)
