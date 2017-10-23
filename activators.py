#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np


def sigmoid(weighted_input: float):
    return np.longfloat(1.0 / (1.0 + np.exp(-weighted_input)))
