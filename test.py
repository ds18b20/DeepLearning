#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" A function that can read MNIST's idx file format into numpy arrays.

    The MNIST data files can be downloaded from here:
    
    http://yann.lecun.com/exdb/mnist/

    This relies on the fact that the MNIST dataset consistently uses
    unsigned char types with their data segments.
    Reference:
    https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
"""

import struct
import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        print(zero, data_type, dims)
        print('type of dims: ', type(dims))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        # print(shape)
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

if __name__ == '__main__':
    # ret = read_idx('train-images.idx3-ubyte')
    ret = read_idx('train-labels.idx1-ubyte')
    print(ret.shape)
