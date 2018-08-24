#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import struct
import numpy as np
import os
import platform
from six.moves import cPickle as pickle
from util import im2col, show_img

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images.
The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 
"""
class CIFAR10(object):
    def __init__(self, root):
        """
        initialize CIFAR Loader with file path.
        root: file path
        """
        self.root = root
        self.category_list = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    def load_pickle(self, f):
        version = platform.python_version_tuple()
        if version[0] == '2':
            return  pickle.load(f)
        elif version[0] == '3':
            return  pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))

    def load_CIFAR_batch(self, filename):
      """ load single batch of cifar """
      with open(filename, 'rb') as f:
        datadict = self.load_pickle(f)
        X = datadict['data']  # --> numpy.ndarray
        Y = datadict['labels']  # --> list
        # X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")  # float == float64(8 bytes)
        # X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float32")  # float32(4 bytes)
        X = X.reshape(10000, 3, 32, 32).astype("float32")  # float32(4 bytes)
        Y = np.array(Y)
        return X, Y

    def load_CIFAR10(self, normalize=True):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(self.root, 'data_batch_%d' % (b, ))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)    
            Xtr = np.concatenate(xs)
            Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = self.load_CIFAR_batch(os.path.join(self.root, 'test_batch'))
        if normalize:
            Xtr = Xtr / 255.0
            Xte = Xte / 255.0
        return Xtr, Ytr, Xte, Yte

      
    def load_CIFAR10_batch_one(self, normalize=True):
        """ load cifar train_batch_1 & test_batch"""
        xs = []
        ys = []

        f = os.path.join(self.root, 'data_batch_%d' % (1, ))
        X, Y = self.load_CIFAR_batch(f)

        Xtr = X
        Ytr = Y
        del X, Y
        Xte, Yte = self.load_CIFAR_batch(os.path.join(self.root, 'test_batch'))
        if normalize:
            Xtr = Xtr / 255.0
            Xte = Xte / 255.0
        return Xtr, Ytr, Xte, Yte
      
    def index2name(self, index):
        try:
            return self.category_list[index]
        except:
            pass
"""
TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
"""
"""
TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
"""

# basic class for loading data
class Loader(object):
    def __init__(self):
        """
        initialize Loader
        path: file path
        count: sample count
        data_type: data type
        dims: data dimensions
        """
        self.data_type = None
        self.dims = None
        self.shape = None

    def load_raw(self, path):
        """ A function that can read MNIST's idx file format into numpy arrays.
        The MNIST data files can be downloaded from here:
        http://yann.lecun.com/exdb/mnist/

        This relies on the fact that the MNIST dataset consistently uses
        unsigned char types with their data segments.
        
        Reference:
        https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
        """
        with open(path, 'rb') as f:
            zero, self.data_type, self.dims = struct.unpack('>HBB', f.read(4))
            self.shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(self.dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(self.shape)

            
class MNIST(Loader):
    def __init__(self, root):
        super(MNIST, self).__init__()
        self.root = root
        self.train_image_path = os.path.join(self.root, 'train-images.idx3-ubyte')
        self.train_label_path = os.path.join(self.root, 'train-labels.idx1-ubyte')
        self.test_image_path = os.path.join(self.root, 't10k-images.idx3-ubyte')
        self.test_label_path = os.path.join(self.root, 't10k-labels.idx1-ubyte')
        
    def load(self, normalize=True, image_flat=False, label_one_hot=False):
        train_image = self.load_raw(self.train_image_path)
        train_label = self.load_raw(self.train_label_path)
        test_image = self.load_raw(self.test_image_path)
        test_label = self.load_raw(self.test_label_path)
        
        if normalize:
            train_image = train_image / 255.0
            test_image = test_image / 255.0

        if image_flat:
            train_image = train_image.reshape(train_image.shape[0], -1)
            test_image = test_image.reshape(test_image.shape[0], -1)

        if label_one_hot:
            train_label = one_hot(train_label)
            test_label = one_hot(test_label)
            
        return train_image, train_label, test_image, test_label


def one_hot(vec, length=10):
    """
    (vec_count, )-->(vec_count, length)
    """
    row_count = len(vec)
    column_count = length

    label_one_hot = np.zeros((row_count, column_count))
    label_one_hot[range(row_count), vec] = 1
    
    return label_one_hot
    

def labels_to_text(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]
    
    
def get_one_batch(image_set, label_set, batch_size=100):
    """
    get batch data
    """
    set_size = len(image_set)
    index = np.random.choice(set_size, batch_size)
    return image_set[index], label_set[index]


if __name__ == '__main__':
    """
    mnist = MNIST('datasets/mnist')
    # train_x, train_y, test_x, test_y = mnist.load(image_flat=False, label_one_hot=False)
    train_x, train_y, test_x, test_y = mnist.load(image_flat=True, label_one_hot=True)
    logging.info('train shape: {}{}, test shape: {}{}'.format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
    train_x_batch, train_y_batch = get_one_batch(train_x, train_y)
    logging.info('batch train shape: {}{}'.format(train_x_batch.shape, train_y_batch.shape))
    """
    cifar10 = CIFAR10('datasets/cifar10')
    train_x, train_y, test_x, test_y = cifar10.load_CIFAR10_batch_one(normalize=True)
    logging.info('train_x shape: {}'.format(train_x.shape))
    logging.info('train_y shape: {}'.format(train_y.shape))
    logging.info('test_x shape: {}'.format(test_x.shape))
    logging.info('test_y shape:: {}'.format(test_y.shape))
    imgs = train_x[0:10].transpose(0,2,3,1)
    sm = show_img()
    sm.__next__()
    for i in imgs:
        print(i.shape)
        print(i.dtype)
        print("***")
        sm.send(i)
        