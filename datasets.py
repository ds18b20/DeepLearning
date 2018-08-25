#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import struct
import numpy as np
import os
from util import im2col, show_img, show_imgs, load_pickle, one_hot, get_one_batch, label2name

mnist_fashion_name_list = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]


class Loader(object):
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
    --- *** ---
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
        logging.info('mnist load data: normalize={}, image_flat={}, label_one_hot={}').format(normalize, image_flat, label_one_hot)
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


class CIFAR10(object):
    """
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
    There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with 10000 images.
    The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
    """

    def __init__(self, root):
        """
        initialize CIFAR Loader with file path.
        root: file path
        """
        self.root = root
        self.data_mata = 'batches.meta'
        self.data_batch_file_list = [('data_batch_%d' % idx) for idx in range(1, 6)]
        self.test_batch_file = 'test_batch'
        self.category_list = np.array(
            ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    def _load_data(self, filename):
        """
        load single batch of cifar10
        :param filename: file name to load
        :return:
        """
        with open(filename, 'rb') as f:
            datadict = load_pickle(f)
            logging.info('dict keys: {}'.format(datadict.keys()))

            data = datadict['data']  # --> numpy.ndarray
            data = data.reshape(-1, 3, 32, 32).astype("float32")  # float32(4 bytes)

            labels = np.array(datadict['labels'])  # convert list to numpy.ndarray

            return data, labels

    def load_cifar10(self, normalize=True):
        """
        load all of cifar10 dataset
        :param normalize: [0, 255] to [0, 1]
        :return:
        """
        logging.info('cifar10 load all batches: normalize={}'.format(normalize))

        # load training data
        xs = []
        ys = []
        x = None
        y = None
        file_path_list = [os.path.join(self.root, fn) for fn in self.data_batch_file_list]
        for file_path in file_path_list:
            x, y = self._load_data(file_path)
            xs.append(x)
            ys.append(y)
        train_image = np.concatenate(xs)
        train_label = np.concatenate(ys)
        del x, y

        # load test data
        test_image, test_label = self._load_data(os.path.join(self.root, self.test_batch_file))

        # data processing
        if normalize:
            train_image = train_image / 255.0
            test_image = test_image / 255.0

        logging.info('train_image shape: {}'.format(train_image.shape))
        logging.info('train_label shape: {}'.format(train_image.shape))
        logging.info('test_image shape: {}'.format(test_image.shape))
        logging.info('test_label shape: {}'.format(test_label.shape))
        return train_image, train_label, test_image, test_label

    def load_cifar10_batch_one(self, normalize=True):
        """
        load cifar train_batch_1 & test_batch
        :param normalize: [0, 255] to [0, 1]
        :return:
        """
        logging.info('cifar10 load batch_one: normalize={}'.format(normalize))

        # load training data
        f = os.path.join(self.root, 'data_batch_%d' % (1,))
        train_image, train_label = self._load_data(f)

        # load test data
        test_image, test_label = self._load_data(os.path.join(self.root, 'test_batch'))

        # data processing
        if normalize:
            train_image = train_image / 255.0
            test_image = test_image / 255.0
        return train_image, train_label, test_image, test_label


class CIFAR100(object):
    """
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
    There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with 10000 images.
    The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
    """

    def __init__(self, root):
        """
        initialize CIFAR Loader with file path.
        root: file path
        """
        self.root = root
        self.data_mata = 'meta'
        self.train_batch_file = 'train'
        self.test_batch_file = 'test'
        self.fine_label_names = None
        self.coarse_label_names = None
        self.load_meta()

    def _extract_meta(self, filename):
        """
        extract meta data
        :param filename:
        :return:
        """
        with open(filename, 'rb') as f:
            meta_dict = load_pickle(f)
            logging.info('dict keys: {}'.format(meta_dict.keys()))
            self.fine_label_names = np.array(meta_dict['fine_label_names'])  # convert list to numpy.ndarray
            self.coarse_label_names = np.array(meta_dict['coarse_label_names'])  # convert list to numpy.ndarray

    def load_meta(self,):
        self._extract_meta(os.path.join(self.root, self.data_mata))

    def _load_data(self, filename):
        """
        load single batch of cifar10
        :param filename: file name to load
        :return:
        """
        with open(filename, 'rb') as f:
            datadict = load_pickle(f)
            logging.info('dict keys: {}'.format(datadict.keys()))

            data = datadict['data']  # --> numpy.ndarray
            data = data.reshape(-1, 3, 32, 32).astype("float32")  # float32(4 bytes)

            fine_labels = np.array(datadict['fine_labels'])  # convert list to numpy.ndarray
            coarse_labels = np.array(datadict['coarse_labels'])  # convert list to numpy.ndarray

            return data, fine_labels, coarse_labels

    def load_cifar100(self, normalize=True):
        """
        load all of cifar100 dataset
        :param normalize: [0, 255] to [0, 1]
        :return:
        """
        logging.info('cifar100 load all batches: normalize={}'.format(normalize))

        # load training data
        train_image, train_label, _ = self._load_data(os.path.join(self.root, self.train_batch_file))

        # load test data
        test_image, test_label, _ = self._load_data(os.path.join(self.root, self.test_batch_file))

        # data processing
        if normalize:
            train_image = train_image / 255.0
            test_image = test_image / 255.0

        logging.info('train_image shape: {}'.format(train_image.shape))
        logging.info('train_label shape: {}'.format(train_image.shape))
        logging.info('test_image shape: {}'.format(test_image.shape))
        logging.info('test_label shape: {}'.format(test_label.shape))
        return train_image, train_label, test_image, test_label


if __name__ == '__main__':
    # test MNIST
    """
    mnist = MNIST('datasets/mnist')
    # train_x, train_y, test_x, test_y = mnist.load(image_flat=False, label_one_hot=False)
    train_x, train_y, test_x, test_y = mnist.load(image_flat=True, label_one_hot=True)
    logging.info('train shape: {}{}, test shape: {}{}'.format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
    train_x_batch, train_y_batch = get_one_batch(train_x, train_y)
    logging.info('batch train shape: {}{}'.format(train_x_batch.shape, train_y_batch.shape))
    """

    # test CIFAR10
    """
    cifar10 = CIFAR10('datasets/cifar10')
    train_x, train_y, test_x, test_y = cifar10.load_cifar10_batch_one(normalize=True)

    n = 10
    images = train_x[0:n].transpose(0, 2, 3, 1)
    labels = train_y[0:n]
    sm = show_img()  # coroutine by generator
    sm.__next__()
    for i in range(n):
        sm.send((images[i], labels[i]))
    show_imgs(images, cifar10.label2text(labels))
    """

    # test CIFAR100
    cifar100 = CIFAR100('datasets/cifar100')
    train_x, train_y, test_x, test_y = cifar100.load_cifar100()

    n = 10
    images = train_x[0:n].transpose(0, 2, 3, 1)
    labels = train_y[0:n]
    sm = show_img()  # coroutine by generator
    sm.__next__()
    for i in range(n):
        sm.send((images[i], labels[i]))
    show_imgs(images, label2name(index_array=labels, label_array=cifar100.fine_label_names))
