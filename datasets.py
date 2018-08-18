#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import struct
import numpy as np
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
    def __init__(self):
        super(MNIST, self).__init__()
        self.train_image_path = 'data/train-images.idx3-ubyte'
        self.train_label_path = 'data/train-labels.idx1-ubyte'
        self.test_image_path = 'data/t10k-images.idx3-ubyte'
        self.test_label_path = 'data/t10k-labels.idx1-ubyte'
        
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
    mnist = MNIST()
    # train_x, train_y, test_x, test_y = mnist.load(image_flat=False, label_one_hot=False)
    train_x, train_y, test_x, test_y = mnist.load(image_flat=True, label_one_hot=True)
    print('train shape: {}{}, test shape: {}{}'.format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
    train_x_batch, train_y_batch = get_one_batch(train_x, train_y)
    print('batch train shape: {}{}'.format(train_x_batch.shape, train_y_batch.shape))