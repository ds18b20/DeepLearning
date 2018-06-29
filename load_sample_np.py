#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import struct
import numpy as np


# 数据加载器基类
class Loader(object):
    def __init__(self, path):
        """
        initialize Loader
        path: file path
        count: sample count
        data_type: data type
        dims: data dimensions
        """
        self.path = path
        self.data_type = None
        self.dims = None
        self.shape = None

    def load_raw(self):
        """ A function that can read MNIST's idx file format into numpy arrays.
        The MNIST data files can be downloaded from here:
        http://yann.lecun.com/exdb/mnist/

        This relies on the fact that the MNIST dataset consistently uses
        unsigned char types with their data segments.
        
        Reference:
        https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
        """
        with open(self.path, 'rb') as f:
            zero, self.data_type, self.dims = struct.unpack('>HBB', f.read(4))
            self.shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(self.dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(self.shape)


# 图像数据加载器
class ImageLoader(Loader):
    """
    Get all images
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
    def __init__(self, path):
        super(ImageLoader, self).__init__(path)

    def load(self):
        """
        convert data to vector
        """
        ret = self.load_raw()
        assert self.dims == 3
        return ret.reshape(self.shape[0], -1)

    def get_one_image(self, index):
        assert self.dims == 3
        return self.load_raw()[index]


# 标签数据加载器
class LabelLoader(Loader):
    """
    Get all labels
    Convert labels to one-hot vectors
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
    def __init__(self, path):
        super(LabelLoader, self).__init__(path)

    def norm(self, label_count=10):
        """
        (sample_count, )-->(sample_count, label_count)
        """
        row_count = self.shape[0]
        column_count = label_count

        label_one_hot = np.zeros((row_count, column_count))
        label_one_hot[range(row_count), self.load()] = 1
        
        return label_one_hot

    def load(self):
        """
        convert data to vector
        """
        ret = self.load_raw()
        assert self.dims == 1
        return ret


def get_training_data_set(sample_count=100):
    """
    get training data set
    """
    image_population = ImageLoader('train-images.idx3-ubyte').load()
    label_population = LabelLoader('train-labels.idx1-ubyte').load()
    assert len(image_population) == len(label_population)
    population_count = len(image_population)
    index = np.random.choice(population_count, sample_count)
    return image_population[index], label_population[index]


def get_test_data_set(sample_count=50):
    """
    get test data set
    """
    image_population = ImageLoader('t10k-images.idx3-ubyte').load()
    label_population = LabelLoader('t10k-labels.idx1-ubyte').load()
    assert len(image_population) == len(label_population)
    population_count = len(image_population)
    index = np.random.choice(population_count, sample_count)
    return image_population[index], label_population[index]


if __name__ == '__main__':
    start_time = time.time()
    images, labels = get_training_data_set(10000)
    img = np.array(images[0]).reshape(28, 28)
    end_time = time.time()
    print('Run time: {}'.format(end_time-start_time))
    print('image shape: {}, label shape: {}'.format(images.shape, labels.shape))
    print(img[14])
    plt.imshow(img, cmap='gray')
    plt.show()
    
