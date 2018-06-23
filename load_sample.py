#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import struct
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np

# 数据加载器基类
class Loader(object):
    def __init__(self, path, count):
        """
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        """
        self.path = path
        self.count = count

    def get_file_content(self):
        """
        读取文件内容
        """
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

    def to_int(self, byte):
        """
        将unsigned byte字符转换为整数
        """
        return struct.unpack('B', byte)[0]


# 图像数据加载器
class ImageLoader(Loader):
    def get_picture(self, content, index):
        """
        内部函数，从文件中获取图像
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
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                # picture[i].append(self.to_int(content[start + i * 28 + j]))
                picture[i].append(content[start + i * 28 + j])
        return picture

    def get_one_sample(self, picture):
        """
        内部函数，将图像转化为样本的输入向量
        数据降维，[[a,b],...,[c,d]] -> [a,b,...,c,d]
        """
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        """
        加载数据文件，获得全部样本的输入向量
        """
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set


# 标签数据加载器
class LabelLoader(Loader):
    def load(self):
        """
        加载数据文件，获得全部样本的标签向量
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
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        """
        内部函数，将一个值转换为10维标签向量
        数据降维，[[a,b],...,[c,d]] -> [a,b,...,c,d]
        """
        label_vec = []
        # label_value = self.to_int(label)
        label_value = label
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def get_training_data_set():
    """
    获得训练数据集
    """
    image_loader = ImageLoader('train-images.idx3-ubyte', 10000)
    label_loader = LabelLoader('train-labels.idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    """
    获得测试数据集
    """
    image_loader = ImageLoader('t10k-images.idx3-ubyte', 5000)
    label_loader = LabelLoader('t10k-labels.idx1-ubyte', 5000)
    return image_loader.load(), label_loader.load()

if __name__ == '__main__':
    start_time = time.time()
    images, labels = get_training_data_set()
    img = np.array(images[0]).reshape(28, 28)
    end_time = time.time()
    print('Run time: {}'.format(end_time-start_time))
    print('image shape: {}, label shape: {}'.format((len(images),len(labels[0])), len(labels)))
    print(img[0])
    plt.imshow(img, cmap='gray')
    plt.show()
    