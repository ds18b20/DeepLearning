#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import datasets
import matplotlib.pyplot as plt
import numpy as np


class Classification(object):
    def __init__(self, input_num, output_num, learning_rate):
        self.W = np.random.normal(loc=0.0, scale=0.1, size=(input_num, output_num))  # (28*28, 10)
        self.b = np.zeros(output_num)  # (10,)
        self.lr = learning_rate
    
        self.fc_out = None
        self.sm_out = None
        self.ce_out = None
    
        self.d_W = None
        self.d_b = None
    
        self.d_fc_out = None
    
    def fc(self, input_batch):
        """
        calculate fully-connected layer
        :param input_batch: (5, 28*28)
                    self.W: (28*28, 10)
        :return: X dot W + b = (5, 10)
        """
        return np.dot(input_batch, self.W) + self.b  # (5, 10)

    def forward(self, input_batch, label_batch):
        """
        forward calculate & store mediate nodes value
        :param input_batch: (5, 28*28)
        :param label_batch: (5,)
        :return: None
        """
        self.fc_out = self.fc(input_batch)  # (5, 10)
        self.sm_out = softmax(self.fc_out)  # (5, 10)
        self.ce_out = cross_entropy(self.sm_out, label_batch)  # (1,)
    
    def bp_softmax_cross_entropy(self, labels):
        """
        softmax_cross_entropy back propagation
        :param labels: (5,)
        :return: None
        """
        assert labels.ndim == 1
        self.d_fc_out = self.sm_out - datasets.one_hot(labels)  # (5, 10)

    def bp_fc(self, input_batch):
        """
        fully-connected back propagation
        :param input_batch: (5, 28*28)
        :return: None
        """
        self.d_W = np.dot(input_batch.T, self.d_fc_out)  # (28*28, 10)
        self.d_b = np.sum(self.d_fc_out, axis=0, keepdims=False)  # (10,)
    
    def backward(self, input_batch, labels):
        """
        backward calculate & update weights and bias
        :param input_batch: (5, 28*28)
        :param labels: (5,)
        :return: None
        """
        self.bp_softmax_cross_entropy(labels)  # (5， 10)
        self.bp_fc(input_batch)

    def update_para(self):
        """
        update parameters: W & b
        :return: None
        """
        self.W -= self.lr * self.d_W
        self.b -= self.lr * self.d_b


def softmax(array):
    array -= array.max(axis=1, keepdims=True)  # max of array in axis 1
    exp = np.exp(array)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(y_hat, y):
    delta = 1e-6  # in case of log(0)
    row_count = y_hat.shape[0]
    index_row = range(row_count)
    index_column = y
    picked = y_hat[index_row, index_column] + delta
    return np.sum(-np.log(picked)) / row_count


def accuracy(y_hat: np.array, y: np.array):
    tmp = y_hat.argmax(axis=1) == y  # type: np.ndarray
    return np.mean(tmp)


# def epoch_accuracy(self, data_iter, fc):
#     acc = 0
#     for X, y in data_iter:
#     acc += accuracy(fc(X), y)
#     return acc / len(data_iter)

def show_fashion_imgs(images, titles):
    n = images.shape[0]
    # _, figs = plt.subplots(1, n, figsize=(15, 15))
    _, figs = plt.subplots(1, n)
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)), cmap='gray')
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
        figs[i].axes.set_title(titles[i])
    plt.show()


if __name__ == '__main__':
    mnist = datasets.MNIST()
    train_x, train_y, test_x, test_y = mnist.load(normalize=True, image_flat=True, label_one_hot=False)
    # show sample images
    sample_train_x, sample_train_y = datasets.get_one_batch(train_x, train_y, batch_size=5)
    # show_fashion_imgs(sample_train_x, sample_train_y)
    print(sample_train_x[0])
    # train & evaluate
    op = Classification(input_num=28 * 28, output_num=10, learning_rate=0.01)
    for _ in range(1000):
        sample_train_x, sample_train_y = datasets.get_one_batch(train_x, train_y, batch_size=5)
        op.forward(sample_train_x, sample_train_y)
        op.backward(sample_train_x, sample_train_y)
        op.update_para()
        if _ % 50 == 0:
            acc = accuracy(op.fc(test_x), test_y)
            print("accuracy: {}".format(acc))
