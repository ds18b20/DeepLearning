#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from common.datasets import MNIST
from common.util import one_hot, get_one_batch, show_imgs


class Classification2(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.W1 = np.random.normal(loc=0.0, scale=0.1, size=(input_size, hidden_size))  # (28*28, 100)
        self.b1 = np.zeros(hidden_size)  # (100,)
        self.W2 = np.random.normal(loc=0.0, scale=0.1, size=(hidden_size, output_size))  # (100, 10)
        self.b2 = np.zeros(output_size)  # (10,)

        self.lr = learning_rate
    
        self.fc1_out = None
        self.activate_out = None
        self.fc2_out = None
        self.sm_out = None
        self.ce_out = None
    
        self.d_W1 = None
        self.d_b1 = None
        self.d_fc1_out = None
        self.d_activate_out = None
        self.d_W2 = None
        self.d_b2 = None
        self.d_fc2_out = None

    def fc1(self, input_batch):
        """
        calculate fully-connected layer
        :param input_batch: (5, 28*28)
                   self.W1: (28*28, 100)
        :return: X dot W + b = (5, 100)
        """
        return np.dot(input_batch, self.W1) + self.b1  # (5, 100)

    def activate(self, input_batch):
        return np.maximum(input_batch, 0)

    def fc2(self, input_batch):
        """
        calculate fully-connected layer
        :param input_batch: (5, 100)
                    self.W: (100, 10)
        :return: X dot W + b = (5, 10)
        """
        return np.dot(input_batch, self.W2) + self.b2  # (5, 10)
    
    def net(self, input_batch):
        self.fc1_out = self.fc1(input_batch)  # (5, 100)
        self.activate_out = self.activate(self.fc1_out)  # (5, 100)
        self.fc2_out = self.fc2(self.activate_out)  # (5, 10)
    
    def loss(self, label_batch):
        self.sm_out = softmax(self.fc2_out)  # (5, 10)
        self.ce_out = cross_entropy(self.sm_out, label_batch)  # (1,)
    
    def forward(self, input_batch, label_batch):
        """
        forward calculate & store mediate nodes value
        :param input_batch: (5, 28*28)
        :param label_batch: (5,)
        :return: None
        """
        self.net(input_batch)
        self.loss(label_batch)
        
    def bp_softmax_cross_entropy(self, labels):
        """
        softmax_cross_entropy back propagation
        :param labels: (5,)
        :return: None
        """
        assert labels.ndim == 1
        self.d_fc2_out = self.sm_out - one_hot(labels)  # (5, 10)

    def bp_fc2(self):
        """
        fully-connected back propagation
        :param: None
        :return: None
        self.activate_out: (5, 100)
        """
        self.d_W2 = np.dot(self.activate_out.T, self.d_fc2_out)  #(5, 100).T dot (5, 10)->(100, 10)
        self.d_b2 = np.sum(self.d_fc2_out, axis=0, keepdims=False)  # (10,)
        self.d_activate_out = np.dot(self.d_fc2_out, self.W2.T)  #(5, 10) dot (100, 10).T->(5, 100)
    
    def bp_activate(self):
        idx = self.fc1_out < 0
        tmp = self.d_activate_out.copy()  # (5, 100)
        tmp[idx] = 0
        self.d_fc1_out = tmp  # (5, 100)
    
    def bp_fc1(self, input_batch):
        """
        fully-connected back propagation
        :param input_batch: (5, 28*28)
        :return: None
        """
        self.d_W1 = np.dot(input_batch.T, self.d_fc1_out)  #(5, 28*28).T dot (5, 100)->(28*28, 100)
        self.d_b1 = np.sum(self.d_fc1_out, axis=0, keepdims=False)  # (100,)
    
    def backward(self, input_batch, label_batch):
        """
        backward calculate & update weights and bias
        :param input_batch: (5, 28*28)
        :param label_batch: (5,)
        :return: None
        """
        self.bp_softmax_cross_entropy(label_batch)  # (5， 10)
        self.bp_fc2()
        self.bp_activate()
        self.bp_fc1(input_batch)

    def update_para(self):
        """
        update parameters: W & b
        :return: None
        """
        self.W2 -= self.lr * self.d_W2
        self.b2 -= self.lr * self.d_b2
        self.W1 -= self.lr * self.d_W1
        self.b1 -= self.lr * self.d_b1
        

def softmax(array):
    tmp = array.copy()
    tmp -= tmp.max(axis=1, keepdims=True)  # max of array in axis 1
    exp = np.exp(tmp)
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
    mnist = MNIST('data/mnist')
    train_x, train_y, test_x, test_y = mnist.load(image_flat=True, label_one_hot=False)
    # show sample images
    sample_train_x, sample_train_y = get_one_batch(train_x, train_y, batch_size=5)
    show_imgs(sample_train_x.reshape(-1, 28, 28), sample_train_y)
    # train & evaluate
    op = Classification2(input_size=28 * 28, hidden_size=100, output_size=10, learning_rate=0.001)
    for _ in range(1000):
        sample_train_x, sample_train_y = get_one_batch(train_x, train_y, batch_size=5)
        op.forward(sample_train_x, sample_train_y)
        op.backward(sample_train_x, sample_train_y)
        op.update_para()
        if _ % 50 == 0:
            op.net(test_x)
            acc = accuracy(op.fc2_out, test_y)
            print("accuracy: {}".format(acc))
            
    
    # sample_train_x = np.arange(5*3).reshape((5, 3))
    # sample_train_y = np.ones_like(range(5))
    # op = Classification2(input_size=3, hidden_size=10, output_size=2, learning_rate=0.01)
    # op.forward(sample_train_x, sample_train_y)
    # op.fc1(sample_train_x)
    # print("fc1 out: {}".format(op.fc1_out.shape))
    # op.activate(op.fc1_out)
    # print("activate out: {}".format(op.activate_out.shape))
    # op.fc2(op.activate_out)
    # print("fc2 out: {}".format(op.fc2_out.shape))
    # sm = softmax(op.fc2_out)
    # print("sm out: {}".format(sm.shape))
    # print("labels: {}".format(sample_train_y.shape))
    # ce = cross_entropy(sm, sample_train_y)
    # print(ce)