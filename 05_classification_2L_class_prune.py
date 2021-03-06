#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from common import layers
from common.datasets import MNIST
from common.util import one_hot, get_one_batch
from common.visualize import show_imgs, show_accuracy_loss


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size):
        # init para
        weight_init_std = 0.01
        self.params = OrderedDict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # self.params['W1'] = weight_init_std * np.random.normal(loc=0.0, scale=1.0, size=(input_size, hidden_size))
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # self.params['W2'] = weight_init_std * np.random.normal(loc=0.0, scale=1.0, size=(hidden_size, output_size))
        self.params['b2'] = np.zeros(output_size)

        # create layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = layers.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = layers.Relu()
        self.layers['Affine2'] = layers.Affine(self.params['W2'], self.params['b2'])

        self.lossLayer = layers.SoftmaxCrossEntropy(class_num=10)

        self.loss_list = []

    def predict(self, x_batch):
        tmp = x_batch.copy()
        for layer in self.layers.values():
            tmp = layer.forward(tmp)
            # print(layer)
        return tmp

    def loss(self, x_batch, t_batch):
        y = self.predict(x_batch)
        ret = self.lossLayer.forward(y, t_batch)
        # print(self.lossLayer)
        return ret

    def accuracy(self, x_batch, t_batch):
        y = self.predict(x_batch)
        y = np.argmax(y, axis=1)
        # if t_batch.ndim != 1:
        #     tmp = t_batch.copy()
        #     tmp = np.argmax(tmp, axis=1)
        accuracy = np.sum(y == t_batch) / float(x_batch.shape[0])
        return accuracy

    def gradient(self, x_batch, t_batch):
        # forward
        loss = self.loss(x_batch, t_batch)
        self.loss_list.append(loss)
        # backward
        d_y = 1
        d_y = self.lossLayer.backward(d_y)
        layers_list = list(self.layers.values())
        layers_list.reverse()
        for layer in layers_list:
            d_y = layer.backward(d_y)
        # calculate gradients
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Affine1"].d_W, self.layers["Affine1"].d_b
        grads["W2"], grads["b2"] = self.layers["Affine2"].d_W, self.layers["Affine2"].d_b

        return grads


def show(inputs):
    bins_range = 100
    for idx, key in enumerate(inputs.keys()):
        ax = plt.subplot(1, len(inputs), idx + 1)
        ax.set_title(key)

        # if isinstance(act, layers.Sigmoid):
        #     ran = (0, 1)
        # else:
        #     ran = None
        ran = (-1, 1)
        data = inputs[key].flatten()
        ax.hist(data, bins=bins_range, range=ran)
        # data_non_zero = data[np.where(data != 0)]
        # ax = plt.subplot(1, len(inputs), idx + 1)
        # ax.set_title(key)
        # ax.hist(data_non_zero, bins=bins_range, range=ran)
        # if idx != 0:
        #     plt.yticks([], [])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mnist = MNIST('datasets\\mnist')
    train_x, train_y, test_x, test_y = mnist.load(normalize=True, image_flat=True, label_one_hot=False)
    # show sample images
    sample_train_x, sample_train_y = get_one_batch(train_x, train_y, batch_size=5)
    # show_imgs(sample_train_x.reshape(-1, 28, 28), sample_train_y)
    learning_rate = 0.05
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    network = TwoLayerNet(input_size=28 * 28, hidden_size=50, output_size=10)
    # print(network.params["W1"][0])
    # show(network.params)

    epoch = 200
    # train & evaluate
    for i in range(2000):
        sample_train_x, sample_train_y = get_one_batch(train_x, train_y, batch_size=5)
        gradients = network.gradient(sample_train_x, sample_train_y)
        # update parameters: mini-batch gradient descent
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * gradients[key]
        if i % epoch == 0:
            # calculate accuracy
            train_acc = network.accuracy(sample_train_x, sample_train_y)
            train_acc_list.append(train_acc)
            test_acc = network.accuracy(test_x, test_y)
            test_acc_list.append(test_acc)
            print("train accuracy: {:.3f}".format(train_acc), "test accuracy: {:.3f}".format(test_acc))
            # calculate loss
            train_loss = network.loss(train_x, train_y)
            train_loss_list.append(train_loss)
            test_loss = network.loss(test_x, test_y)
            test_loss_list.append(test_loss)
            print("train loss: {:.3f}".format(train_loss), "test loss: {:.3f}".format(test_loss))

    # show_accuracy_loss(train_acc_list, test_acc_list, train_loss_list, test_loss_list)
    print(network.params["b2"])
    print("*************************************************************")
    test_acc = network.accuracy(test_x, test_y)
    print("test accuracy: {:.3f}".format(test_acc))
    print("*************************************************************")
    show(network.params)

    # prune
    threshold = 0.05
    network.params["W1"][np.where(np.abs(network.params["W1"]) < threshold)] = 0
    network.params["W2"][np.where(np.abs(network.params["W2"]) < threshold)] = 0

    print("*************************************************************")
    test_acc = network.accuracy(test_x, test_y)
    print("test accuracy: {:.3f}".format(test_acc))
    print("*************************************************************")
    show(network.params)


#   2nd training
    for i in range(2000):
        sample_train_x, sample_train_y = get_one_batch(train_x, train_y, batch_size=5)
        gradients = network.gradient(sample_train_x, sample_train_y)
        # update parameters: mini-batch gradient descent
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * gradients[key]
        if i % epoch == 0:
            # calculate accuracy
            train_acc = network.accuracy(sample_train_x, sample_train_y)
            train_acc_list.append(train_acc)
            test_acc = network.accuracy(test_x, test_y)
            test_acc_list.append(test_acc)
            print("train accuracy: {:.3f}".format(train_acc), "test accuracy: {:.3f}".format(test_acc))
            # calculate loss
            train_loss = network.loss(train_x, train_y)
            train_loss_list.append(train_loss)
            test_loss = network.loss(test_x, test_y)
            test_loss_list.append(test_loss)
            print("train loss: {:.3f}".format(train_loss), "test loss: {:.3f}".format(test_loss))

    # show_accuracy_loss(train_acc_list, test_acc_list, train_loss_list, test_loss_list)

    print("*************************************************************")
    test_acc = network.accuracy(test_x, test_y)
    print("test accuracy: {:.3f}".format(test_acc))
    print("*************************************************************")
    show(network.params)

    # prune
    threshold = 0.05
    network.params["W1"][np.where(np.abs(network.params["W1"]) < threshold)] = 0
    network.params["W2"][np.where(np.abs(network.params["W2"]) < threshold)] = 0

    print("*************************************************************")
    test_acc = network.accuracy(test_x, test_y)
    print("test accuracy: {:.3f}".format(test_acc))
    print("*************************************************************")
    show(network.params)
