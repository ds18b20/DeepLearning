#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import datasets
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import layers


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, weight_decay_lambda=0.0):
        # init para
        weight_init_std = 0.01
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # create layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = layers.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = layers.Relu()
        self.layers['Affine2'] = layers.Affine(self.params['W2'], self.params['b2'])

        self.lossLayer = layers.SoftmaxCrossEntropy()

        self.loss_list = []

        self.weight_decay_lambda = weight_decay_lambda

    def predict(self, x_batch):
        tmp = x_batch.copy()
        for layer in self.layers.values():
            tmp = layer.forward(tmp)
            # print(layer)
        return tmp

    def loss(self, x_batch, t_batch):
        y = self.predict(x_batch)
        weight_decay = 0
        L2_penalty = 0.5 * self.weight_decay_lambda * np.sum(self.params['W1']**2)
        L2_penalty += 0.5 * self.weight_decay_lambda * np.sum(self.params['W2']**2)
        ret = self.lossLayer.forward(y, t_batch) + L2_penalty
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
        grads["W1"], grads["b1"] = self.layers["Affine1"].d_W + self.weight_decay_lambda * self.layers["Affine1"].W, self.layers["Affine1"].d_b
        grads["W2"], grads["b2"] = self.layers["Affine2"].d_W + self.weight_decay_lambda * self.layers["Affine2"].W, self.layers["Affine2"].d_b

        return grads


def show_sample_imgs(images, titles):
    n = images.shape[0]
    # _, figs = plt.subplots(1, n, figsize=(15, 15))
    _, figs = plt.subplots(1, n)
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)), cmap='gray')
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
        figs[i].axes.set_title(titles[i])
    plt.show()


def show_accuracy_loss(train_acc, test_acc, loss):
    n = 2
    _, figs = plt.subplots(1, n)
    # fig[0]: train accuracy & test accuracy
    figs[0].plot(train_acc, label='train accuracy')
    figs[0].plot(test_acc, label='test accuracy')
    figs[0].legend()
    # fig[1]: loss
    figs[1].plot(loss, label='loss')
    figs[1].legend()
    plt.show()


if __name__ == '__main__':
    mnist = datasets.MNIST()
    train_x, train_t, test_x, test_t = mnist.load(normalize=True, image_flat=True, label_one_hot=False)
    # reduce training data count to N
    N = 100
    train_x, train_t = train_x[:N], train_t[:N]

    # show sample images
    # sample_train_x, sample_train_t = datasets.get_one_batch(train_x, train_t, batch_size=5)
    # show_sample_imgs(sample_train_x, sample_train_y)
    learning_rate = 0.01
    train_acc_list = []
    test_acc_list = []
    net = TwoLayerNet(input_size=28 * 28, hidden_size=50, output_size=10, weight_decay_lambda=0.2)
    # # train & evaluate
    for i in range(10000):
        sample_train_x, sample_train_t = datasets.get_one_batch(train_x, train_t, batch_size=10)
        gradients = net.gradient(sample_train_x, sample_train_t)
        # update parameters: mini-batch gradient descent
        for key in ("W1", "b1", "W2", "b2"):
            net.params[key] -= learning_rate * gradients[key]
        if i % 50 == 0:
            acc_train = net.accuracy(train_x, train_t)
            train_acc_list.append(acc_train)
            acc_test = net.accuracy(test_x, test_t)
            test_acc_list.append(acc_test)
            print("train accuracy: {:.3f}".format(acc_train), "test accuracy: {:.3f}".format(acc_test))

    tmp = np.mean(np.array(net.loss_list).reshape(-1, 50), axis=1)
    show_accuracy_loss(train_acc_list, test_acc_list, tmp)
