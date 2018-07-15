#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import datasets
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import layers


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size):
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

        self.lastLayer = layers.SoftmaxCrossEntropy()

    def predict(self, x_batch, t_batch):
        tmp = x_batch.copy()
        for layer in self.layers.values():
            tmp = layer.forward(tmp)
            print(layer)

        HERE
        tmp = self.lastLayer.forward(tmp, t_batch)
        print(self.lastLayer)
        return tmp


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


if __name__ == '__main__':
    mnist = datasets.MNIST()
    train_x, train_t, test_x, test_t = mnist.load(normalize=True, image_flat=True, label_one_hot=False)
    # show sample images
    sample_train_x, sample_train_t = datasets.get_one_batch(train_x, train_t, batch_size=5)
    # show_sample_imgs(sample_train_x, sample_train_y)

    # # train & evaluate
    # op = Classification2(input_size=28 * 28, hidden_size=50, output_size=10, learning_rate=0.01)
    # for _ in range(1000):
    #     sample_train_x, sample_train_y = datasets.get_one_batch(train_x, train_y, batch_size=5)
    #     op.forward(sample_train_x, sample_train_y)
    #     op.backward(sample_train_x, sample_train_y)
    #     op.update_para()
    #     if _ % 50 == 0:
    #         acc = accuracy(op.fc2(op.activate(op.fc1(test_x))), test_y)
    #         print("accuracy: {}".format(acc))
    net = TwoLayerNet(input_size=28 * 28, hidden_size=50, output_size=10)
    net.predict(sample_train_x, sample_train_t)
