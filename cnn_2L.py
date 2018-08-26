#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
import numpy as np
import layers
from collections import OrderedDict
from datasets import datasets
from util import get_one_batch, show_imgs, show_accuracy_loss


class SimpleConvNet(object):
    def __init__(self,
                 input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 pool_param={'pool_size': 2, 'pool_stride': 2},
                 hidden_size=100,
                 output_size=10,
                 weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        pool_size = pool_param['pool_size']
        pool_stride = pool_param['pool_stride']
        channel, input_size, _ = input_dim
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/pool_size) * (conv_output_size/pool_size))

        # init para
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, channel, filter_size, filter_size)  # 30,1,5,5
        self.params['b1'] = np.zeros(filter_num)  # 30,
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)  # (30*12*12),100 => 4320,100
        self.params['b2'] = np.zeros(hidden_size)  # 100,
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)  # 100,10
        self.params['b3'] = np.zeros(output_size)  # 10,

        # create layers
        self.layers = OrderedDict()
        self.layers['Conv1'] = layers.Convolution(self.params['W1'], self.params['b1'],
                                                  filter_stride, filter_pad)

        self.layers['Relu1'] = layers.Relu()
        self.layers['Pool1'] = layers.Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = layers.Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = layers.Relu()
        self.layers['Affine2'] = layers.Affine(self.params['W3'], self.params['b3'])

        self.lossLayer = layers.SoftmaxCrossEntropy()

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
        grads["W1"], grads["b1"] = self.layers["Conv1"].d_W, self.layers["Conv1"].d_b
        grads["W2"], grads["b2"] = self.layers["Affine1"].d_W, self.layers["Affine1"].d_b
        grads["W3"], grads["b3"] = self.layers["Affine2"].d_W, self.layers["Affine2"].d_b

        return grads


if __name__ == '__main__':
    mnist = datasets.MNIST(r'datasets/mnist')
    train_x, train_y, test_x, test_y = mnist.load(normalize=True, image_flat=False, label_one_hot=False)
    # # show sample images
    # sample_train_x, sample_train_y = datasets.get_one_batch(train_x, train_y, batch_size=5)
    # show_imgs(sample_train_x, sample_train_y)

    learning_rate = 0.01
    train_acc_list = []
    test_acc_list = []
    net = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        pool_param={'pool_size': 2, 'pool_stride': 2},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

    for i in range(100):
        sample_train_x, sample_train_y = get_one_batch(train_x, train_y, batch_size=10)
        logging.info('input sample_train_x shape: {}'.format(sample_train_x.shape))
        logging.info('input sample_train_y shape: {}'.format(sample_train_y.shape))

        gradients = net.gradient(sample_train_x, sample_train_y)
        # update parameters: mini-batch gradient descent
        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            net.params[key] -= learning_rate * gradients[key]
        if i % 10 == 0:
            # acc_train = net.accuracy(train_x, train_y)
            # train_acc_list.append(acc_train)
            acc_test = net.accuracy(test_x, test_y)
            test_acc_list.append(acc_test)
            logging.info('test accuracy: {}'.format(acc_test))
            # print("train accuracy: {:.3f}".format(acc_train), "test accuracy: {:.3f}".format(acc_test))

    # tmp = np.mean(np.array(net.loss_list).reshape(-1, 50), axis=1)
    # show_accuracy_loss(train_acc_list, test_acc_list, tmp)
