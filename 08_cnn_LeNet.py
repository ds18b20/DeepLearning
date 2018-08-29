#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
import common.layers as layers
from collections import OrderedDict
from common.datasets import MNIST
from common.util import get_one_batch, show_imgs, show_accuracy_loss
import common.optimizer as optimizer


class LeNet(object):
    def __init__(self,
                 input_dim=(1, 28, 28),
                 conv_param_1=None,
                 conv_param_2=None,
                 pool_param_1=None,
                 pool_param_2=None,
                 hidden_size_1=120,
                 hidden_size_2=84,
                 output_size=10,
                 weight_init_std=0.01):

        conv_1_output_h = (input_dim[1] - conv_param_1['filter_size'] + 2*conv_param_1['pad']) / conv_param_1['stride'] + 1
        pool_1_output_h = int(conv_1_output_h/pool_param_1['pool_h'])
        conv_2_output_h = (pool_1_output_h - conv_param_2['filter_size'] + 2*conv_param_2['pad']) / conv_param_2['stride'] + 1
        pool_2_output_size = int(conv_param_2['filter_num'] * (conv_2_output_h/pool_param_2['pool_h']) * (conv_2_output_h/pool_param_2['pool_h']))

        # init parameters
        self.params = {}
        # conv 1
        self.params['W1'] = weight_init_std * np.random.randn(conv_param_1['filter_num'], input_dim[0], conv_param_1['filter_size'], conv_param_1['filter_size'])  # 6,1,5,5
        self.params['b1'] = np.zeros(conv_param_1['filter_num'])  # 6,
        # conv 2
        self.params['W2'] = weight_init_std * np.random.randn(conv_param_2['filter_num'], conv_param_1['filter_num'], conv_param_2['filter_size'], conv_param_2['filter_size'])  # 16,1,5,5
        self.params['b2'] = np.zeros(conv_param_2['filter_num'])  # 16,
        # affine 1
        self.params['W3'] = weight_init_std * np.random.randn(pool_2_output_size, hidden_size_1)  # (N*4*4),100 => 4320,100
        self.params['b3'] = np.zeros(hidden_size_1)  # 100,
        # affine 2
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size_1, hidden_size_2)  # 100,10
        self.params['b4'] = np.zeros(hidden_size_2)  # 10,
        # affine 3 --- out
        self.params['W5'] = weight_init_std * np.random.randn(hidden_size_2, output_size)  # 100,10
        self.params['b5'] = np.zeros(output_size)  # 10,

        # create layers
        self.layers = OrderedDict()
        # conv 1
        self.layers['Conv1'] = layers.Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad'])
        # relu 1
        self.layers['Relu1'] = layers.Relu()
        # pool 1
        self.layers['Pool1'] = layers.Pooling(pool_h=pool_param_1['pool_h'], pool_w=pool_param_1['pool_h'], stride=pool_param_1['pool_stride'])
        # conv 2
        self.layers['Conv2'] = layers.Convolution(self.params['W2'], self.params['b2'], conv_param_1['stride'], conv_param_1['pad'])
        # relu 2
        self.layers['Relu2'] = layers.Relu()
        # pool 2
        self.layers['Pool2'] = layers.Pooling(pool_h=pool_param_1['pool_h'], pool_w=pool_param_1['pool_h'], stride=pool_param_1['pool_stride'])
        # affine 1
        self.layers['Affine1'] = layers.Affine(self.params['W3'], self.params['b3'])
        # relu 3
        self.layers['Relu2'] = layers.Relu()
        # affine 2
        self.layers['Affine2'] = layers.Affine(self.params['W4'], self.params['b4'])
        # relu 4
        self.layers['Relu2'] = layers.Relu()
        # affine 3
        self.layers['Affine3'] = layers.Affine(self.params['W5'], self.params['b5'])
        # loss
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
        grads["W2"], grads["b2"] = self.layers["Conv2"].d_W, self.layers["Conv2"].d_b
        grads["W3"], grads["b3"] = self.layers["Affine1"].d_W, self.layers["Affine1"].d_b
        grads["W4"], grads["b4"] = self.layers["Affine2"].d_W, self.layers["Affine2"].d_b
        grads["W5"], grads["b5"] = self.layers["Affine3"].d_W, self.layers["Affine3"].d_b

        return grads


def show_structure(net, x_batch, y_batch):
    ret = net.loss(x_batch, y_batch)
    for layer in network.layers.values():
        print(layer)
    print(net.lossLayer)
    print('****** Print structure with values: OK ******')
    return ret


if __name__ == '__main__':
    mnist = MNIST('data\\mnist')
    train_x, train_y, test_x, test_y = mnist.load(normalize=True, image_flat=False, label_one_hot=False)
    # # show sample images
    # sample_train_x, sample_train_y = get_one_batch(train_x, train_y, batch_size=5)
    # show_imgs(sample_train_x.reshape(-1, 28, 28), sample_train_y)

    learning_rate = 0.01
    train_acc_list = []
    test_acc_list = []
    network = LeNet(input_dim=(1, 28, 28),
                    conv_param_1={'filter_num': 6, 'filter_size': 5, 'pad': 0, 'stride': 1},
                    conv_param_2={'filter_num': 16, 'filter_size': 5, 'pad': 0, 'stride': 1},
                    pool_param_1={'pool_h': 2, 'pool_stride': 2},
                    pool_param_2={'pool_h': 2, 'pool_stride': 2},
                    hidden_size_1=120,
                    hidden_size_2=84,
                    output_size=10,
                    weight_init_std=0.01)
    # for layer in network.layers.values():
    #     print(layer)
    # print(network.lossLayer)
    # print('****** Print structure without values: OK ******')

    sample_train_x, sample_train_y = get_one_batch(train_x, train_y, batch_size=10)
    show_structure(network, sample_train_x, sample_train_y)

    op = optimizer.Adam(lr=0.001)
    epoch = 100
    for i in range(1000):
        sample_train_x, sample_train_y = get_one_batch(train_x, train_y, batch_size=10)
        grads = network.gradient(sample_train_x, sample_train_y)
        try:
            op.update(network.params, grads)
        except ZeroDivisionError as e:
            print('Handling run-time error:', e)

        if i % epoch == 0:
            # evaluate train accuracy
            acc_train = network.accuracy(sample_train_x, sample_train_y)
            train_acc_list.append(acc_train)
            # evaluate test accuracy
            acc_test = network.accuracy(test_x, test_y)
            test_acc_list.append(acc_test)
            logging.info("train accuracy: {:.3f}, test accuracy: {:.3f}".format(acc_train, acc_test))

    tmp = np.mean(np.array(network.loss_list).reshape(-1, epoch), axis=1)
    show_accuracy_loss(train_acc_list, test_acc_list, tmp)
