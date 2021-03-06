#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common import layers
from collections import OrderedDict
from common.util import numerical_gradient, get_one_batch
from common.optimizer import SGD


class MultiLayerRegression(object):
    def __init__(self, input_size: int, hidden_size_list: list, output_size: int, weight_init_std, activation, batch_size: int):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden_layer_num = len(hidden_size_list)

        self.params = {}
        self.init_weight(weight_init_std)

        activation_layer = {'sigmoid': layers.Sigmoid, 'relu': layers.Relu}
        self.layers = OrderedDict()
        for idx in range(1, len(self.hidden_size_list) + 1):
            self.layers['Affine' + str(idx)] = layers.Affine(weights=self.params['W'+str(idx)],
                                                             bias=self.params['b'+str(idx)])
            self.layers['Activation' + str(idx)] = activation_layer[activation]()  # must use () to create instance !!!
        idx = len(self.hidden_size_list) + 1
        self.layers['Affine' + str(idx)] = layers.Affine(weights=self.params['W' + str(idx)],
                                                         bias=self.params['b' + str(idx)])
        self.last_layer = layers.MSE()

    def init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if isinstance(weight_init_std, str):
                if weight_init_std.lower() in ['sigmoid', 'xavier']:
                    scale = np.sqrt(1.0 / all_size_list[idx - 1])
                elif weight_init_std.lower() in ['relu', 'he']:
                    scale = np.sqrt(2.0 / all_size_list[idx - 1])
                else:
                    print('str ')

            self.params['W'+str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b'+str(idx)] = np.random.randn(all_size_list[idx])

    def predict(self, x_batch):
        logging.info('Predict Start...')
        logging.info('Input> x_batch shape: {}'.format(x_batch.shape))
        tmp = x_batch.copy()  # .copy() is not necessary!
        for layer in self.layers.values():
            tmp = layer.forward(tmp)
            logging.info('Forward Layer> {}'.format(layer))
        logging.info('Output> y_batch shape: {}'.format(next(reversed(self.layers.values())).y.shape))
        logging.info('Predict End.')
        return tmp

    def loss(self, x_batch, t_batch):
        logging.info('Loss Cal Start...')
        y_batch = self.predict(x_batch)
        loss = self.last_layer.forward(y_batch, t_batch)
        logging.info('Loss Cal  End.')
        return loss

    def gradient(self, x_batch, t_batch):
        # forward
        logging.info('Forward Start...')
        self.loss(x_batch, t_batch)
        logging.info('Forward End.')
        # backward
        logging.info('Backward Start...')
        dout = 1
        logging.info('Loss Layer> {}'.format(self.last_layer))
        dout = self.last_layer.backward(d_y=dout)
        layers = list(self.layers.values())  # self.layers is a dict not a list!!!
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
            logging.info('Backward Layer> {}'.format(layer))
        logging.info('Backward End.')
        grad = {}
        for idx in range(1, len(self.hidden_size_list) + 2):
            grad['W'+str(idx)] = self.layers['Affine' + str(idx)].d_W
            grad['b'+str(idx)] = self.layers['Affine' + str(idx)].d_b
        return grad
    
    def numerical_gradient(self, x, t):
        """勾配を求める（数値微分）
        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            # grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            # grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.layers['Affine' + str(idx)].W)
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.layers['Affine' + str(idx)].b)

        return grads

    def generate_simple_dataset(self, num_examples):
        true_w = np.array([3.0, 7.0, 5.0], dtype=float)
        true_b = np.array([8.0, ], dtype=float)

        features = np.random.randn(num_examples, self.input_size)
        labels = np.dot(features, true_w) + true_b
        epsilon = np.random.normal(0.0, scale=0.01)
        labels += epsilon

        return features, labels.reshape(-1, 1)

    def update_batch(self, features_set, labels_set):
        # assert len(features_set) == len(labels_set)
        num_examples = len(features_set)
        index = np.random.choice(num_examples, self.batch_size)
        batch_features = features_set[index]
        batch_labels = labels_set[index]

        return batch_features, batch_labels

    def numerical_gradient_test(self, x, t):
        """勾配を求める（数値微分）
        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_W = lambda W: self.loss(x, t)

        print('W shape:', self.params['W1'].shape)
        print('b shape:', self.params['b1'].shape)
        grads_W1 = foo(loss_W, self.params['W1'])

        return grads_W1


def foo(f, x):
    print('ori_x:\n', x)
    return f(x)


if __name__ == '__main__':
    # reg = OneLayerRegression(input_vec_size=2, batch_size=10)
    # sample_features, sample_labels = reg.generate_simple_dataset(num_examples=1000)
    #
    # loss = []
    # for i in range(1000):
    #     reg.update_batch(sample_features, sample_labels)
    #     reg.forward_pass()
    #     reg.backward_pass()
    #
    #     loss.append(reg.squared_loss())
    #
    # print('W: {}'.format(reg.get_w()))
    # print('b: {}'.format(reg.get_b()))
    #
    # plt.plot(loss)
    # plt.show()

    reg = MultiLayerRegression(input_size=3, hidden_size_list=[5, 7, 9], output_size=1, weight_init_std=0.01, activation='sigmoid', batch_size=10)
    """
    # # reg.init_weight(0.01)  # second use -> error!!!
    """
    # print(reg.params['W1'])
    # print(reg.layers['Affine1'].W)

    x, t = reg.generate_simple_dataset(num_examples=1000)
    print('Source Data: x.shape', x.shape)
    print('Source Data: t.shape', t.shape)
    sample_x, sample_t = reg.update_batch(x, t)
    print('Sample Data: sample_x.shape', sample_x.shape)
    print('Sample Data: sample_t.shape', sample_t.shape)
    # print(sample_x[0], sample_t[0])


