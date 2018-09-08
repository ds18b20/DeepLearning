#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from common import layers


class ActivationNet(object):
    def __init__(self, input_size, hidden_size, output_size, layer_num, w_std, activator):
        # init para
        weight_init_std = w_std
        self.params = {}
        for idx in range(layer_num):
            if idx == 0:
                self.params['W'+str(idx)] = weight_init_std * np.random.randn(input_size, hidden_size)
            else:
                self.params['W'+str(idx)] = weight_init_std * np.random.randn(hidden_size, hidden_size)

            self.params['b'+str(idx)] = np.zeros(hidden_size)

        # create layers
        self.layers = OrderedDict()
        for idx in range(layer_num):
            self.layers['Affine'+str(idx)] = layers.Affine(self.params['W'+str(idx)], self.params['b'+str(idx)])
            self.layers['Activate'+str(idx)] = activator

        self.activation_dict = OrderedDict()

    def forward(self, x_batch):
        for layer_name, layer in self.layers.items():
            x_batch = layer.forward(x_batch)
            if 'Activate' in layer_name:
                self.activation_dict[layer_name] = x_batch
        return x_batch


if __name__ == '__main__':
    # activate function
    # act = layers.Sigmoid()
    act = layers.Relu()

    input_size = 100

    # w_std = 1
    # w_std = .01
    # init std = Xavier
    w_std = (1 / input_size) ** (1 / 2)
    # init std = He
    # w_std = (2 / input_size) ** (1 / 2)
    # print input
    network = ActivationNet(input_size=input_size,
                            hidden_size=100,
                            output_size=10,
                            layer_num=5,
                            w_std=w_std,
                            activator=act)
    x = np.random.randn(1000, input_size) * 1
    print('Input', x.shape)
    # print activation tensor(activator output) shape
    network.forward(x)
    for key, value in network.activation_dict.items():
        print(key, value.shape)

    bins_range = 30

    # plot input tensor histogram
    plt.figure(0)
    plt.title('Input')
    plt.hist(x.flatten(), bins=bins_range, range=(0, 1))

    # plot activation tensors(activator output) histogram
    plt.figure(1)
    for idx, key in enumerate(network.activation_dict.keys()):
        ax = plt.subplot(1, 5, idx + 1)
        ax.set_title(key)
        # if isinstance(act, layers.Sigmoid):
        #     ran = (0, 1)
        # else:
        #     ran = None
        ran = (0, 1)
        ax.hist(network.activation_dict[key].flatten(), bins=bins_range, range=ran)
        if idx != 0:
            plt.yticks([], [])

    plt.tight_layout()
    plt.show()
