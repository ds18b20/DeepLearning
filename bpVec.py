#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from activators import SigmoidActivator


# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        """
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # fg权重数组W
        # bg权重数组W.T
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))
        self.input = np.zeros((input_size, 1))

    def forward(self, input_array):
        """
        前向计算
        input_array: 输入向量，维度必须等于input_size
        """
        # 式2
        # self.input = np.array(input_array).T
        self.input = np.array(input_array).reshape(len(input_array), 1)
        self.output = self.activator.forward(np.dot(self.W, self.input) + self.b)

    def backward(self, delta_array):
        """
        反向计算W和b的梯度
        delta_array: 从上一层(L+1层)传递过来的误差项
        """
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array
        
    def update(self, learning_rate):
        """
        使用梯度下降算法更新权重
        """
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def __str__(self):
        return 'W.shape: {} input.shape: {} output.shape: {}'.format(self.W.shape, self.input.shape, self.output.shape)


# 神经网络类
class Network(object):
    def __init__(self, layers):
        """
        构造函数
        """
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )

    def predict(self, sample):
        """
        使用神经网络实现预测
        sample: 输入样本
        """
        output = sample
        for layer in self.layers:
            # print(layer)
            layer.forward(output)
            # print(layer)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        """
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        label = np.array(label).reshape(len(label), 1)
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        # print(str(delta.shape))
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)
        
if __name__ == '__main__':
    # network_example = Network([784, 300, 10])
    network_example = Network([2, 3, 2])

