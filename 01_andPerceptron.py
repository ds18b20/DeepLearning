#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import reduce


class Perceptron(object):
    def __init__(self, weights_num, activator):
        """
        define activator function
        initialize weights & bias
        :param weights_num: number of weights
        :param activator: activator function
        """
        self.activator = activator
        self.weights = [0.0 for _ in range(weights_num)]
        self.bias = 0.0

    def __str__(self):
        """
        print instance
        :return: present weights and bias
        """
        return 'weights: {}\nbias: {}'.format(self.weights, self.bias)

    def predict(self, input_vec):
        """
        use present model to predict result
        :param input_vec: input vector for prediction
        :return: prediction result
        """
        reduce_list = [x * y for (x, y) in zip(input_vec, self.weights)]
        return self.activator(reduce(lambda x, y: x + y, reduce_list) + self.bias)

    def train(self, input_vectors, labels, iteration, rate):
        """
        iterate train function for iteration times
        :param input_vectors: input vectors
        :param labels: labels
        :param iteration: iteration number
        :param rate: learning rate
        :return: None
        """
        for i in range(iteration):
            self._one_iteration(input_vectors, labels, rate)

    def _one_iteration(self, input_vectors, labels, rate):
        """
        one iteration to go through all data
        :param input_vectors: input vectors
        :param labels: labels
        :param rate: learning rate
        :return: None
        """
        samples = zip(input_vectors, labels)
        for (input_vector, label) in samples:
            output = self.predict(input_vector)
            self._update_weights(input_vector, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        """
        update weights
        :param input_vec:
        :param output:
        :param label:
        :param rate:
        :return:
        """
        delta = label - output
        self.weights = [w + rate * delta * x for x, w in zip(input_vec, self.weights)]
        self.bias += rate * delta


def f(x):
    """
    perceptron function
    :param x:
    :return:
    """
    return 1 if x > 0 else 0


def get_train_dataset():
    """
    get dataset for training
    :return: input_vectors labels
    """
    input_vectors = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vectors, labels


def train_perceptron():
    """
    iterate for 10 times with rate 0.1
    :return: Perceptron instance
    """
    p = Perceptron(weights_num=2, activator=f)
    input_vectors, labels = get_train_dataset()
    p.train(input_vectors, labels, iteration=10, rate=0.1)
    return p


if __name__ == '__main__':
    and_perceptron = train_perceptron()
    print(and_perceptron)

    print('1 and 1 = {}'.format(and_perceptron.predict([1, 1])))
    print('1 and 0 = {}'.format(and_perceptron.predict([1, 0])))
    print('0 and 1 = {}'.format(and_perceptron.predict([0, 1])))
    print('0 and 0 = {}'.format(and_perceptron.predict([0, 0])))
