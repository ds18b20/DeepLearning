#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import datasets
import matplotlib.pyplot as plt
import numpy as np


def softmax(X):
    X = X - X.max(axis=1, keepdims=True)
    exp = np.exp(X)
    return exp / np.sum(exp, axis=1, keepdims=True)


def net(X, W, b):
    return softmax(np.dot(X, W) + b)
    
    
def cross_entropy(y_hat, y):
    index_row = range(y_hat.shape[0])
    index_column = y
    picked = y_hat[index_row, index_column]
    return -np.log(picked)    
    
    
def accuracy(y_hat: np.array, y: np.array):
    return (y_hat.argmax(axis=1) == y).mean()
    
    
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
    mnist = datasets.MNIST()
    train_x, train_y, test_x, test_y = mnist.load(image_flat=True, label_one_hot=False)
    sample_x, sample_y = datasets.get_one_batch(train_x, train_y, 5)
    # show_fashion_imgs(sample_x, sample_y)
    # print(softmax(np.arange(12).reshape((3,4))))
    P = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
    L = np.array([2, 1])
    print(cross_entropy(P, L))
    print(-np.log([0.7, 0.3]))
    print(accuracy(P, L))
    
    num_inputs, num_outputs = 28*28, 10
    W = np.random.normal(loc=0.0, scale=0.1, size=(num_inputs, num_outputs))
    b = np.zeros(num_outputs)
    
    print(net(sample_x, W, b).shape)
