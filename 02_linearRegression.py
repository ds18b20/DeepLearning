import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class OneLayerRegression(object):
    def __init__(self, input_vec_size: int, batch_size: int=10):
        self.__input_vec_size = input_vec_size
        self.__batch_size = batch_size

        self.__batch_features = None
        self.__batch_labels = None
        
        self.__batch_outputs = None

        self.__w = np.random.normal(loc=0.0, scale=0.01, size=[2, ])
        self.__b = np.zeros(shape=[1, ])
        self.__grad_w = np.random.normal(loc=0.0, scale=0.01, size=[2, ])
        self.__grad_b = np.zeros(shape=[1, ])

    def set_w(self, value):
        self.__w = value

    def set_b(self, value):
        self.__b = value

    def get_w(self):
        return self.__w

    def get_b(self):
        return self.__b

    def generate_simple_dataset(self, num_examples):
        true_w = np.array([3.0, 7.0])
        true_b = np.array([5.0, ])

        features = np.random.randn(num_examples, self.__input_vec_size)
        labels = np.dot(features, true_w) + true_b
        epsilon = np.random.normal(0.0, scale=0.01)
        labels += epsilon

        return features, labels
    
    # linear regression
    def forward_pass(self):
        self.__batch_outputs = np.dot(self.__batch_features, self.__w) + self.__b  # (batch, )

    def squared_loss(self):
        return ((self.__batch_outputs - self.__batch_labels)**2).sum() / self.__batch_size

    def backward_pass(self):
        self.gradient_cal()
        self.sgd(lr=0.01)

    def gradient_cal(self):
        # assert len(batch_features) == len(batch_labels)
        delta = self.__batch_outputs - self.__batch_labels  # (mini batch, )
        self.__grad_w = np.dot(delta, self.__batch_features) / self.__batch_size  # (batch, )dot(batch, vec_size)=(vec_size, )
        self.__grad_b = delta.sum() / self.__batch_size

    def sgd(self, lr=0.01):
        self.__w -= lr * self.__grad_w
        self.__b -= lr * self.__grad_b

    def update_batch(self, features_set, labels_set):
        # assert len(features_set) == len(labels_set)
        num_examples = len(features_set)
        index = np.random.choice(num_examples, self.__batch_size)
        self.__batch_features = features_set[index]
        self.__batch_labels = labels_set[index]


if __name__ == '__main__':
    # fig = plt.figure(figsize=(5, 5), dpi=80)
    # ax = plt.axes(projection='3d')
    #
    # ax.scatter(features[:, 0], features[:, 1], labels, c='blue')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.set_title('scatter')
    #
    # plt.show()
    reg = OneLayerRegression(input_vec_size=2, batch_size=10)
    sample_features, sample_labels = reg.generate_simple_dataset(num_examples=1000)

    loss = []
    for i in range(1000):
        reg.update_batch(sample_features, sample_labels)
        reg.forward_pass()
        reg.backward_pass()

        loss.append(reg.squared_loss())

    print('W: {}'.format(reg.get_w()))
    print('b: {}'.format(reg.get_b()))

    plt.plot(loss)
    plt.show()
