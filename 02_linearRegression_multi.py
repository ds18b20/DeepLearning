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
            self.layers['Activation' + str(idx)] = activation_layer[activation]
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
        for layer in self.layers.values():
            x_batch = layer.forward(x_batch)
        return x_batch

    def loss(self, x_batch, t_batch):
        y_batch = self.predict(x_batch)
        loss = self.last_layer.forward(y_batch, t_batch)
        return loss

    def gradient(self, x_batch, t_batch):
        # forward
        self.predict(x_batch)
        # backward
        dout = 1
        self.last_layer.backward(d_y=dout)
        layers = list(self.layers)
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

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
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def generate_simple_dataset(self, num_examples):
        true_w = np.array([3.0, 7.0, 5.0])
        true_b = np.array([8.0, ])

        features = np.random.randn(num_examples, self.input_size)
        labels = np.dot(features, true_w) + true_b
        epsilon = np.random.normal(0.0, scale=0.01)
        labels += epsilon

        return features, labels

    def update_batch(self, features_set, labels_set):
        # assert len(features_set) == len(labels_set)
        num_examples = len(features_set)
        index = np.random.choice(num_examples, self.batch_size)
        batch_features = features_set[index]
        batch_labels = labels_set[index]

        return batch_features, batch_labels

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
    reg.init_weight(0.01)
    # for key, value in reg.params.items():
    #     print(key, value.shape)
    x, t = reg.generate_simple_dataset(num_examples=1000)
    sample_x, sample_t = reg.update_batch(x, t)

    print(sample_x.shape, sample_t.shape)
    print(sample_x[0], sample_t[0])
    g = reg.gradient(x_batch=sample_x, t_batch=sample_t)
    g_n = reg.numerical_gradient(sample_x, sample_t)

