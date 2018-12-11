# coding: utf-8
# https://qiita.com/inoory/items/3ea2d447f6f1e8c40ffa#%E3%81%AF%E3%81%98%E3%82%81%E3%81%AB
# import logging; logging.basicConfig(level=logging.INFO)
# import logging; logging.basicConfig(level=logging.DEBUG)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def create_sig():
    target_sig = np.random.normal(size=1000) * 1.0
    delay = 800
    sig1 = np.random.normal(size=2000) * 0.2
    sig1[delay:delay + 1000] += target_sig
    sig2 = np.random.normal(size=2000) * 0.2
    sig2[:1000] += target_sig
    return sig1, sig2


def show_sig(graph_dict):
    n = len(graph_dict)
    c = cm.rainbow(np.linspace(0, 1, n))
    _, figs = plt.subplots(n, 1)
    idx = 0
    for key, value in graph_dict.items():
        # fig[0]: train accuracy & test accuracy
        figs[idx].plot(value[0], value[1], color=c[idx], label=key)
        figs[idx].legend()
        idx += 1
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sig1, sig2 = create_sig()
    assert len(sig1) == len(sig2)
    length = len(sig1)
    x = np.arange(length)

    cor = np.correlate(sig1, sig2, 'full')
    estimated_delay = cor.argmax() - (len(sig2) - 1)
    print('estimated_delay:', estimated_delay)

    x_shift = np.arange(len(cor)) - (len(sig2) - 1)
    sig2_shift = np.roll(sig2, estimated_delay)
    graph_dict = {'sig1': (x, sig1), 'sig2': (x, sig2), 'sig2_shift': (x, sig2_shift), 'cor': (x, cor[length-1:])}
    show_sig(graph_dict)
