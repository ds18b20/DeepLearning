#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
import os


class TEXT(object):
    def __init__(self, root):
        self.root = root
        self.corpus_chars = None
        self.char_to_idx_dict = None
        self.idx_to_char_dict = None

    def load(self, filename='jaychou_lyrics.txt', convert=True):
        with open(os.path.join(self.root, filename), 'r', encoding='utf-8') as f:
            data = f.read()  # --> str
        if convert:
            data = data.replace('\n', ' ').replace('\r', ' ').replace('\u3000', ' ')
        # print(data.find('\u3000'))
        # print(data[11044-2: 11044+2])
        data = np.array(list(data))
        self.corpus_chars = list(set(data))
        logging.info('data size: {}, vocab size: {}'.format(len(data), len(self.corpus_chars)))

        self.char_to_idx_dict = {ch: i for i, ch in enumerate(self.corpus_chars)}
        self.idx_to_char_dict = {i: ch for i, ch in enumerate(self.corpus_chars)}
        return data

    def char_to_idx(self, batch_x_char):
        batch_x_idx = np.zeros_like(batch_x_char, dtype=np.int)
        it = np.nditer(batch_x_char, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            batch_x_idx[idx] = self.char_to_idx_dict[batch_x_char[idx]]
            it.iternext()
        return batch_x_idx

    def idx_to_char(self, batch_x_idx):
        batch_x_char = np.zeros_like(batch_x_idx)
        it = np.nditer(batch_x_idx, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            batch_x_char[idx] = self.idx_to_char_dict[batch_x_idx[idx]]
            it.iternext()
        return batch_x_char

    def get_one_batch_random(self, data, batch_size, steps_num):
        x = np.array([])
        y = np.array([])
        valid_len = len(data) - steps_num - 1
        start_pts = np.random.choice(valid_len, batch_size)
        for idx, start_pt in enumerate(start_pts):
            tmp_x = data[start_pt: start_pt + steps_num]
            tmp_y = data[start_pt + 1: start_pt + 1 + steps_num]
            if idx == 0:
                x = tmp_x
                y = tmp_y
            else:
                x = np.concatenate((x, tmp_x))
                y = np.concatenate((y, tmp_y))
        return x.reshape(batch_size, steps_num), y.reshape(batch_size, steps_num)

    def data_iter_random(self, data, batch_size, steps_num):
        num_examples = (len(data) - 1) // steps_num
        epoch_size = num_examples // batch_size
        example_indices = list(range(num_examples))
        np.random.shuffle(example_indices)
        # 返回从 pos 开始的长为 num_steps 的序列
        _data = lambda pos: data[pos: pos + steps_num]
        for i in range(epoch_size):
            # 每次读取 batch_size 个随机样本。
            idx = i * batch_size
            batch_indices = example_indices[idx: idx + batch_size]
            x = [_data(j * steps_num) for j in batch_indices]
            y = [_data(j * steps_num + 1) for j in batch_indices]
            yield x, y

    def data_iter_consecutive(self, data, batch_size, steps_num):
        batch_len = len(data) // batch_size
        indices = data[0: batch_size * batch_len].reshape((batch_size, batch_len))
        epoch_size = (batch_len - 1) // steps_num
        for i in range(epoch_size):
            idx = i * steps_num
            x = indices[:, idx: idx + steps_num]
            y = indices[:, idx + 1: idx + steps_num + 1]
            yield x, y


lyrics = TEXT('../datasets/text')
lyrics_data = lyrics.load()

print(lyrics.char_to_idx(np.array([['想', '要', '有'], ['想', '要', '有']])))

print(lyrics_data[0:10])
print(lyrics.char_to_idx(lyrics_data[0:10]))
print(lyrics_data[1:11])
print(lyrics.char_to_idx(lyrics_data[1:11]))
