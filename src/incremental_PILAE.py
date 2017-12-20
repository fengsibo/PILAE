'''
for input matrix row vector
'''

import sys
import os
sys.path.append("..")
import collections
import random
import numpy as np
import time
import src.tools as tools

HParams = collections.namedtuple('HParams',
                                 'batch_size, num_classes, ae_k, pil_k, '
                                 'ae_p, pil_p, ae_layers, pil_layers, '
                                 'acFunc')

class PILAE(object):
    def __init__(self, X, y, hyperParams):
        self.X, self.y = self.random_shuffle(X, y)
        self.hyperParams = hyperParams
        self.num_batch = X.shape[0]//hyperParams.batch_size
        self.fit()

    def fit(self):
        for i in range(self.num_batch):
            X, y = self.next_batch(i)
            for layer in range(self.hyperParams.ae_layers):
                pinv_X = np.linalg.pinv(X)
                print(pinv_X.shape)
                input_H = X.dot(pinv_X[:, 0: self.hyperParams.ae_p[layer]])
                output_H = self.activeFunction(input_H, self.hyperParams.acFunc)
                inv_H = np.linalg.inv(output_H.T.dot(
                    output_H + self.hyperParams.ae_k[layer] * np.eye(output_H.shape[1])))
                Wd = inv_H.dot(output_H.T).dot(X)


    def next_batch(self, index):
        m, n = index * self.hyperParams.batch_size, (index + 1) * self.hyperParams.batch_size
        return self.X[m: n, :], self.y[m: n, :]

    def random_shuffle(self, X, y):
        m, n = X.shape
        index = [i for i in range(m)]
        random.shuffle(index)
        X = X[index]
        y = y[index]
        y = tools.to_categorical(y)
        return X, y

    def activeFunction(self, tempH, func='sig'):
        switch = {
            'sig': lambda x: 1 / (1 + np.exp(-x)),
            'sin': lambda x: np.sin(x),
            'srelu': lambda x: np.log(1 + np.exp(x)),
            'tanh': lambda x: np.tanh(x),
            'swish':lambda x: x/(1 + np.exp(-x)),
            'relu' : lambda x: np.maximum(0, x),
        }
        fun = switch.get(func)
        return fun(tempH)
