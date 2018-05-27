# -*- coding: utf-8 -*-
import sys
import os

workpath = os.path.abspath("..")
sys.path.append(workpath)
from src.pilae import PILAE
import src.tools as tools
import numpy as np
import matplotlib.pyplot as plt

data_dict = {'mnist.npz': 784,
             'fashionmnist.npz': 784,
             'cifar10.npz': 1024,
             'cifar10RGB.npz': 3072}

DATASET = 'mnist.npz'  # 更改数据集只需要替换这里

#这个是自己写的读数据方法，知道保证读出来的数据是numpy格式的二维数据就可以
(X_train, y_train), (X_test, y_test) = tools.load_npz("../dataset/" + DATASET)

X_train = X_train.reshape(-1, data_dict[DATASET]).astype('float64') / 255.
X_test = X_test.reshape(-1, data_dict[DATASET]).astype('float64') / 255.
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])
X_train, y_train, X_test, y_test = tools.split_dataset(X, y, 0.5)


pilae = PILAE(pilae_p=[p],
              pil_p=[],
              ae_k_list=[0.7],
              pil_k=0.0,
              acFunc='sig')
pilae.train_pilae(train_X, train_y)
pilae.classifier(train_X, train_y, test_X, test_y)


