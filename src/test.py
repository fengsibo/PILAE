# -*- coding: utf-8 -*-
import sys
import os
workpath = os.path.abspath("..")
sys.path.append(workpath)
from src.pilae import PILAE
import src.tools as tools

data_dict = {'mnist.npz': 784,
             'fashionmnist.npz': 784,
             'cifar10.npz': 1024,
             'cifar10RGB.npz': 3072}

DATASET = 'mnist.npz' # 更改数据集只需要替换这里

(X_train, y_train), (X_test, y_test) = tools.load_npz("../dataset/" + DATASET)
X_train = X_train.reshape(-1, data_dict[DATASET]).astype('float64') / 255.
X_test = X_test.reshape(-1, data_dict[DATASET]).astype('float64') / 255.

# 划分小一点的数据集进行训练
num = 500
X_train = X_train[0: num, :]
y_train = y_train[0: num]
X_test = X_test[0: num, :]
y_test = y_test[0: num]

# 创建对象时注意 num_*_layers和len(list)和len(pil*_p)对应(层数和list长度对应)
pilae = PILAE(pilae_p=[500, 400],
              pil_p=[300],
              ae_k_list=[0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              pil_k=0.0,
              acFunc='sig')

pilae.train_pilae(X_train, y_train)
