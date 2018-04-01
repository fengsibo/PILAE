# -*- coding: utf-8 -*-
import sys
import os
workpath = os.path.abspath("..")
sys.path.append(workpath)
from src.pilae import PILAE
import src.tools as tools
import src.Hog as hg


data_dict = {'mnist.npz': 784,
             'fashionmnist.npz': 784,
             'cifar10.npz': 1024,
             'cifar10RGB.npz': 3072}

DATASET = 'cifar10.npz' # 更改数据集只需要替换这里

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
pilae = PILAE(ae_k_list=[0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              pilae_p=[500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300],
              pil_p=[300],
              pil_k=0.0,
              alpha=[0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
              num_ae_layers=5,
              num_pil_layers=0,
              acFunc='sig')

pilae.train_pilae(X_train, y_test)