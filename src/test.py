# -*- coding: utf-8 -*-
import sys
import os

workpath = os.path.abspath("..")
sys.path.append(workpath)
from src.pilae import PILAE
import src.tools as tools
import src.Hog as hg
from sklearn import preprocessing
import numpy as np
import time
import multiprocessing
import csv

DATASET = 'mnist'
(X_train, y_train), (_, _) = tools.load_npz("../dataset/" + DATASET + "/" + DATASET + ".npz")
X_train = X_train.reshape(-1, 784).astype('float64') / 255
# X_test = X_test.reshape(-1, 784).astype('float64')/255

X_train = X_train[0: 500, :]
y_train = y_train[0: 500]

# f = np.load("../dataset/PMPS/subbands.npz")
# X_train = f['X']
# y_train = f['y']


ae_k_list = [0.78, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
pil_k = 0.03
alpha_list = [0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
pil_p = [400, 300]
ae_layers = 10
pil_layers = 0
pilae = PILAE(ae_k_list=ae_k_list, pil_p=pil_p, pil_k=pil_k, alpha=alpha_list, ae_layers=ae_layers,
              pil_layers=pil_layers, acFunc='sig')
pilae.train_pilae(X_train, y_train)

# # write into .csv file
# csv_file = "../log/mnist.csv"
# with open(csv_file, 'at+') as csvfile:
#     writer = csv.writer(csvfile)
#     if os.path.getsize(csv_file):
#         writer.writerow(["map", "dims", "layer", "train_acc", "test_acc", "k", "alpha", "beta", "acf"])
#         writer.writerow(
#             [num, X_train.shape[1], pilae.ae_layers, pilae.train_acc, pilae.test_acc, pilae.ae_k_list, pilae.alpha, pilae.acFunc])
#     else:
#         writer.writerow(
#             [num, X_train.shape[1], pilae.ae_layers, pilae.train_acc, pilae.test_acc, pilae.ae_k_list, pilae.alpha, pilae.acFunc])
