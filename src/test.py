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
num = 0

DATASET = 'cifar10'
(X_train, y_train), (X_test, y_test) = tools.load_npz("../dataset/"+DATASET+"/"+DATASET+".npz")
X_train = X_train.reshape(-1, 1024).astype('float64')/255
X_test = X_test.reshape(-1, 1024).astype('float64')/255

X_train = X_train[0: 500, :]
y_train = y_train[0: 500]

# f = np.load("../dataset/PMPS/subbands.npz")
# X_train = f['X']
# y_train = f['y']

# # for hog data
# for i in range(1, 27):
#     num = i
#     print("the "+str(num)+" scriptor")
#     X_train, X_test = hg.load_hog("../data/cifar10/cifar10", 0, num)

# data_path = "../data/fashionmnist/fashionmnist"
# hog_list = [0, 2, 4, 6, 8, 16, 18, 20, 22, 24, 26]
# X_train, X_test = hg.select_hog(data_path, hog_list)

# for i in range(1, 100):
#     for j in range(1, 100):
t1 = time.time()
ae_k_list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
pil_k = 0.03
alpha_list = [0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
pil_p = [800, 600]
pilae = PILAE(ae_k_list=ae_k_list, pil_p=pil_p, pil_k=pil_k, alpha=alpha_list, ae_layers=10, pil_layers=0, acFunc='sig')


pilae.fit(X_train, y_train)
cost_time = time.time() - t1
print("Total cost time: %.2f" %cost_time)
# write into .csv file
csv_file = "../log/mnist.csv"
with open(csv_file, 'at+') as csvfile:
    writer = csv.writer(csvfile)
    if os.path.getsize(csv_file):
        writer.writerow(["map", "dims", "layer", "time", "train_acc", "test_acc", "k", "alpha", "beta", "acf"])
        writer.writerow(
            [num, X_train.shape[1], pilae.ae_layers, cost_time, pilae.train_acc, pilae.test_acc, pilae.ae_k_list, pilae.alpha, pilae.acFunc])
    else:
        writer.writerow(
            [num, X_train.shape[1], pilae.ae_layers, cost_time, pilae.train_acc, pilae.test_acc, pilae.ae_k_list, pilae.alpha, pilae.acFunc])
