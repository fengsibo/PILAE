import sys
import os
workpath = os.path.abspath("..")
sys.path.append(workpath)
import src.row_PILAE as rp
import src.tools as tools
import src.Hog as hg
from sklearn import preprocessing
import numpy as np
import time
import multiprocessing
import csv
num = 0

DATASET = 'mnist'
(X_train, y_train), (X_test, y_test) = tools.load_npz("../dataset/"+DATASET+"/"+DATASET+".npz")
X_train = X_train.reshape(-1, 784).astype('float32')/255
X_test = X_test.reshape(-1, 784).astype('float32')/255

# # for hog data
# for i in range(1, 27):
#     num = i
#     print("the "+str(num)+" scriptor")
#     X_train, X_test = hg.load_hog("../data/cifar10/cifar10", 0, num)

# data_path = "../data/fashionmnist/fashionmnist"
# hog_list = [0, 2, 4, 6, 8, 16, 18, 20, 22, 24, 26]
# X_train, X_test = hg.select_hog(data_path, hog_list)


# minmax_scaler = preprocessing.MinMaxScaler()
# X_train = minmax_scaler.fit_transform(X_train)
# X_test = minmax_scaler.fit_transform(X_test)

# train_mean = X_train.mean(axis=1)
# train_mean = train_mean.reshape(60000, 1)
# train_std = X_train.std(axis=1)
# train_std = train_std.reshape(60000, 1)
# X_train = (X_train - train_mean)/train_std

# test_mean = X_test.mean(axis=1)
# test_mean = test_mean.reshape(10000, 1)
# test_std = X_test.std(axis=1)
# test_std = test_std.reshape(10000, 1)
# X_test = (X_test - test_mean)/test_std

# X_train = preprocessing.scale(X_train, axis=1)
# X_test = preprocessing.scale(X_test, axis=1)

t1 = time.time()
k_list = [0.78, 0.0]
pilk_list = [0.04, 0.03]
pilae = rp.PILAE(k=k_list, pilk=pilk_list, alpha=0.9, beta=0.9, layer=2, activeFunc='sig')
pilae.fit(X_train, y_train)
pilae.predict(X_train, y_train, X_test, y_test)
t2 = time.time()
cost_time = t2 - t1
print("Total cost time: %.2f" %cost_time)
# write into .csv file
with open("../log/"+DATASET+"_hog.csv", 'at+') as csvfile:
    writer = csv.writer(csvfile)
    if num == 1:
        writer.writerow(["map", "dims", "layer", "time", "train_acc", "test_acc", "k", "alpha", "beta", "acf"])
        writer.writerow(
            [num, X_train.shape[1], pilae.layer, cost_time, pilae.train_acc, pilae.test_acc, pilae.k, pilae.alpha,
             pilae.beta, pilae.acFunc])
    else:
        writer.writerow(
            [num, X_train.shape[1], pilae.layer, cost_time, pilae.train_acc, pilae.test_acc, pilae.k, pilae.alpha,
             pilae.beta, pilae.acFunc])
