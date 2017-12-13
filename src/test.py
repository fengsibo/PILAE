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
k_list = [0.78, 0.43]
pilk = 0.07
alpha_list = [0.8, 0.7]
pil_p = [2000, 1000]
pilae = rp.PILAE(k=k_list, pilk=pilk, alpha=alpha_list, pil_p = pil_p, AE_layer=1, PIL_layer=2, activeFunc='sig')
pilae.fit(X_train, y_train)
# pilae.predict(X_train, y_train, X_test, y_test)
t2 = time.time()
cost_time = t2 - t1
print("Total cost time: %.2f" %cost_time)
# write into .csv file
with open("../log/pulsar.csv", 'at+') as csvfile:
    writer = csv.writer(csvfile)
    if num == 0:
        num = 1
        writer.writerow(["map", "dims", "layer", "time", "train_acc", "test_acc", "k", "alpha", "beta", "acf"])
        writer.writerow(
            [num, X_train.shape[1], pilae.ae_layer, cost_time, pilae.train_acc, pilae.test_acc, pilae.k, pilae.alpha, pilae.acFunc])
    else:
        writer.writerow(
            [num, X_train.shape[1], pilae.ae_layer, cost_time, pilae.train_acc, pilae.test_acc, pilae.k, pilae.alpha, pilae.acFunc])
