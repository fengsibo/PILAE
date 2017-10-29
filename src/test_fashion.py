import sys
import os
workpath = os.path.abspath("..")
sys.path.append(workpath)
import src.row_PILAE as rp
import src.tools as tools
import src.Hog as hg
import time
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

# X_train, y_train = tools.load_fashionMNIST()
# X_test, y_test = tools.load_fashionMNIST(kind='t10k')

(_, y_train), (_, y_test) = tools.load_npz("../dataset/fashion_mnist/fashionmnist.npz")

for i in range(1, 27):
    num = i
    print("the " + str(num) + " scriptor")
    X_train, X_test = hg.load_hog("../data/fashion_mnist/fashion_mnist", num)
    X_train *= 10
    X_test *= 10

    t1 = time.time()
    pilae = rp.row_PILAE(k=1.5, alpha=0.8, beta=0.95, activeFunc='sig')
    pilae.fit(X_train, layer=1)
    pilae.predict(X_train, y_train, X_test, y_test)
    t2 = time.time()
    cost_time = t2 - t1
    print("Total cost time: %.2f" %cost_time)
    # write into .csv file
    with open("../log/fashinmnist_hog_maps.csv", 'at+') as csvfile:
        writer = csv.writer(csvfile)
        if num == 1:
            writer.writerow(["map", "dims", "layer", "time", "train_acc", "test_acc", "k", "alpha", "beta", "acf"])
        else:
            writer.writerow([num, X_train.shape[1], pilae.layer, cost_time, pilae.train_acc, pilae.test_acc, pilae.k, pilae.alpha, pilae.beta, pilae.acFunc])