import sys
sys.path.append("../src")
import src.row_PILAE as rp
import src.tools as tools
import src.Hog as hg
from sklearn import preprocessing
import numpy as np
import time
import multiprocessing
import csv


(X_train, y_train), (X_test, y_test) = tools.load_MNISTData()


# X_train = X_train.reshape(-1, 784).astype('float32')
# X_test = X_test.reshape(-1, 784).astype('float32')


print(X_train.shape, X_test.shape)
minmax_scaler = preprocessing.MinMaxScaler()
X_train = minmax_scaler.fit_transform(X_train)
X_test = minmax_scaler.fit_transform(X_test)
# X_train /= 255
# X_test /= 255

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
pilae = rp.row_PILAE(k=2.7, alpha=0.2, beta=0.8, activeFunc='tanh')
pilae.fit(X_train, layer=2)
pilae.predict(X_train, y_train, X_test, y_test)
t2 = time.time()
cost_time = t2 - t1
print("Total cost time: %.2f" %cost_time)
# write into .csv file
with open("../log/mnist_info.csv", 'at+') as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(["layer", "time", "train_acc", "test_acc", "k", "alpha", "beta", "acf"])
    writer.writerow([pilae.layer, cost_time, pilae.train_acc, pilae.test_acc, pilae.k, pilae.alpha, pilae.beta, pilae.acFunc])