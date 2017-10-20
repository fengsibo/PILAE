import sys
sys.path.append("../src")
import src.row_PILAE as rp
import src.tools as tools
import src.Hog as hg
from sklearn import preprocessing
import numpy as np
import time
import multiprocessing


(X_train, y_train), (X_test, y_test) = tools.load_MNISTData()

# def fun(array, type, num):
#     t1 = time.time()
#     hg.extract_featuer(array, type, num)
#     t2 = time.time()
#     print(type, t2 - t1)
#
# if __name__ == '__main__':
#     i = 0
#     while i < 9:
#         fun(X_train, "train", i + 1)
#         fun(X_test, "test", i + 1)
#         i+=1

X_train = X_train.reshape(-1, 784).astype('float32')
X_train2 = X_train.reshape(-1, 784, order='F').astype('float32')
X_train = np.concatenate((X_train, X_train2), axis=1)
X_test = X_test.reshape(-1, 784).astype('float32')
X_test2 = X_test.reshape(-1, 784, order='F').astype('float32')
X_test = np.concatenate((X_test, X_test2), axis=1)

# X_train = tools.load_pickle("../data/mnist_hog_feature_train_5.plk")
# X_test = tools.load_pickle("../data/mnist_hog_feature_test_5.plk")
# X_train = X_train.T[:60000, :]*100
# X_test = X_test.T[:10000, :]*100
# minmax_scaler = preprocessing.MinMaxScaler()
# X_train = minmax_scaler.fit_transform(X_train)
# X_test = minmax_scaler.fit_transform(X_test)
X_train /= 255
X_test /= 255

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
pilae = rp.row_PILAE(k=2.8, alpha=0.9, beta=0.9)
pilae.fit(X_train, layer=1)
pilae.predict(X_train, y_train, X_test, y_test)
t2 = time.time()
print("Total cost time: %.4f" %(t2 - t1))