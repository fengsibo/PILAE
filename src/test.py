import src.row_PILAE as rp
import src.tools as tools
import src.Hog as hg
from sklearn import preprocessing
import numpy as np
import time
import multiprocessing


(X_train, y_train), (X_test, y_test) = tools.load_MNISTData()

def fun(array, type):
    t1 = time.time()
    hg.extract_featuer(array, type)
    t2 = time.time()
    print(type, t2 - t1)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=2)
    pool.apply_async(fun, (X_train, ))
    pool.apply_async(fun, (X_test, ))
    pool.close()
    pool.join()

# t = time.time()
# hg.extract_featuer(X_train)
# t2 = time.time()
# print(t2 - t)

# X_train = X_train.reshape(-1, 784).astype('float32')
# X_test = X_test.reshape(-1, 784).astype('float32')
#
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

# pilae = rp.row_PILAE(k=1, alpha=0.8, beta=0.7)
# pilae.fit(X_train, layer=1)
# pilae.predict_softmax(X_train, y_train, X_test, y_test)