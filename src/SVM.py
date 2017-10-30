import sys
import os
workpath = os.path.abspath("..")
sys.path.append(workpath)
import numpy as np
import math
import time
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import src.tools as tools
from sklearn.metrics import accuracy_score

def linearSVM(X_train, y_train, X_test, y_test):
    print(X_train.shape)
    t1 = time.time()
    from sklearn.svm import SVC
    model = SVC(kernel='linear', probability=False)
    model.fit(X_train, y_train)
    train_predict = model.predict(X_train)
    train_acc = accuracy_score(train_predict, y_train)*100
    print("Accuracy of train data set: %.2f" %train_acc, "%")
    test_predict = model.predict(X_test)
    test_acc = accuracy_score(test_predict, y_test)*100
    print("Accuracy of test data set: %.2f" %test_acc, "%")
    t2 = time.time()
    cost_time = t2 - t1
    print("linear cost time: %.2f" % cost_time)

def rbfSVM(X_train, y_train, X_test, y_test):
    t1 = time.time()
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', C=5, gamma=0.05, probability=False)
    model.fit(X_train, y_train)
    train_predict = model.predict(X_train)
    train_acc = accuracy_score(train_predict, y_train)*100
    print("Accuracy of train data set: %.2f" %train_acc, "%")
    test_predict = model.predict(X_test)
    test_acc = accuracy_score(test_predict, y_test)*100
    print("Accuracy of test data set: %.2f" %test_acc, "%")
    t2 = time.time()
    cost_time = t2 - t1
    print("rbf cost time: %.2f" % cost_time)

def polySVM(X_train, y_train, X_test, y_test):
    (X_train, y_train), (X_test, y_test) = tools.load_npz(path)
    t1 = time.time()
    from sklearn.svm import SVC
    model = SVC(kernel='poly', probability=False)
    model.fit(X_train, y_train)
    train_predict = model.predict(X_train)
    train_acc = accuracy_score(train_predict, y_train)*100
    print("Accuracy of train data set: %.2f" %train_acc, "%")
    test_predict = model.predict(X_test)
    test_acc = accuracy_score(test_predict, y_test)*100
    print("Accuracy of test data set: %.2f" %test_acc, "%")
    t2 = time.time()
    cost_time = t2 - t1
    print("poly cost time: %.2f" % cost_time)

def sigSVM(X_train, y_train, X_test, y_test):
    t1 = time.time()
    from sklearn.svm import SVC
    model = SVC(kernel='sigmoid', probability=False)
    model.fit(X_train, y_train)
    train_predict = model.predict(X_train)
    train_acc = accuracy_score(train_predict, y_train)*100
    print("Accuracy of train data set: %.2f" %train_acc, "%")
    test_predict = model.predict(X_test)
    test_acc = accuracy_score(test_predict, y_test)*100
    print("Accuracy of test data set: %.2f" %test_acc, "%")
    t2 = time.time()
    cost_time = t2 - t1
    print("sig cost time: %.2f" % cost_time)

path = "../dataset/mnist/mnist.npz"
path2 = "../dataset/fashionmnist/fashionmnist.npz"
path3 = "../dataset/cifar10/cifar10.npz"

(X_train, y_train), (X_test, y_test) = tools.load_npz(path)
X_train = X_train.reshape(-1, 784).astype('float32')/255
X_test = X_test.reshape(-1, 784).astype('float32')/255
print("===========mnist==========")
linearSVM(X_train, y_train, X_test, y_test)
rbfSVM(X_train, y_train, X_test, y_test)
polySVM(X_train, y_train, X_test, y_test)
sigSVM(X_train, y_train, X_test, y_test)

(X_train, y_train), (X_test, y_test) = tools.load_npz(path2)
X_train = X_train.reshape(-1, 784).astype('float32')/255
X_test = X_test.reshape(-1, 784).astype('float32')/255
print("===========fashionmnist==========")
linearSVM(X_train, y_train, X_test, y_test)
rbfSVM(X_train, y_train, X_test, y_test)
polySVM(X_train, y_train, X_test, y_test)
sigSVM(X_train, y_train, X_test, y_test)

(X_train, y_train), (X_test, y_test) = tools.load_npz(path3)
X_train = X_train.reshape(-1, 1024).astype('float32')/255
X_test = X_test.reshape(-1, 1024).astype('float32')/255
print("===========cifar10==========")
linearSVM(X_train, y_train, X_test, y_test)
rbfSVM(X_train, y_train, X_test, y_test)
polySVM(X_train, y_train, X_test, y_test)
sigSVM(X_train, y_train, X_test, y_test)


