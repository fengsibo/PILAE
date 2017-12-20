import sys
import os
workpath = os.path.abspath("..")
sys.path.append(workpath)
import numpy as np
from sklearn import preprocessing
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import pickle
import csv
import h5py
import pandas as pd
import src.tools as tools


class PILAE(object):

    def __init__(self, hidden_units, batchsize, k, actFun='sig'):
        self.hidden_units = hidden_units
        self.batchsize = batchsize
        self.k = k
        self.actFun = actFun
        self.layers = len(HiddernNeurons)
        self.w = list(range(self.layers))
        self.preiS = list(range(self.layers))
        self.InputWeight = list(range(self.layers))

    def makebatches(self, data):
        randomorder = random.sample(range(len(data)), len(data))
        numbatches = int(len(data) / self.batchsize)
        for batch in range(numbatches):
            order = randomorder[batch * self.batchsize:(batch + 1) * self.batchsize]
            yield data[order, :]

    def fit(self, data):
        hidden_weight = []
        hidden_layer = []
        for numbatch, delta_X in enumerate(self.makebatches(data)):
            for i in range(self.layers):
                print("the "+str(i)+" th layer"+"the " + str(numbatch) + " th batch")
                U, s, V = np.linalg.svd(delta_X)
                V = V.T
                V = V[:, :self.HiddernNeurons[i]]
                S = np.zeros((U.shape[0], V.shape[0]))
                S[:s.shape[0], :s.shape[0]] = np.diag(s)
                S = S[:self.HiddernNeurons[i], :self.HiddernNeurons[i]]
                S[S < 1e-5] = 0
                S[S != 0] = 1 / S[S != 0]
                self.InputWeight[i] = V.dot(S)

                hidden_weight.append(V.dot(S))

                if numbatch == 0:
                    H = delta_X.dot(self.InputWeight[i])
                    delta_H = self.activeFunction(H, self.actFun)
                    iS = np.linalg.inv(delta_H.T.dot(delta_H) + np.eye(delta_H.shape[1]) * self.k[i])
                    OW = iS.dot(delta_H.T).dot(delta_X)
                else:
                    H = delta_X.dot(self.w[i].T)
                    delta_H = self.activeFunction(H, self.actFun)
                    iC = np.linalg.inv(np.eye(delta_H.shape[0]) + delta_H.dot(self.preiS[i]).dot(delta_H.T))
                    iS = self.preiS[i] - self.preiS[i].dot(delta_H.T).dot(iC).dot(delta_H).dot(self.preiS[i])
                    alpha = np.eye(delta_H.shape[1]) - self.preiS[i].dot(delta_H.T).dot(iC).dot(delta_H)
                    belta = self.preiS[i].dot(
                        np.eye(delta_H.shape[1]) - delta_H.T.dot(iC).dot(delta_H).dot(self.preiS[i])).dot(
                        delta_H.T).dot(delta_X)
                    OW = alpha.dot(self.w[i]) + belta
                hidden_layer.append(delta_H)

                self.w[i] = OW
                self.preiS[i]=iS

                tempH = delta_X.dot(OW.T)
                delta_X = self.activeFunction(tempH, self.actFun)
                delta_X = preprocessing.scale(delta_X.T)
                delta_X = delta_X.T

    def feature_extrackted(self, x):
        feature = x
        for i in range(self.layers):
            feature = feature.dot(self.w[i].T)
            feature = self.activeFunction(feature, self.actFun)
        return feature

    def predict(self, X_train, X_test, y_train, y_test):
        train_feature = pilae.feature_extrackted(X_train)
        test_feature = pilae.feature_extrackted(X_test)
        model = self.regression_classifier(train_feature, y_train)
        train_predict = model.predict(train_feature)
        self.train_acc = accuracy_score(train_predict, y_train) * 100
        print("Accuracy of train data set: %.2f" % self.train_acc, "%")
        test_predict = model.predict(test_feature)
        self.test_acc = accuracy_score(test_predict, y_test) * 100
        print("Accuracy of test data set: %.2f" % self.test_acc, "%")

    def regression_classifier(self, train_X, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=200)
        model.fit(train_X, train_y)
        return model

    def svm_classifier(self, train_X, train_y):
        from sklearn.svm import SVC
        model = SVC(kernel='linear', probability=False)
        model.fit(train_X, train_y)
        return model

    def linearsvm_classifier(self, train_X, train_y):
        from sklearn import svm
        model = svm.LinearSVC()
        model.fit(train_X, train_y)
        return model

    def activeFunction(self, tempH, func='sig', p=1.):
        def relu(x, p):
            x[x < 0] = 0
            return x

        switch = {
            'sig': lambda x, p: 1 / (1 + np.exp(-p * x)),
            'sin': lambda x, p: np.sin(x),
            'relu': relu,
            'srelu': lambda x, p: np.log(1 + np.exp(x)),
            'tanh': lambda x, p: np.tanh(p * x),
        }
        fun = switch.get(func)
        return fun(tempH, p)


if __name__ == '__main__':

    batchsize = 2000
    HiddernNeurons = [600]
    k = [1.2]

# mnist data++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    (X_train, y_train), (X_test, y_test) = tools.load_npz("../dataset/mnist/mnist.npz")
    X_train = X_train.reshape(-1, 784).astype('float32')/255
    X_test = X_test.reshape(-1, 784).astype('float32')/255

    t1 = time.time()
    pilae = PILAE(HiddernNeurons, batchsize, k, actFun='sig')
    pilae.fit(X_train)
    pilae.predict(X_train, X_test, y_train, y_test)
    t2 = time.time()
    print("time cost of training PILAE: %.4f" % (t2 - t1))








