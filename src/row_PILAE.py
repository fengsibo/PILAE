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


class PILAE(object):
    def __init__(self, k, pilk, alpha=0.8, beta=0.9, layer=1, activeFunc='sig'):
        self.k = k
        self.pilk = pilk
        self.alpha = alpha
        self.beta = beta
        self.layer = layer
        self.train_acc = 0
        self.test_acc = 0
        self.acFunc = activeFunc
        self.weight = []
        if self.layer > len(self.k) or self.layer > len(self.pilk):
            print("the k list is too small! check the k list")
            sys.exit()

    def activeFunction(self, tempH, func='sig'):
        switch = {
            'sig': lambda x: 1 / (1 + np.exp(-x)),
            'sin': lambda x: np.sin(x),
            'srelu': lambda x: np.log(1 + np.exp(x)),
            'tanh': lambda x: np.tanh(x),
            'swish':lambda x: x/(1 + np.exp(-x)),
            'relu' : lambda x: np.max((0, x)),
        }
        fun = switch.get(func)
        return fun(tempH)

    def autoEncoder(self, input_X, layer):
        t1 = time.time()
        U, s, transV = np.linalg.svd(input_X, full_matrices=0)
        print("the ", layer, " layer SVD matrix shape:", "U:", U.shape, "s:", s.shape, "V:", transV.shape)  #(784, 784) (784,) (784, 60000)
        dim_x = input_X.shape[1]
        rank_x = np.sum(s > 1e-3)
        print("the ", layer, " layer, dim_x:", dim_x, " rank_x:", rank_x)
        S = np.zeros((U.shape[1], transV.shape[0]))
        S[:s.shape[0], :s.shape[0]] = np.diag(s)
        V = transV.T
        transU = U.T
        S[S != 0] = 1 / S[S != 0]
        if rank_x < dim_x :
            p = rank_x + self.alpha*(dim_x - rank_x)
        else:
            p = self.beta*dim_x
        print("the ", layer, " layer, cut p:", int(p))
        transU = transU[:, 0:int(p)]
        print("the ", layer, " layer psedoinverse matrix shape:", "U:", transU.shape, "S:", S.shape, "V:", V.shape) #(705, 784) (784, 784) (784, 784)
        # W_e = V.dot(S).dot(transU)
        input_H = U.dot(transU)

        H = self.activeFunction(input_H, self.acFunc)
        invH = np.linalg.inv(H.T.dot(H) + np.eye(H.shape[1]) * self.k[layer])
        W_d = invH.dot(H.T).dot(input_X)
        t2 = time.time()
        print("the ", layer, " layer train time cost:%.2f" %(t2 - t1))
        return W_d.T

    def fit(self, X, y, one_hot=1):
        t1 = time.time()
        m, n = X.shape
        split = int(5*m/6)
        train_X = X[: split]
        train_y = y[: split]
        valid_X = X[split:]
        valid_y = y[split:]
        if one_hot:
            train_y = tools.to_categorical(train_y)
            valid_y = tools.to_categorical(valid_y)
        for i in range(self.layer):
            w = self.autoEncoder(train_X, i)
            self.weight.append(w)
            train_H = self.activeFunction(train_X.dot(w), self.acFunc)
            valid_H = self.activeFunction(valid_X.dot(w), self.acFunc)
            H = train_H
            invH = np.linalg.inv(H.T.dot(H) + np.eye(H.shape[1]) * self.k[i]) # recompute W_d
            W_d = invH.dot(H.T).dot(train_X)
            O = H.dot(W_d)
            meanSquareError = mean_squared_error(train_X, O)
            print("the ", i, " layer meanSquareError:%.2f" % meanSquareError)
            u,s,v = np.linalg.svd(H, full_matrices=0)
            print("H usv")
            lossF = u.dot(u.T) - np.ones((H.shape[0], H.shape[0]))
            print("compute loosF")
            error = np.linalg.norm(lossF)
            print("compute norm")
            print("the ", i, " layer lossError:%.2f" % error)
            self.PIL_classifier(train_H, train_y, valid_H, valid_y, i)
            train_X = train_H
            valid_X = valid_H
        t2 = time.time()
        print("fit cost time :%.2f" %(t2 - t1))

    def extractFeature(self, input_X):
        feature = input_X
        len = self.weight.__len__()
        for i in range(len):
            feature = self.activeFunction(feature.dot(self.weight[i]), self.acFunc)
        return feature

    def predict(self, train_X, train_y, test_X, test_y):
        from sklearn.metrics import accuracy_score
        train_feature = self.extractFeature(train_X)
        test_feature = self.extractFeature(test_X)

        model = self.regression_classifier(train_feature, train_y)
        train_predict = model.predict(train_feature)
        self.train_acc = accuracy_score(train_predict, train_y)*100
        print("Accuracy of train data set: %.2f" %self.train_acc, "%")
        test_predict = model.predict(test_feature)
        self.test_acc = accuracy_score(test_predict, test_y)*100
        print("Accuracy of test data set: %.2f" %self.test_acc, "%")

    def PIL_classifier(self, train_X, train_y, test_X, test_y, layer):
        from sklearn.metrics import accuracy_score
        invH = np.linalg.inv(train_X.T.dot(train_X) + np.eye(train_X.shape[1]) * self.pilk[layer])  # recompute W_d
        pred_W = invH.dot(train_X.T).dot(train_y)
        train_predict = self.deal_onehot(train_X.dot(pred_W))
        test_predict = self.deal_onehot(test_X.dot(pred_W))
        self.train_acc = accuracy_score(train_predict, train_y) * 100
        print("Accuracy of train data set: %.2f" % self.train_acc, "%")
        self.test_acc = accuracy_score(test_predict, test_y) * 100
        print("Accuracy of test data set: %.2f" % self.test_acc, "%")

    def deal_onehot(self, matrix):
        onehot = self.activeFunction(matrix)
        m, n = matrix.shape
        for row in onehot:
            max = np.max(row)
            for i in range(n):
                if max == row[i]:
                    row[i] = 1
                    max = 1000
                else:
                    row[i] = 0
        return onehot


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



