import sys
import os
workpath = os.path.abspath("..")
sys.path.append(workpath)
import numpy as np
import math
import time
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import src.tools as tools


class PILAE(object):
    def __init__(self, ae_k, pil_k, alpha, pil_p, ae_layer=1, pil_layer=1, activeFunc='sig'):
        self.k = ae_k
        self.pilk = pil_k
        self.alpha = alpha
        self.pil_p = pil_p
        self.ae_layer = ae_layer
        self.pil_layer = pil_layer
        self.train_acc = 0
        self.test_acc = 0
        self.acFunc = activeFunc
        self.weight = []
        if self.ae_layer > len(self.k):
            print("the k list is too small! check the k list")
            sys.exit()

    def activeFunction(self, tempH, func='sig'):
        switch = {
            'sig': lambda x: 1 / (1 + np.exp(-x)),
            'sin': lambda x: np.sin(x),
            'srelu': lambda x: np.log(1 + np.exp(x)),
            'tanh': lambda x: np.tanh(x),
            'swish':lambda x: x/(1 + np.exp(-x)),
            'relu' : lambda x: np.maximum(0, x),
        }
        fun = switch.get(func)
        return fun(tempH)

    def autoEncoder(self, input_X, layer):
        """
        :param input_X:
        :param layer:
        :return:
        """
        t1 = time.time()
        U, s, transV = np.linalg.svd(input_X, full_matrices=0)
        dim_x = input_X.shape[1]
        rank_x = np.linalg.matrix_rank(input_X)
        transU = U.T
        if rank_x < dim_x :
            p = rank_x + self.alpha[layer]*(dim_x - rank_x)
        else:
            p = self.alpha[layer]*dim_x
        print("[INFO] the {} layer message:")
        print("the SVD matrix shape U:{}, s:{}, VT:{}".format(U.shape, s.shape, transV.shape))
        print("the dim_x:{}, rank_x:{}, cut_p:{}".format(dim_x, rank_x, int(p)))
        transU = transU[:, 0:int(p)]
        input_H = U.dot(transU)

        H = self.activeFunction(input_H, self.acFunc)
        invH = np.linalg.inv(H.T.dot(H) + np.eye(H.shape[1]) * self.k[layer])
        W_d = invH.dot(H.T).dot(input_X)
        t2 = time.time()
        print("the ", layer, " layer train time cost:%.2f" %(t2 - t1))
        return W_d.T

    def PIL_fit(self, train_X, train_y):
        """
        @param train_X: the train data, usually the feature in row
        @param layer: the number of pil layers
        @return: the classcification result
        """
        X = train_X
        y = train_y
        layer = self.pil_layer

        for i in range(layer):
            # U, s, transV = np.linalg.svd(X, full_matrices=0)
            # dim_x = X.shape[1]
            # rank_x = np.linalg.matrix_rank(X)
            # transU = U.T
            # transU = transU[:, 0: self.pil_p[i]]
            pinvX = np.linalg.pinv(X)
            transU = pinvX[:, 0: self.pil_p[i]]
            print(X.shape, transU.shape)
            self.weight.append(transU)
            tempH = X.dot(transU)
            X = self.activeFunction(tempH, self.acFunc)
        invH = np.linalg.inv(X.T.dot(X) + np.eye(X.shape[1]) * self.pilk)  # recompute W_d
        pred_W = invH.dot(X.T).dot(y)
        self.weight.append(pred_W)


    def fit(self, X, y):
        train_X = X
        train_y = y
        t1 = time.time()
        for i in range(self.ae_layer):
            w = self.autoEncoder(train_X, i)
            H = self.activeFunction(train_X.dot(w), self.acFunc)
            invH = np.linalg.inv(H.T.dot(H) + np.eye(H.shape[1]) * self.k[i]) # recompute W_d
            W_d = invH.dot(H.T).dot(X)
            O = H.dot(W_d)
            meanSquareError = mean_squared_error(X, O)
            print("the ", i, " layer meanSquareError:%.2f" % meanSquareError)

            # u,s,v = np.linalg.svd(H, full_matrices=0)
            # print("H usv")
            # lossF = u.dot(u.T) - np.eye((H.shape[0], H.shape[0]))
            # print("compute loosF")
            # error = np.linalg.norm(lossF)
            # print("compute norm")
            # print("the ", i, " layer lossError:%.2f" % error)

            shuffleX, shuffley = self.random_shuffle(H, train_y)
            (train_H, train_y), (valid_H, valid_y) = self.split_set(shuffleX, shuffley)
            self.PIL_classifier(train_H, train_y, valid_H, valid_y)
            self.predict(train_H, train_y, valid_H, valid_y)
            train_X = H
        t2 = time.time()
        print("fit cost time :%.2f" %(t2 - t1))

    def PIL_feedforward(self, input_X):
        predict = input_X
        len = self.weight.__len__()
        for i in range(len):
            predict = self.activeFunction(predict.dot(self.weight[i]), self.acFunc)
        return predict

    def predict(self, train_X, train_y, test_X, test_y):
        model = self.regression_classifier(train_X, train_y)
        train_predict = model.predict(train_X)
        test_predict = model.predict(test_X)
        print("==================Linear classification=================")
        self.train_acc = accuracy_score(train_predict, train_y) * 100
        self.test_acc = accuracy_score(test_predict, test_y) * 100
        print("Train accuracy:{}% | Test accuracy:{}%".format(self.train_acc, self.test_acc))
        self.test_recall_score = recall_score(test_y, test_predict, average='micro')*100
        self.test_f1_score = f1_score(test_y, test_predict, average='micro')*100
        self.test_classification_report = classification_report(test_y, test_predict)
        print("test recall{}%, f1_score:{}%".format(self.test_recall_score, self.test_f1_score))
        print(self.test_classification_report)


    def PIL_classifier(self, train_X, train_y, test_X, test_y):
        train_y = tools.to_categorical(train_y)
        test_y = tools.to_categorical(test_y)
        self.PIL_fit(train_X, train_y)
        predict_train = self.PIL_feedforward(train_X)
        predict_test = self.PIL_feedforward(test_X)
        train_predict = self.deal_onehot(predict_train)
        test_predict = self.deal_onehot(predict_test)
        self.train_acc = accuracy_score(train_predict, train_y) * 100
        self.test_acc = accuracy_score(test_predict, test_y) * 100
        print("==================pil=================")
        print("PIL classifier layer {}:".format(self.pil_layer))
        print("PIL Train accuracy: {} | Test accuracy: {}".format(self.train_acc, self.test_acc))
        self.test_recall_score = recall_score(test_y, test_predict, average='micro') * 100
        self.test_f1_score = f1_score(test_y, test_predict, average='micro') * 100
        self.test_classification_report = classification_report(test_y, test_predict)
        print("PIL test recall: {} and f1_score: {}".format(self.test_recall_score, self.test_f1_score))
        print(self.test_classification_report)

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

    def random_shuffle(self, X, y):
        m, n = X.shape
        index = [i for i in range(m)]
        import random as rd
        rd.shuffle(index)
        X = X[index]
        y = y[index]
        return X, y

    def split_set(self, X, y):
        m, n = X.shape
        split = int(5 * m / 6)
        train_H = X[:split]
        valid_H = X[split:]
        train_y = y[: split]
        valid_y = y[split:]
        return (train_H, train_y), (valid_H, valid_y)

    def regression_classifier(self, train_X, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=200)
        model.fit(train_X, train_y)
        return model

    def svm_classifier(self, train_X, train_y):
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=False)
        model.fit(train_X, train_y)
        return model

    def linearsvm_classifier(self, train_X, train_y):
        from sklearn import svm
        model = svm.LinearSVC()
        model.fit(train_X, train_y)
        return model
