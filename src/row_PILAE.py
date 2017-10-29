import numpy as np
import math
import time
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import src.tools as tools


class PILAE(object):
    def __init__(self, k=0.5, alpha = 0.8, beta=0.9, activeFunc='tanh'):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.layer = 0
        self.train_acc = 0
        self.test_acc = 0
        self.acFunc = activeFunc
        self.weight = []

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
        self.layer = layer
        t1 = time.time()
        U, s, transV = np.linalg.svd(input_X, full_matrices=0)
        print("the ", layer, " layer SVD matrix shape:", "U:", U.shape, "s:", s.shape, "V:", transV.shape)  #(784, 784) (784,) (784, 60000)
        dim_x = input_X.shape[1]
        rank_x = np.sum(s > 1e-3)
        print("the ", layer, " layer, dim_x:", dim_x, " rank_x:", rank_x)
        S = np.zeros((U.shape[1], V.shape[0]))
        S[:s.shape[0], :s.shape[0]] = np.diag(s)
        V = transV.T
        transU = U.T
        S[S != 0] = 1 / S[S != 0]
        if rank_x < dim_x :
            p = rank_x + self.alpha*(dim_x - rank_x)
            # p = self.beta*dim_x
        else:
            p = self.beta*dim_x
        # p = self.beta*dim_x
        print("the ", layer, " layer, cut p:", int(p))
        transU = transU[:, 0:int(p)]
        print("the ", layer, " layer psedoinverse matrix shape:", "U:", transU.shape, "S:", S.shape, "V:", V.shape) #(705, 784) (784, 784) (784, 784)
        # W_e = V.dot(S).dot(transU)
        input_H = U.dot(transU)

        H = self.activeFunction(input_H, self.acFunc)
        invH = np.linalg.inv(H.T.dot(H) + np.eye(H.shape[1]) * self.k)
        W_d = invH.dot(H.T).dot(input_X)
        t2 = time.time()
        print("the ", layer, " layer train time cost:%.2f" %(t2 - t1))
        return W_d.T

    def fit(self, train_X, layer):
        t1 = time.time()
        X = train_X
        for i in range(layer):
            w = self.autoEncoder(X, i)
            self.weight.append(w)
            H = self.activeFunction(X.dot(w), self.acFunc)
            invH = np.linalg.inv(H.T.dot(H) + np.eye(H.shape[1]) * self.k) # recompute W_d
            W_d = invH.dot(H.T).dot(X)
            # H = H/(H.max() - H.min())
            O = H.dot(W_d)
            meanSquareError = mean_squared_error(X, O)
            print("the ", i, " layer meanSquareError:%.2f" % meanSquareError)
            lossF = H.dot(np.linalg.pinv(H)) - np.ones((H.shape[0], H.shape[0]))
            error = np.linalg.norm(lossF)
            print("the ", i, " layer lossError:%.2f" % error)
            X = H

        t2 = time.time()
        print("fit cost time :%.2f" %(t2 - t1))

    def extractFeature(self, input_X):
        feature = input_X
        len = self.weight.__len__()
        for i in range(len):
            ff = feature.dot(self.weight[i])
            w = self.weight[i]
            feature = self.activeFunction(feature.dot(self.weight[i]), self.acFunc)
            e = feature
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



