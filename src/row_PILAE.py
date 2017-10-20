import numpy as np
import math
import time
from sklearn import preprocessing
import src.tools as tools


class row_PILAE(object):
    def __init__(self, k=0.5, alpha = 0.8, beta=0.9, activeFunc='tanh'):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.layer = 0
        self.train_acc = 0
        self.test_acc = 0
        self.acFunc = activeFunc
        self.weight = []

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
            # 'max' : lambda x,p : np.max(0, x),
        }
        fun = switch.get(func)
        return fun(tempH, p)

    def autoEncoder(self, input_X, layer):
        self.layer = layer
        t1 = time.time()
        U, s, V = np.linalg.svd(input_X, full_matrices=0)
        print("the ", layer, " layer SVD matricx shape:", "U:", U.shape, "s:", s.shape, "V:", V.shape)  #(784, 784) (784,) (784, 60000)
        dim_x = input_X.shape[1]
        rank_x = np.sum(s > 1e-3)
        print("the ", layer, " layer, dim_x:", dim_x, " rank_x:", rank_x)
        S = np.zeros((U.shape[1], V.shape[0]))
        S[:s.shape[0], :s.shape[0]] = np.diag(s)
        V = V.T
        U = U.T
        S[S != 0] = 1 / S[S != 0]
        if rank_x < dim_x :
            p = rank_x + self.alpha*(dim_x-rank_x)
            # p = self.beta*dim_x
        else:
            p = self.beta*dim_x
        # p = self.beta*dim_x
        print("the ", layer, " layer, cut p:", int(p))
        U = U[:, 0:int(p)]
        print("the ", layer, " layer pseduinverse matricx shape:", "U:", U.shape, "S:", S.shape, "V:", V.shape) #(705, 784) (784, 784) (784, 784)
        W_e = V.dot(S).dot(U)

        H = self.activeFunction(input_X.dot(W_e), self.acFunc)
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
            H = H/(H.max() - H.min())
            O = H.dot(w.T)
            square = np.linalg.norm(X - O)
            meanSquareError = square**2/X.size
            X = H
            print("the ", i, " layer meanSquareError:%.2f" %meanSquareError)

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
        # tools.save_pickle(train_feature, "../data/one_train_feature.plk")
        # tools.save_pickle(test_feature, "../data/one_test_feature.plk")
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



