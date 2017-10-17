import numpy as np
import time

class col_PILAE(object):
    def __init__(self, lamda=0.05, beta=0.9, activeFunc='sig'):
        self.lamda = lamda
        self.beta = beta
        self.acFunc = activeFunc
        self.weight = []

    def f(self, tempH, func='sig', p=1e-5):
        def relu(x,p):
            x[x<0]=0
            return x
        switch ={
            'sig' : lambda x,p:1/(1+np.exp(-p*x)),
            'sin' : lambda x,p:np.sin(x),
            'relu' : relu,
            'srelu' : lambda x,p : np.log(1+np.exp(x)),
            'tanh' : lambda x,p : np.tanh(p*x),
            # 'max' : lambda x,p : np.max(0, x),
        }
        fun = switch.get(func)
        return fun(tempH, p)

    def autoEncoder(self, input_X, layer):
        t1 = time.time()
        U, s, V = np.linalg.svd(input_X, full_matrices=0)
        print("the ", layer, " layer SVD matricx shape:", "U:", U.shape, "s:", s.shape, "V:", V.shape)  #(784, 784) (784,) (784, 60000)
        dim_x = input_X.shape[0]
        rank_x = np.sum(s > 1e-3)
        print("the", layer, "layer, dim_x:", dim_x, " rank_x:", rank_x)
        S = np.zeros((U.shape[1], V.shape[0]))
        S[:s.shape[0], :s.shape[0]] = np.diag(s)
        V = V.T
        U = U.T
        S[S != 0] = 1 / S[S != 0]
        p = self.beta*dim_x
        print("the ", layer, " layer, cut p:", p)
        V = V[0:int(p), :]
        print("the ", layer, " layer pseduinverse matricx shape:", "U:", U.shape, "S:", S.shape, "V:", V.shape) #(705, 784) (784, 784) (784, 784)
        W_e = V.dot(S).dot(U)

        H = self.f(W_e.dot(input_X), self.acFunc)
        invH = np.linalg.inv(H.dot(H.T) + np.eye(H.shape[0]) * self.lamda)
        W_d = input_X.dot(H.T).dot(invH)
        t2 = time.time()
        print("the ", layer, " layer train time cost:", t2 - t1)
        return W_d.T

    def fit(self, train_X, layer):
        t1 = time.time()
        X = train_X
        for i in range(layer - 1):
            w = self.autoEncoder(X, i)
            self.weight.append(w)
            H = self.f(w.dot(X), self.acFunc)
            O = w.T.dot(H)
            square = np.linalg.norm(X - O)
            meanSquareError = square**2/X.size
            X = H
            print("the ", i, " layer meanSquareError:", meanSquareError)
        t2 = time.time()
        print("fit cost time:", t2 - t1)

    def extractFeature(self, input_X):
        feature = input_X
        len = self.weight.__len__()
        for i in range(len):
            feature = self.f(self.weight[i].dot(feature), self.acFunc)
        return feature

    def predict(self, train_X, train_y, test_X, test_y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        train_feature = self.extractFeature(train_X)
        train_feature = train_feature.T
        test_feature = self.extractFeature(test_X)
        test_feature = test_feature.T
        reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")
        reg.fit(train_feature, train_y)
        train_predict = reg.predict(train_feature)
        print("Accuracy of train data set: %f" %accuracy_score(train_predict, train_y))
        test_predict = reg.predict(test_feature)
        print("Accuracy of train data set: %f" % accuracy_score(test_predict, test_y))
