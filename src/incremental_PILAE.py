
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


class PILAE(object):

    def __init__(self,HiddernNeurons,batchsize,k,para,actFun):
        self.HiddernNeurons = HiddernNeurons          # 隐层神经元
        self.batchsize = batchsize            # 批次大小
        self.k = k             # 误差
        self.para = para            # sigmoid p值
        self.actFun = actFun            # 激活函数
        self.layers = len(HiddernNeurons)   #隐层层数
        self.w=list(range(self.layers))        #权重
        self.preiS=list(range(self.layers))        #is
        self.InputWeight=list(range(self.layers))       #输入权重

    def makebatches(self, data):
        '''
        将样本分批次
        '''
        randomorder = random.sample(range(len(data)), len(data))
        numbatches = int(len(data) / self.batchsize)
        for batch in range(numbatches):
            order = randomorder[batch * self.batchsize:(batch + 1) * self.batchsize]
            yield data[order, :]

    def train(self, data):
        '''
        训练样本
        data 输入矩阵 换一个名字 input_data
        '''
        hidden_weight = []
        hidden_layer = []
        error_i = []
        y_out = []
        for numbatch,delta_X in enumerate(self.makebatches(data)):
            for i in range(self.layers):
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

                if numbatch ==0:
                    H = delta_X.dot(self.InputWeight[i])
                    delta_H = ActivationFunc(H, self.actFun[i], self.para[i])
                    iS = np.linalg.inv(delta_H.T.dot(delta_H) + np.eye(delta_H.shape[1]) * self.k[i])
                    OW = iS.dot(delta_H.T).dot(delta_X)
                else:
                    H = delta_X.dot(self.w[i].T)
                    delta_H = ActivationFunc(H, self.actFun[i], self.para[i])
                    iC = np.linalg.inv(np.eye(delta_H.shape[0]) + delta_H.dot(self.preiS[i]).dot(delta_H.T))
                    iS = self.preiS[i] - self.preiS[i].dot(delta_H.T).dot(iC).dot(delta_H).dot(self.preiS[i])
                    alpha = np.eye(delta_H.shape[1]) - self.preiS[i].dot(delta_H.T).dot(iC).dot(delta_H)
                    belta = self.preiS[i].dot(
                        np.eye(delta_H.shape[1]) - delta_H.T.dot(iC).dot(delta_H).dot(self.preiS[i])).dot(
                        delta_H.T).dot(delta_X)
                    OW = alpha.dot(self.w[i]) + belta
                hidden_layer.append(delta_H)
                delta_X0 = delta_H.dot(self.InputWeight[i].T)
                error = np.linalg.norm(delta_X0 - delta_X)
                print(error)
                y_out.append(delta_X0)
                error_i.append(error)
                self.w[i] = OW
                self.preiS[i]=iS

                tempH = delta_X.dot(OW.T)
                delta_X = ActivationFunc(tempH,self.actFun[i],self.para[i])
                delta_X = preprocessing.scale(delta_X.T)
                delta_X = delta_X.T


    def feature_extrackted(self,x):
        '''
        自编码器的特征提取
        '''
        feature = x
        for i in range(self.layers):
            feature = feature.dot(self.w[i].T)
            feature = ActivationFunc(feature, self.actFun[i], self.para[i])
        return feature


def  ActivationFunc(tempH, ActivationFunction, p):
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
    fun = switch.get(ActivationFunction)
    return fun(tempH,p)

def save_pickle(file, filesavepath):
    filepath = open(filesavepath, "wb")
    pickle.dump(file, filepath)
    filepath.close()
    print("save suc!")

def load_pickle(picklepath):
    file = open(picklepath, "rb")
    data = pickle.load(file)
    file.close()
    return data

def load_data(path='../data/mnist.npz'):
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':


    batchsize = 2000
    HiddernNeurons = [600]
    k = [1e-7, 1e-4, 1e-5]
    para = [1.5, 1.5, 1]
    actFun = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh']
    for i in range(0, 10):
        # HiddernNeurons.append(700 - i * 50)
        k.append(1e-3)
        para.append(1.5)
        actFun.append('tanh')



# mnist data++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    (X_train, y_train), (X_test, y_test) = load_data("../data/mnist.npz")

    data = X_train
    X_train = X_train.reshape(-1, 784).astype('float32')
    # X_train = preprocessing.scale(X_train)
    X_test = X_test.reshape(-1, 784).astype('float32')
    # X_test = preprocessing.scale(X_test)
    X_train /= 255
    X_train -= np.mean(X_train)
    # X_train /= np.std(X_train, axis=0)
    #
    X_test = X_test.reshape(-1, 784).astype('float32')
    X_test /= 255
    X_test -= np.mean(X_test)



    t1 = time.time()
    pilae = PILAE(HiddernNeurons, batchsize, k, para, actFun)  # 初始化PILAE模型
    pilae.train(X_train)  # 训练模型
    t2 = time.time()
    print("time cost of training PILAE: %.4f" % (t2 - t1))

    train_feature = pilae.feature_extrackted(X_train)  # 提取特征训练集
    test_feature = pilae.feature_extrackted(X_test)  # 提取特征测试集

    # 特征分类
    reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")

    reg.fit(train_feature, y_train)
    # np.savetxt('coef_softmax_sklearn.txt', reg.coef_, fmt='%.6f')  # Save coefficients to a text file
    ptrain = reg.predict(train_feature)
    print("Accuracy of train set: %f" % accuracy_score(ptrain, y_train))

    # reg.fit(testfeatureofdate, Label2)
    ptest = reg.predict(test_feature)
    print("Accuracy of test set: %f" % accuracy_score(ptest, y_test))







