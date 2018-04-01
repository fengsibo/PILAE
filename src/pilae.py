# -*- coding: utf-8 -*-
import sys
import os
import itertools

workpath = os.path.abspath("..")
sys.path.append(workpath)
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
import src.tools as tools


class PILAE(object):
    def __init__(self, pilae_p, pil_p, ae_k_list, pil_k, alpha=0.9, acFunc='sig'):
        self.pilae_p = pilae_p
        self.pil_p = pil_p
        self.ae_k_list = ae_k_list
        self.pil_k = pil_k
        self.alpha = alpha
        self.acFunc = acFunc

        self.num_ae_layers = len(pilae_p)
        self.pil_layers = len(pil_p)

        if len(ae_k_list) < self.num_ae_layers:
            print("ae_k_list的长度和层数不符，检查ae_k_list参数")

        self.error = []
        self.loss = []
        self.pilae_weight = []
        self.pil_weight = []
        self.pil_train_acc = []
        self.pil_test_acc = []
        self.softmax_train_acc = []
        self.softmax_test_acc = []
        self.dim = []
        self.rank = []
        self.layers = []


    # 自编码结构
    # @param
    # input_X:训练数据，要求是行向量(矩阵的行是特征)
    # layer:第几层自编码(从0计数)
    # return:返回编码器权重
    def autoEncoder(self, input_X, layer_th):
        t1 = time.time()
        U, s, transV = np.linalg.svd(input_X,
                                     full_matrices=0)  # thinSVD分解 full_matrices=0: thinSVD, full_matrices=1: full_SVD
        dim_x = input_X.shape[1]  # 获取输入矩阵的维数(输入数据的第二个维度即维数)
        self.dim.append(dim_x)  # 存储每层数据的维数到self.dim
        rank_x = np.linalg.matrix_rank(input_X)  # 获取输入矩阵的秩
        self.rank.append(rank_x)  # 储存秩到self.rank
        transU = U.T

        ## ① WangKe2017年SMC论文 根据经验公式计算隐层神经元个数的方法
        # if rank_x < dim_x:
        #     p = rank_x + self.alpha[layer] * (dim_x - rank_x)
        # else:
        #     p = self.alpha[layer] * dim_x

        ## ② 自定义隐层单元个数方法
        p = self.pilae_p[layer_th]  # 从self.pilae_p列表中读取第layer_th层的隐层单元个数
        print("\033[0;36;40m[BASE INFO]\033[0m the {} layer message:".format(layer_th))
        print("the dim_x:{}, rank_x:{}, cut_p:{}".format(dim_x, rank_x, int(p)))

        cutU = transU[:, 0:int(p)]  # 矩阵截断
        input_H = U.dot(cutU)
        H = self.activeFunction(input_H, self.acFunc)

        invH = np.linalg.inv(H.T.dot(H) + np.eye(H.shape[1]) * self.ae_k_list[layer_th])
        W_d = invH.dot(H.T).dot(input_X)
        print("train time cost {:.2f}s".format(time.time() - t1))
        return W_d.T #返回解码器权重的转置(权重捆绑)

    # 训练PILAE
    # @param
    # X: 输入训练数据，行向量
    # y: 数据标签
    # return: None(无返回值) 调用分类器
    def train_pilae(self, X, y):
        t1 = time.time()
        train_X = X

        # 根据self.ae_layers设置的层数 使用for循环进行逐层训练
        for i in range(self.num_ae_layers):
            w = self.autoEncoder(train_X, i) # 训练该层的自编码器编码权重
            self.layers.append(i)
            self.pilae_weight.append(w) # 保存权重
            H = self.activeFunction(train_X.dot(w), self.acFunc) # 计算自编码器的编码(特征) 也就是下一个自编码器的输入
            train_X = H
        print("train auto-encoders cost time {:.5f}s".format(time.time() - t1))
        self.classifier(X, y) # 训练完成后调用分类函数

    # 分类方法
    # @param
    # X: 输入训练数据，行向量
    # y: 数据标签
    # return: None(无返回值) 调用分类器
    def classifier(self, X, y):
        train_X = X
        # 首先数据与训练好的权重相乘 计算出自编码器的特征(自编码器的编码)
        # 说明: 之前是使用训练集训练权重，然后使用测试集与训练权重计算得到特征
        # 但是2017年末 Guo老师说要训练测试数据放到一起在自编码器中 输出特征之后再划分训练集 测试集 目的是使数据同分布
        for i in range(self.num_ae_layers):
            print("\033[0;31;40m[CLASSIFIER INFO]\033[0m the {} layer result".format(i))
            H = self.activeFunction(train_X.dot(self.pilae_weight[i]), self.acFunc) #根据输入 加载保存的权重 逐层计算自编码器提取的特征
            train_X = H
            loss = self.compute_loss(H)  # 计算每层的loss
            print("loss: {}".format(loss))
            shuffle_X, shuffle_y = self.__random_shuffle(H, y) # 随机打乱数据集 相当于随机抽样
            (train_H, train_y), (valid_H, valid_y) = self.__split_dataset(shuffle_X, shuffle_y) # 划分数据集为训练集合测试集
            self.predict_pil(train_H, train_y, valid_H, valid_y, i) # 使用pil分类器预测
            self.predict_softmax(train_H, train_y, valid_H, valid_y, i) # 使用softmax分类器
        # self.model_analysis()
        self.result_figure() # 会把结果画在一张图上面

    # 用来画图 对模型进行分析 这个可以改写里面的函数 保存图片用于论文用图
    def model_analysis(self):
        self.draw_one_chart(self.layers, self.loss)
        self.draw_two_chart(self.layers, self.pil_train_acc, self.pil_test_acc)
        self.draw_two_chart(self.layers, self.softmax_train_acc, self.softmax_test_acc)
        self.draw_two_chart(self.layers, self.dim, self.rank, y_label='rank')

    # 使用伪逆学习训练多层网络 用来分类
    def train_pil(self, train_X, train_y):
        X = train_X
        y = tools.to_categorical(train_y) # 转换成one_hot编码
        # 根据self.pil_layers设置的层数 使用for循环进行逐层训练
        for i in range(self.pil_layers):
            pinvX = np.linalg.pinv(X)
            pinvX = pinvX[:, 0: int(self.pil_p[i])]
            self.pil_weight.append(pinvX)
            tempH = X.dot(pinvX)
            X = self.activeFunction(tempH, self.acFunc)
        invH = np.linalg.inv(X.T.dot(X) + np.eye(X.shape[1]) * self.pil_k)
        pred_W = invH.dot(X.T).dot(y)
        self.pil_weight.append(pred_W)

    # 使用训练好的mlp 进行分类
    def predict_pil(self, X_train, y_train, X_test, y_test, layer_th=None):
        train_X = X_train
        train_y = y_train
        test_X = X_test
        test_y = y_test

        # 调用这个predict函数 在predict之前训练
        # 这样的写法有点不符合规范 但是也还说的过去
        self.train_pil(train_X, train_y)
        train_y_true = tools.to_categorical(train_y)
        test_y_true = tools.to_categorical(test_y)
        # 逐层读取保存好的权重 得到分类结果
        for i in range(self.pil_layers):
            train_X = self.activeFunction(train_X.dot(self.pil_weight[i]), self.acFunc)
            test_X = self.activeFunction(test_X.dot(self.pil_weight[i]), self.acFunc)
        train_y_predict = self.get_onehot(train_X.dot(self.pil_weight[-1]))
        test_y_predict = self.get_onehot(test_X.dot(self.pil_weight[-1]))
        self.pil_weight = []

        train_acc = accuracy_score(train_y_true, train_y_predict) * 100
        test_acc = accuracy_score(test_y_true, test_y_predict) * 100
        # 将结果保存到self.pil_train_acc中
        self.pil_train_acc.append(train_acc)
        self.pil_test_acc.append(test_acc)
        print("PIL classifier layer {}:".format(self.pil_layers))
        print("PIL Train accuracy: {}% | Test accuracy: {}%".format(train_acc, test_acc))
        # test_recall_score = recall_score(train_y_true, train_y_predict, average='micro') * 100
        # test_f1_score = f1_score(train_y_true, train_y_predict, average='micro') * 100
        # test_classification_report = classification_report(train_y_true, train_y_predict)
        # print("PIL test recall: {} and f1_score: {}".format(test_recall_score, test_f1_score))
        # print(self.test_classification_report)
        cnf_matrix = confusion_matrix(np.argmax(test_y_predict, axis=1), np.argmax(test_y_true, axis=1))
        # 打印混淆矩阵
        # print('confusion_matrix', cnf_matrix)

    # 训练逻辑回归
    def train_softmax(self, train_X, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=200)
        model.fit(train_X, train_y)
        return model

    # 测试逻辑回归
    def predict_softmax(self, train_X, train_y, test_X, test_y, layer_th):
        model = self.train_softmax(train_X, train_y)
        train_y_predict = model.predict(train_X)
        test_y_predict = model.predict(test_X)
        train_acc = accuracy_score(train_y, train_y_predict) * 100
        test_acc = accuracy_score(test_y, test_y_predict) * 100
        self.softmax_train_acc.append(train_acc)
        self.softmax_test_acc.append(test_acc)
        print("Softmax Train accuracy:{}% | Test accuracy:{}%".format(train_acc, test_acc))
        # test_recall_score = recall_score(test_y, test_y_predict, average='micro') * 100
        # test_f1_score = f1_score(test_y, test_y_predict, average='micro') * 100
        # test_classification_report = classification_report(test_y, test_y_predict)
        # print("test recall:{}, f1_score:{}".format(test_recall_score, test_f1_score))
        # print(self.test_classification_report)

    def compute_reconstruct_error(self, y_true, y_predict):
        pass

    def compute_loss(self, H):
        pinvH = np.linalg.pinv(H)
        sq = H.dot(pinvH) - np.eye(H.shape[0])
        loss = np.linalg.norm(sq)
        self.loss.append(loss)
        return loss

    def activeFunction(self, tempH, func='sig'):
        switch = {
            'sig': lambda x: 1 / (1 + np.exp(-x)),
            'sin': lambda x: np.sin(x),
            'srelu': lambda x: np.log(1 + np.exp(x)),
            'tanh': lambda x: np.tanh(x),
            'swish': lambda x: x / (1 + np.exp(-x)),
            'relu': lambda x: np.maximum(0, x),
        }
        fun = switch.get(func)
        return fun(tempH)

    def get_onehot(self, matrix):
        m, n = matrix.shape
        one_hot = np.zeros((m, n))
        index = np.argmax(matrix, axis=1)
        for i in range(m):
            one_hot[i][index[i]] = 1
        return one_hot

    # 绘制混淆矩阵的API函数 由sklearn包提供
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # 随机打乱数据集 相当于随机抽样
    def __random_shuffle(self, X, y):
        m, n = X.shape
        index = [i for i in range(m)]
        import random as rd
        rd.shuffle(index)
        X = X[index]
        y = y[index]
        return X, y

    def __split_dataset(self, X, y):
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

    def result_figure(self):
        plt.figure()

        plt.subplot(221)
        plt.plot(self.layers, self.loss, marker='^')
        plt.xlabel("autoencoder layers")
        plt.ylabel("model loss")

        plt.subplot(222)
        plt.plot(self.layers, self.pil_train_acc, marker='^', label='train')
        plt.plot(self.layers, self.pil_test_acc, marker='o', label='test')
        plt.xlabel("autoencoder layers")
        plt.ylabel("pil classifier accuracy")

        print(self.layers, self.pil_train_acc)
        print(self.layers, self.softmax_train_acc)
        plt.subplot(223)
        plt.plot(self.layers, self.softmax_train_acc, marker='^', label='train')
        plt.plot(self.layers, self.softmax_test_acc, marker='o', label='test')
        plt.xlabel("autoencoder layers")
        plt.ylabel("softmax classifier accuracy")

        plt.subplot(224)
        plt.plot(self.layers, self.dim, marker='^', label='dimension')
        plt.plot(self.layers, self.rank, marker='o', label='rank')
        plt.xlabel("autoencoder layers")
        plt.ylabel("dim & rank")

        plt.subplots_adjust(wspace=0.5, hspace=0.6)
        plt.show()
        plt.close()

    def draw_one_chart(self, x, y, x_label='model layers', y_label='model loss'):
        plt.figure()
        plt.plot(x, y, marker='^')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        plt.close()

    def draw_two_chart(self, x, y1, y2, x_label='model layers', y_label='accuracy(%)'):
        plt.figure()
        plt.plot(x, y1, marker='^')
        plt.plot(x, y2, marker='o')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        plt.close()
