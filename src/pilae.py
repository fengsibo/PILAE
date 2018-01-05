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
    def __init__(self, ae_k_list, pil_p, pil_k=0.03, alpha=0.9, ae_layers=1, pil_layers=1, acFunc='sig'):
        self.ae_k_list = ae_k_list
        self.pil_k = pil_k
        self.alpha = alpha
        self.pil_p = pil_p
        self.ae_layers = ae_layers
        self.pil_layers = pil_layers
        self.acFunc = acFunc
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
        for i in range(self.ae_layers):
            self.layers.append(i + 1)
        if self.ae_layers > len(self.ae_k_list):
            print("the k list is too small! check the k list")
            sys.exit()

    def autoEncoder(self, input_X, layer):
        t1 = time.time()
        U, s, transV = np.linalg.svd(input_X, full_matrices=0)
        dim_x = input_X.shape[1]
        self.dim.append(dim_x)
        rank_x = np.linalg.matrix_rank(input_X)
        self.rank.append(rank_x)
        transU = U.T
        if rank_x < dim_x:
            p = rank_x + self.alpha[layer] * (dim_x - rank_x)
        else:
            p = self.alpha[layer] * dim_x
        print("[INFO] the {} layer message:".format(layer))
        print("the dim_x:{}, rank_x:{}, cut_p:{}".format(dim_x, rank_x, int(p)))

        cutU = transU[:, 0:int(p)]
        input_H = U.dot(cutU)

        # plt.hist(input_H.flatten(), bins=24, range=(0, 0.1))

        H = self.activeFunction(input_H, self.acFunc)

        invH = np.linalg.inv(H.T.dot(H) + np.eye(H.shape[1]) * self.ae_k_list[layer])
        W_d = invH.dot(H.T).dot(input_X)
        print("train time cost {:.2f}s".format(time.time() - t1))
        return W_d.T

    def train_pilae(self, X, y):
        t1 = time.time()
        train_X = X
        train_y = y
        for i in range(self.ae_layers):
            w = self.autoEncoder(train_X, i)
            self.pilae_weight.append(w)
            H = self.activeFunction(train_X.dot(w), self.acFunc)
            train_X = H
        print("train auto-encoders cost time {:.5f}s".format(time.time() - t1))
        self.classifier(X, y)

    def classifier(self, X, y):
        train_X = X
        for i in range(self.ae_layers):
            H = self.activeFunction(train_X.dot(self.pilae_weight[i]), self.acFunc)
            train_X = H
            print("[info]====================the {} layer information=====================".format(i + 1))
            self.compute_loss(H)
            shuffle_X, shuffle_y = self.random_shuffle(H, y)
            (train_H, train_y), (valid_H, valid_y) = self.split_dataset(shuffle_X, shuffle_y)
            self.predict_pil(train_H, train_y, valid_H, valid_y, i)
            self.predict_softmax(train_H, train_y, valid_H, valid_y, i)
        self.model_analysis()

    def model_analysis(self):
        self.draw_one_chart(self.layers, self.loss)
        self.draw_two_chart(self.layers, self.pil_train_acc, self.pil_test_acc)
        self.draw_two_chart(self.layers, self.softmax_train_acc, self.softmax_test_acc)
        self.draw_two_chart(self.layers, self.dim, self.rank, y_label='rank')

    def train_pil(self, train_X, train_y):
        X = train_X
        y = tools.to_categorical(train_y)
        for i in range(self.pil_layers):
            pinvX = np.linalg.pinv(X)
            self.pil_weight.append(pinvX)
            # pinvX = pinvX[:, 0: int(self.pil_p)]
            tempH = X.dot(pinvX)
            X = self.activeFunction(tempH, self.acFunc)
        invH = np.linalg.inv(X.T.dot(X) + np.eye(X.shape[1]) * self.pil_k)  # recompute W_d
        pred_W = invH.dot(X.T).dot(y)
        self.pil_weight.append(pred_W)

    def predict_pil(self, train_X, train_y, test_X, test_y, layer):
        self.train_pil(train_X, train_y)
        train_y_true = tools.to_categorical(train_y)
        test_y_true = tools.to_categorical(test_y)
        for i in range(self.pil_layers):
            train_X = self.activeFunction(train_X.dot(self.pil_weight[i]), self.acFunc)
            test_X = self.activeFunction(test_X.dot(self.pil_weight[i]), self.acFunc)
        train_y_predict = self.get_onehot(train_X.dot(self.pil_weight[-1]))
        test_y_predict = self.get_onehot(test_X.dot(self.pil_weight[-1]))
        self.pil_weight = []

        train_acc = accuracy_score(train_y_true, train_y_predict) * 100
        test_acc = accuracy_score(test_y_true, test_y_predict) * 100
        self.pil_train_acc.append(train_acc)
        self.pil_test_acc.append(test_acc)
        print("[INFO]+==================the {} layer pil===================".format(layer + 1))
        print("PIL classifier layer {}:".format(self.pil_layers))
        print("PIL Train accuracy: {} | Test accuracy: {}".format(train_acc, test_acc))
        test_recall_score = recall_score(train_y_true, train_y_predict, average='micro') * 100
        test_f1_score = f1_score(train_y_true, train_y_predict, average='micro') * 100
        test_classification_report = classification_report(train_y_true, train_y_predict)
        print("PIL test recall: {} and f1_score: {}".format(test_recall_score, test_f1_score))
        # print(self.test_classification_report)

        cnf_matrix = confusion_matrix(np.argmax(test_y_predict, axis=1), np.argmax(test_y_true, axis=1))

    def train_softmax(self, train_X, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=200)
        model.fit(train_X, train_y)
        return model

    def predict_softmax(self, train_X, train_y, test_X, test_y, layer):
        model = self.train_softmax(train_X, train_y)
        train_y_predict = model.predict(train_X)
        test_y_predict = model.predict(test_X)
        print("==================softmax classification the {} layer=================".format(layer + 1))
        train_acc = accuracy_score(train_y, train_y_predict) * 100
        test_acc = accuracy_score(test_y, test_y_predict) * 100
        self.softmax_train_acc.append(train_acc)
        self.softmax_test_acc.append(test_acc)
        print("Train accuracy:{}% | Test accuracy:{}%".format(train_acc, test_acc))
        test_recall_score = recall_score(test_y, test_y_predict, average='micro') * 100
        test_f1_score = f1_score(test_y, test_y_predict, average='micro') * 100
        test_classification_report = classification_report(test_y, test_y_predict)
        print("test recall:{}%, f1_score:{}%".format(test_recall_score, test_f1_score))
        # print(self.test_classification_report)

    def compute_reconstruct_error(self, y_true, y_predict):
        pass

    def compute_loss(self, H):
        pinvH = np.linalg.pinv(H)
        sq = H.dot(pinvH) - np.eye(H.shape[0])
        loss = np.linalg.norm(sq)
        self.loss.append(loss)
        print("loss:{:.5f}".format(loss))

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

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
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

    def random_shuffle(self, X, y):
        m, n = X.shape
        index = [i for i in range(m)]
        import random as rd
        rd.shuffle(index)
        X = X[index]
        y = y[index]
        return X, y

    def split_dataset(self, X, y):
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
