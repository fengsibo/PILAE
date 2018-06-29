# -*- coding: utf-8 -*-
import sys
import os, time
import datetime
workpath = os.path.abspath("..")
sys.path.append(workpath)
from src.pilae import PILAE
import src.tools as tools
import numpy as np
import matplotlib.pyplot as plt

data_dict = {'mnist.npz': 784,
             'fashionmnist.npz': 784,
             'cifar10.npz': 1024,
             'cifar10RGB.npz': 3072}

DATASET = 'mnist.npz'  # 更改数据集只需要替换这里

## 这个是自己写的读数据方法，知道保证读出来的数据是numpy格式的二维数据就可以
(X_train, y_train), (X_test, y_test) = tools.load_npz("../dataset/" + DATASET)


X_train = X_train.reshape(-1, data_dict[DATASET]).astype('float64') / 255.
X_test = X_test.reshape(-1, data_dict[DATASET]).astype('float64') / 255.
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

X_train, y_train, X_test, y_test = tools.split_dataset(X[:40000], y[:40000], 0.7)

# 实验 找给定数据集（秩一定）,测试误差最低时的隐层神经元个数
rank = np.linalg.matrix_rank(X_train)
list_train_acc = []
list_test_acc = []
list_num_hidden_units = []
for p in range(100, 40000, 200):
    pilae = PILAE(pilae_p=[p],
                  pil_p=[],
                  ae_k_list=[0.7],
                  pil_k=0.0,
                  acFunc='sig')
    pilae.train_pilae(X_train, y_train)
    pilae.classifier(X_train, y_train, X_test, y_test)
    list_train_acc.append(pilae.pil_train_acc)
    list_test_acc.append(pilae.pil_test_acc)
    list_num_hidden_units.append(p)

# rank = 500
# list_train_acc = [0.3, 0.5, 0.7]
# list_test_acc = [0.2, 0.3, 0.5]
# list_num_hidden_units = [10, 20, 30]
fig_save_path = 'save_fig'
tools.create_dir(fig_save_path)
test_acc_max_value = max(list_test_acc)[0]
test_acc_max_index = list_num_hidden_units[list_test_acc.index(test_acc_max_value)]
print(test_acc_max_index, test_acc_max_value)
plt.figure()
plt.plot(list_num_hidden_units, list_train_acc, label='train_acc')
plt.plot(list_num_hidden_units, list_test_acc, label='test_acc')
plt.xlabel("hidden units")
plt.ylabel("acc")
plt.plot(test_acc_max_index, test_acc_max_value, marker='^', color='red')

plt.text(test_acc_max_index - 5, test_acc_max_value, "{}:{}, \n{},{:.4f}".format("rank", rank, "hidden_unit", test_acc_max_value))
plt.legend()
plt.savefig(os.path.join(fig_save_path, "{}_{}_{}.png".format(DATASET, rank, test_acc_max_index)))
plt.show()
plt.close()

tools.write_log("log.txt", "{} {} {}".format(DATASET, rank, test_acc_max_index))

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
