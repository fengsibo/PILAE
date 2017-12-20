import sys
import os
workpath = os.path.abspath("..")
sys.path.append(workpath)
import src.incremental_PILAE as ipilae
import src.tools as tools
import collections
import src.Hog as hg
from sklearn import preprocessing
import numpy as np
import time
import multiprocessing
import csv
num = 0


(X_train, y_train), (X_test, y_test) = tools.load_npz("../dataset/mnist/mnist.npz")
X_train = X_train.reshape(-1, 784).astype('float32')/255
X_test = X_test.reshape(-1, 784).astype('float32')/255



hyperParam = ipilae.HParams(batch_size=128,
                     num_classes=10,
                     ae_k=[0.78, 0.3],
                     pil_k=[0.01, 0.01],
                     ae_p=[740, 720],
                     pil_p=[134, 110],
                     ae_layers=2,
                     pil_layers=2,
                     acFunc='sig'
                     )

ipilae = ipilae.PILAE(X_train, y_train, hyperParam)



