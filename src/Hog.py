import src.tools as tools
from skimage.feature import hog
import numpy as np
import time

hog_descriptor = [
    {'orientations': 9, 'block': (8, 8), 'cell': (3, 3)},
    {'orientations': 9, 'block': (8, 8), 'cell': (2, 2)},
    {'orientations': 9, 'block': (8, 8), 'cell': (1, 1)},
    {'orientations': 12, 'block': (8, 8), 'cell': (3, 3)},
    {'orientations': 12, 'block': (8, 8), 'cell': (2, 2)},
    {'orientations': 12, 'block': (8, 8), 'cell': (1, 1)},
    {'orientations': 15, 'block': (8, 8), 'cell': (3, 3)},
    {'orientations': 15, 'block': (8, 8), 'cell': (2, 2)},
    {'orientations': 15, 'block': (8, 8), 'cell': (1, 1)},
    {'orientations': 18, 'block': (8, 8), 'cell': (3, 3)},
    {'orientations': 18, 'block': (8, 8), 'cell': (2, 2)},
    {'orientations': 18, 'block': (8, 8), 'cell': (1, 1)},
    {'orientations': 24, 'block': (8, 8), 'cell': (3, 3)},
    {'orientations': 24, 'block': (8, 8), 'cell': (2, 1)},
    {'orientations': 24, 'block': (8, 8), 'cell': (1, 1)},
    {'orientations': 9, 'block': (12, 12), 'cell': (3, 3)},
    {'orientations': 9, 'block': (12, 12), 'cell': (2, 2)},
    {'orientations': 9, 'block': (12, 12), 'cell': (1, 1)},
    {'orientations': 12, 'block': (12, 12), 'cell': (2, 2)},
    {'orientations': 12, 'block': (12, 12), 'cell': (1, 1)},
    {'orientations': 15, 'block': (12, 12), 'cell': (2, 2)},
    {'orientations': 15, 'block': (12, 12), 'cell': (1, 1)},
    {'orientations': 18, 'block': (12, 12), 'cell': (2, 2)},
    {'orientations': 18, 'block': (12, 12), 'cell': (1, 1)},
    {'orientations': 24, 'block': (12, 12), 'cell': (2, 2)},
    {'orientations': 24, 'block': (12, 12), 'cell': (1, 1)},
]

def extract_hog_featuer(X, type, num):
    input_X = X
    from skimage.feature import hog
    hog_f = hog(input_X[0], hog_descriptor[num]["orientations"], hog_descriptor[num]["block"], hog_descriptor[num]["cell"])
    hog_f = hog_f.reshape((hog_f.shape[0], 1))
    feature = np.empty(hog_f.shape)
    i = 0
    for arr in input_X:
        hog_f = hog(arr, hog_descriptor[num]["orientations"], hog_descriptor[num]["block"], hog_descriptor[num]["cell"])
        hog_f = hog_f.reshape((hog_f.shape[0], 1))
        print(i)
        i += 1
        feature = np.append(feature, hog_f, axis=1)
        print(feature.shape)
    feature = feature.T[1:, :]
    tools.save_pickle(feature, "../data/"+type+"_"+str(num)+".plk")
    print(feature.shape)

def load_hog(num):
    for i in range(num):
        hog_train_plk_path = "../data/fashion_mnist/mnist_hog_feature_train_" + str(i) + ".plk"
        hog_test_plk_path = "../data/fashion_mnist/mnist_hog_feature_test_" + str(i) + ".plk"
        if i == 0:
            X_train = tools.load_pickle(hog_train_plk_path)
            X_test = tools.load_pickle(hog_test_plk_path)
        else:
            m_train = tools.load_pickle(hog_train_plk_path)
            m_test = tools.load_pickle(hog_test_plk_path)
            X_train = np.concatenate((X_train, m_train), axis=1)
            X_test = np.concatenate((X_test, m_test), axis=1)
    X_train *= 100
    X_test *= 100
    return X_train, X_test

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = tools.load_MNISTData()
    # hog_f = hog(X_train[0], 24, (12, 12), (2, 2))
    # print(hog_f.__len__())

    len = hog_descriptor.__len__()
    i = 0
    while i < len:
        extract_hog_featuer(X_train, "mnist/mnist_hog_feature_train", i)
        extract_hog_featuer(X_test, "mnist/mnist_hog_feature_test", i)
        i += 1

    # X_train, y_train = tools.load_fashionMNIST()
    # X_test, y_test = tools.load_fashionMNIST(kind='t10k')
    # X_train = X_train.reshape((60000, 28, 28))
    # X_test = X_test.reshape((10000, 28, 28))
    # print(X_train.shape, X_test.shape)
    #
    # #
    # len = hog_descriptor.__len__()
    # i = 0
    # while i < len:
    #     extract_hog_featuer(X_train, "fashion_mnist/fashion_mnist_hog_feature_train", i)
    #     extract_hog_featuer(X_test, "fashion_mnist/fashion_mnist_hog_feature_test", i)
    #     i += 1


