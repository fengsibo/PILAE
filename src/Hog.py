import src.tools as tools
from skimage.feature import hog
import numpy as np

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
]

def hog_ex(arr, num):
    hog_feature = np.empty(shape=[0, 0])
    len = 0
    for i in range(num):
        h = hog(arr, hog_descriptor[i]["orientations"], hog_descriptor[i]["block"], hog_descriptor[i]["cell"])
        hog_feature = np.append(hog_feature, h)
    # print(len)
    return hog_feature

def extract_featuer(X, type, num):
    input_X = X
    hog = hog_ex(input_X[0], num)
    hog = hog.reshape((hog.shape[0], 1))
    feature = np.empty(hog.shape)
    i = 0
    for arr in input_X:
        hog = hog_ex(arr, num)
        hog = hog.reshape((hog.shape[0], 1))
        print(i)
        i += 1
        feature = np.append(feature, hog, axis=1)
        print(feature.shape)
    tools.save_pickle(feature, "../data/mnist_hog_feature_"+type+"_"+str(num)+".plk")
    print(feature.shape)

