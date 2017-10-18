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

def hog_ex(arr):
    hog_feature = np.empty(shape=[0, 0])
    len = 0
    for i in hog_descriptor:
        h = hog(arr, i["orientations"], i["block"], i["cell"])
        hog_feature = np.append(hog_feature, h)
    # print(len)
    return hog_feature

def extract_featuer(X, type):
    input_X = X
    hog = hog_ex(input_X[0])
    feature = np.empty(hog.shape)
    i = 0
    for arr in input_X:
        hog = hog_ex(arr)
        print(i)
        i += 1
        feature = np.append(feature, hog, axis=0)
    tools.save_pickle(feature, "../data/mnist_hog_feature_"+type+"9.plk")
    print(feature.shape)

