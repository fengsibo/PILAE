import pickle
import numpy as np

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

def load_MNISTData(path='../dataset/mnist.npz'):
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)