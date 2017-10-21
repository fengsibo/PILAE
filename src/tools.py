import pickle
import numpy as np

def save_pickle(file, filesavepath):
    filepath = open(filesavepath, "wb")
    pickle.dump(file, filepath)
    filepath.close()
    print("save "+filesavepath+"suc!")

def load_pickle(picklepath):
    file = open(picklepath, "rb")
    data = pickle.load(file)
    file.close()
    return data

def load_MNISTData(path='../dataset/mnist/mnist.npz'):
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def load_fashionMNIST(path="../dataset/fashion_mnist", kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def load_cifar10(path):
    import pickle
    import os
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = root + file
            with open(filepath, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                X = dict[b'data']
                X_train = np.array(X).reshape(-1, 3, 32, 32)
                y = dict[b'labels']
                print(len(X[0]))
                # print(X)


# load_cifar10("../dataset/cifar-10/")