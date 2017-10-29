import pickle
import numpy as np
import matplotlib as plt

def save_pickle(file, filesavepath):
    filepath = open(filesavepath, "wb")
    pickle.dump(file, filepath)
    filepath.close()
    print("save "+filesavepath+" suc!")

def load_pickle(picklepath):
    file = open(picklepath, "rb")
    data = pickle.load(file)
    file.close()
    return data

def load_npz(path):
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

def load_pickle_cifar10(filepath):
    import pickle
    with open(filepath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X = dict[b'data']
        X_train = np.array(X).reshape(-1, 3, 32, 32)
        y = dict[b'labels']
    return np.array(X), np.array(y)

def load_cifar10(cifar10_dir):
    trainpath = cifar10_dir + '/' + "data_batch_"
    testpath = cifar10_dir + '/' + "test_batch"
    X_test, y_test = load_pickle_cifar10(testpath)
    for i in range(1, 5):
        if i == 1:
            X_train, y_train = load_pickle_cifar10(trainpath + str(i))
        else:
            X, y = load_pickle_cifar10(trainpath + str(i))
            X_train = np.concatenate((X_train, X))
            y_train = np.concatenate((y_train, y))
    return (X_train, y_train), (X_test, y_test)


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def draw_line_chart12(csvfile, savename, x_name, y_name_1, y_name_2,):
    import matplotlib.pyplot as plt
    import pandas as pd
    csv = pd.read_csv(csvfile)
    x = csv[x_name]
    y1 = csv[y_name_1]
    y2 = csv[y_name_2]
    plt.figure()
    plt.plot(x, y1,  marker='o')
    plt.plot(x, y2,  marker='^')
    plt.xlabel('number of maps')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.savefig('../eps/'+savename)
    plt.show()

def draw_line_chart11(csvfile, savename, x_name, y_name):
    import matplotlib.pyplot as plt
    import pandas as pd
    csv = pd.read_csv(csvfile)
    x = csv[x_name]
    y = csv[y_name]
    plt.figure()
    plt.plot(x, y,  marker='o')
    plt.xlabel('number of maps')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.savefig('../eps/'+savename)
    plt.show()

def draw_line_chart14(csvfile, savename, x_name, y_name1, y_name2):
    import matplotlib.pyplot as plt
    import pandas as pd
    csv = pd.read_csv(csvfile)
    x = csv[x_name]
    y1 = csv[y_name1]
    y2 = csv[y_name2]
    csv2 = pd.read_csv("../log/mnist_mapid_acc10.csv")
    y3 = csv2[y_name1]
    y4 = csv2[y_name2]

    plt.figure()
    plt.plot(x, y1, marker=".")
    plt.plot(x, y2, marker=".")
    plt.plot(x, y3, marker=".")
    plt.plot(x, y4, marker='.')

    plt.xlabel('number of maps')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.savefig('../eps/'+savename)
    plt.show()

def draw_3D(csvfile, savename, x_name, y_name, z_name1, z_name2):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd
    from scipy.interpolate import griddata
    csv = pd.read_csv(csvfile)
    x = csv[x_name]
    y = csv[y_name]
    z1 = csv[z_name1]
    z2 = csv[z_name2]
    # xx, yy = np.meshgrid(x, y)
    # zz = griddata(x, y, z, xx, yy)
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z1)
    ax.scatter(x, y, z2)
    # ax.plot_trisurf(x, y, z)
    ax.set_xlabel('k')
    ax.set_ylabel('alpha')
    ax.set_zlabel('acc')

    plt.savefig('../eps/'+savename)
    plt.show()

#
# path = "../log/mnist_mapid_acc1.csv"
# draw_line_chart14(path, "mnist_hog_map_acc.eps", 'map', 'train_acc', 'test_acc')
# draw_3D(path, 'mnist_k_alpha_acc.eps', 'k', 'alpha', 'train_acc', 'test_acc')

