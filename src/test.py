import src.row_PILAE as rp
import src.tools as tools

(X_train, y_train), (X_test, y_test) = tools.load_MNISTData()
X_train = X_train.reshape(-1, 784).astype('float32')
X_mean = X_train.mean(axis=1)
X_std = X_train.std(axis=1)
# X_train = (X_train - X_mean)/X_std
X_train /= 255
# X_train = preprocessing.scale(X_train, axis=1)

X_test = X_test.reshape(-1, 784).astype('float32')
X_test /= 255
# X_test = preprocessing.scale(X_test, axis=1)

pilae =  rp.row_PILAE()
pilae.fit(X_train, layer=2)
pilae.predict(X_train, y_train, X_test, y_test)