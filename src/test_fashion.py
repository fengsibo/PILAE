import src.row_PILAE as rp
import src.tools as tools

X_train, y_train = tools.load_fashionMNIST()
X_test, y_test = tools.load_fashionMNIST(kind='t10k')

X_train = X_train.reshape(-1, 784).astype('float32')
X_test = X_test.reshape(-1, 784).astype('float32')

X_train /= 255
X_test /= 255

pilae = rp.row_PILAE(k=1.5, alpha=0.8, beta=0.95, activeFunc='sig')
pilae.fit(X_train, layer=4)
pilae.predict_softmax(X_train, y_train, X_test, y_test)