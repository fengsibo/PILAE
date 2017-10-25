import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import time

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data pre-processing
x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)
t1 = time.time()
# in order to plot in a 2D figure
encoding_dim = 400

# this is our input placeholder
input_img = Input(shape=(784,))

# encoder layers
encoded = Dense(712, activation='relu')(input_img)
encoded = Dense(624, activation='relu')(encoded)
encoded = Dense(532, activation='relu')(encoded)
encoder_output = Dense(encoding_dim, name="encoder")(encoded)

# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)

# construct the encoder model for plotting
encoder = Model(input=input_img, output=encoder_output)
# encoder.save("../model/keras/autoencoder.h5")


# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)




encoder_trainX = encoder.predict(x_train)
encoder_testX = encoder.predict(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=200)
model.fit(encoder_trainX, y_train)
train_predict = model.predict(encoder_trainX)
train_acc = accuracy_score(train_predict, y_train)*100
print("Accuracy of train data set: %.2f" %train_acc, "%")
test_predict = model.predict(encoder_testX)
test_acc = accuracy_score(test_predict, y_test)*100
print("Accuracy of test data set: %.2f" %test_acc, "%")

t2 = time.time()
cost_time = t2 - t1
print("Total cost time: %.2f" %cost_time)