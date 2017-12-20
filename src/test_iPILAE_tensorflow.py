import sys
sys.path.append("..")
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import src.incremental_PILAE_tensoflow as ipt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 128
num_classes  = 10
epochs = 100

hyperParam = ipt.HParams(batch_size=batch_size,
                     num_classes=num_classes,
                     ae_k=[0.78, 0.3],
                     pil_k=[0.01, 0.01],
                     ae_p=[740, 720],
                     pil_p=[134, 110],
                     ae_layers=2,
                     pil_layers=2,
                     acFunc='sig'
                     )

X = tf.placeholder(tf.float32, (None, 28, 28, 1))
y = tf.placeholder(tf.float32, (None, num_classes))

model = ipt.PILAE(X, y, hyperParam)

# num_batch = mnist.count()//batch_size
for iter in range(epochs):


