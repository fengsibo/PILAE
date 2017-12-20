import sys
sys.path.append("..")
import collections
import tensorflow as tf

HParams = collections.namedtuple('HParams',
                                 'batch_size, num_classes, ae_k, pil_k, '
                                 'ae_p, pil_p, ae_layers, pil_layers, '
                                 'acFunc')

class PILAE(object):
    def __init__(self, X, y, hps):
        self.X = X
        self.y = y
        self.hps = hps

    def fit(self):
        s, u, v = tf.svd(self.X)
        input_H = tf.matmul(u, tf.transpose(u, perm=[1, 0]))
