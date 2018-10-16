import tensorflow as tf


def get_activation(activation):
    if activation == 'relu':
        return tf.nn.relu
    if activation == 'tanh':
        return tf.nn.tanh
    if activation == 'elu':
        return tf.nn.elu
    return None