import tensorflow as tf


class DqnModel:
    def predict(self, workspace_image):
        conv1 = tf.layers.conv2d(workspace_image, 32, 8, 4, padding='same', activation=tf.nn.relu, use_bias=True)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, padding='same', activation=tf.nn.relu, use_bias=True)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, padding='same', activation=tf.nn.relu, use_bias=True)
        flat = tf.layers.flatten(conv3)
        dense1 = tf.layers.dense(flat, 512, activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, 512, activation=None)
        return dense2

