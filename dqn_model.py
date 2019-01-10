import tensorflow as tf


class DqnModel:
    def __init__(self, prefix):
        self.prefix = '{}_dqn'.format(prefix)

    def predict(self, workspace_image, reuse_flag):
        conv1 = tf.layers.conv2d(workspace_image, 32, 8, 4, padding='same', activation=tf.nn.relu, use_bias=True,
                                 name='{}_conv1'.format(self.prefix), reuse=reuse_flag)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, padding='same', activation=tf.nn.relu, use_bias=True,
                                 name='{}_conv2'.format(self.prefix), reuse=reuse_flag)
        # conv3 = tf.layers.conv2d(conv2, 64, 3, 1, padding='same', activation=tf.nn.relu, use_bias=True)
        # flat = tf.layers.flatten(conv3)
        flat = tf.layers.flatten(conv2, name='{}_flat'.format(self.prefix))
        dense1 = tf.layers.dense(flat, 512, activation=tf.nn.relu, name='{}_dense1'.format(self.prefix),
                                 reuse=reuse_flag)
        dense2 = tf.layers.dense(dense1, 512, activation=None, name='{}_dense2'.format(self.prefix), reuse=reuse_flag)
        return dense2

