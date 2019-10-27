import tensorflow as tf
from coordnet_model import coord_conv


class DqnModel:
    def __init__(self, prefix, config):
        self.prefix = '{}_dqn'.format(prefix)
        self.config = config
        self.use_coordnet = self.config['network']['use_coordnet']

    def predict(self, workspace_image, reuse_flag):
        if self.use_coordnet:
            workspace_image = coord_conv(55, 111, False, workspace_image, 32, 8, 4, padding='same',
                                         activation=tf.nn.relu, use_bias=True, name='{}_conv1'.format(self.prefix),
                                         _reuse=reuse_flag)

        conv2 = tf.layers.conv2d(workspace_image, 64, 4, 2, padding='same', activation=tf.nn.relu, use_bias=True,
                                 name='{}_conv2'.format(self.prefix), reuse=reuse_flag)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, padding='same', activation=tf.nn.relu, use_bias=True)

        flat = tf.layers.flatten(conv3, name='{}_flat'.format(self.prefix))
        dense1 = tf.layers.dense(flat, 512, activation=tf.nn.relu, name='{}_dense1'.format(self.prefix),
                                 reuse=reuse_flag)
        dense2 = tf.layers.dense(dense1, 512, activation=tf.nn.relu, name='{}_dense2'.format(self.prefix),
                                 reuse=reuse_flag)
        dense3 = tf.layers.dense(dense2, 512, activation=tf.nn.relu, name='{}_dense3'.format(self.prefix),
                                 reuse=reuse_flag)
        return dense3
