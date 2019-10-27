import tensorflow as tf
from coordnet_model import coord_conv


def get_flatten_shape(multi_dim_shape):
    flat_shape = 1
    for dim in multi_dim_shape.as_list()[1:]:
        flat_shape *= dim
    return flat_shape


class VAENetwork:
    def __init__(self, config, model_dir, input_shape):
        self.prefix = 'vae'
        self.config = config
        self.latent_dim = self.config['reward']['vae_latent_dim']
        self.input_shape = input_shape.as_list()

        assert(self.input_shape[0] is None)
        self.input_shape[0] = -1
        self.flat_input_shape = get_flatten_shape(input_shape)

        self.workspace_image_inputs = tf.placeholder(tf.float32, input_shape, name='workspace_image_inputs')
        self.images_3d = tf.expand_dims(self.workspace_image_inputs, axis=-1)

        self.encoded, self.mean, self.std_dev = self.encode(self.images_3d, reuse_flag=False)
        self.decoded = self.decode(self.encoded, reuse_flag=False)
        self.epochs = 10

    def predict(self, workspace_image, reuse_flag):
        return self.encode(workspace_image, reuse_flag)[0]

    def encode(self, workspace_image, reuse_flag):
        if self.config['reward']['use_coordnet']:
            print("Using Coordnet")
            workspace_image = coord_conv(55, 111, False, workspace_image, 32, 8, 4, padding='same', activation=tf.nn.relu, use_bias=True,
                               name='{}_conv1'.format(self.prefix), _reuse=reuse_flag)
        workspace_image = tf.Print(workspace_image, [workspace_image], message="my workspace_image:")

        conv2 = tf.layers.conv2d(workspace_image, 64, 4, 2, padding='same', activation=tf.nn.relu, use_bias=True,
                                 name='{}_conv2'.format(self.prefix), reuse=reuse_flag)
        conv2 = tf.Print(conv2, [conv2], message="my encoded_conv2:")

        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, padding='same', activation=tf.nn.relu, use_bias=True)
        conv4 = tf.layers.conv2d(conv3, 32, 8, 4, padding='same', activation=tf.nn.relu, use_bias=True)
        flat = tf.layers.flatten(conv4, name='{}_flat'.format(self.prefix))

        self.encode_last_conv_shape = conv4.shape.as_list()
        self.encode_last_conv_shape[0] = -1
        self.encode_after_conv_flat_shape = flat.shape.as_list()

        self.conv2 = conv2
        self.workspace_image = workspace_image


        dense1 = tf.layers.dense(flat, 512, activation=tf.nn.relu, name='{}_dense1'.format(self.prefix), reuse=reuse_flag)
        dense2 = tf.layers.dense(dense1, 512, activation=tf.nn.relu, name='{}_dense2'.format(self.prefix), reuse=reuse_flag)
        dense3 = tf.layers.dense(dense2, 512, activation=tf.nn.relu, name='{}_dense3'.format(self.prefix), reuse=reuse_flag)

        # Local latent variables
        mean_ = tf.layers.dense(dense3, units=self.latent_dim, name='mean')
        std_dev = tf.nn.softplus(tf.layers.dense(dense3, units=self.latent_dim), name='std_dev')  # softplus to force >0
        mean_ = tf.Print(mean_, [mean_], message="my mean_:")
        std_dev = tf.Print(std_dev, [std_dev], message="my std_dev:")
        # Reparametrization trick
        epsilon = tf.random_normal(tf.stack([tf.shape(dense3)[0], self.latent_dim]), name='epsilon')
        z = mean_ + tf.multiply(epsilon, std_dev)

        return z, mean_, std_dev

    def decode(self, encoded_layer, reuse_flag):
        decoded_dense1 = tf.layers.dense(encoded_layer, 512, activation=tf.nn.relu, name='{}_decoded_dense1'.format(self.prefix), reuse=reuse_flag)
        decoded_dense2 = tf.layers.dense(decoded_dense1, 512, activation=tf.nn.relu, name='{}_decoded_dense2'.format(self.prefix), reuse=reuse_flag)
        decoded_dense3 = tf.layers.dense(decoded_dense2, self.encode_after_conv_flat_shape[1], activation=tf.nn.relu, name='{}_decoded_dense3'.format(self.prefix), reuse=reuse_flag)
        decoded_before_conv = tf.reshape(decoded_dense3, self.encode_last_conv_shape)

        decoded_conv4 = tf.layers.conv2d_transpose(decoded_before_conv, 32, 8, 4, padding='same', activation=tf.nn.relu, use_bias=True,
                                                   name='{}_decoded_conv4'.format(self.prefix), reuse=reuse_flag)
        decoded_conv3 = tf.layers.conv2d_transpose(decoded_conv4, 64, 3, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                                                   name='{}_decoded_conv3'.format(self.prefix), reuse=reuse_flag)
        decoded_conv2 = tf.layers.conv2d_transpose(decoded_conv3, 64, 4, 2, padding='same', activation=tf.nn.relu, use_bias=True,
                                                   name='{}_decoded_conv2'.format(self.prefix), reuse=reuse_flag)
        decoded_conv2 = tf.Print(decoded_conv2, [decoded_conv2], message="my decoded_conv2:")
        img = tf.slice(decoded_conv2, [0,0,0,0], [-1, 55, 111, 1])
        img = tf.Print(img, [img], message="my decoded_img:")

        # flat = tf.layers.flatten(decoded_conv3, name='{}_decode_flat'.format(self.prefix))
        # decoded_dense4 = tf.layers.dense(sliced_image, self.flat_input_shape, activation=tf.nn.sigmoid, name='{}_decoded_dense4'.format(self.prefix), reuse=reuse_flag)
        #
        # img = tf.reshape(decoded_dense4, shape=self.input_shape)

        return img

    def get_loss(self):
        # Reshape input and output to flat vectors
        flat_output = tf.reshape(self.decoded, [-1, get_flatten_shape(self.decoded.shape)]) + 0.0001
        flat_input = tf.reshape(self.images_3d, [-1, get_flatten_shape(self.images_3d.shape)])
        flat_output = tf.Print(flat_output, [flat_output], message="my flat_output:")
        flat_input = tf.Print(flat_input, [flat_input], message="my flat_input:")
        with tf.name_scope('loss'):
            img_loss_vec = flat_input * -tf.log(flat_output) + (1 - flat_input) * -tf.log(1 - flat_output)
            img_loss_vec = tf.Print(img_loss_vec, [img_loss_vec], message="my img_loss_vec:")
            img_loss = tf.reduce_sum(img_loss_vec, 1)
            # img_loss = tf.reduce_sum(flat_input * -tf.log(flat_output) + (1 - flat_input) * -tf.log(1 - flat_output), 1)
            img_loss = tf.Print(img_loss, [img_loss], message="my img_loss:")
            latent_loss = 0.5 * tf.reduce_sum(tf.square(self.mean) + tf.square(self.std_dev) - tf.log(tf.square(self.std_dev)) - 1, 1)
            latent_loss = tf.Print(latent_loss, [latent_loss], message="my latent_loss:")
            self.total_loss = tf.reduce_mean(img_loss + latent_loss)
            self.latent_loss = tf.reduce_mean(latent_loss)
            self.img_loss = tf.reduce_mean(img_loss)

        return self.img_loss, self.latent_loss, self.total_loss



import yaml
import os
import datetime
import time
from reward_data_manager import *

if __name__ == "__main__":
    # read the config
    config_path = os.path.join(os.getcwd(), 'data/config/reward_config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        print('------------ Config ------------')
        print(yaml.dump(config))

    model_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

    image_cache = get_image_cache(config)
    batch_size = 100
    images_data = [image.np_array for image in image_cache.items.values()]
    images_batch_data = [images_data[i:i+batch_size] for i in range(0, len(images_data), batch_size)]

    train_data_count = int(len(images_batch_data) * 0.8)
    train_data = images_batch_data[:train_data_count]
    test_data = images_batch_data[train_data_count:]

    workspace_image_inputs = tf.placeholder(tf.float32, (None, 55, 111), name='workspace_image_inputs')
    model = VAENetwork(model_name, config, workspace_image_inputs.shape)
    with tf.Session() as session:
        model.train(train_data, test_data, session)

    # model = DqnModel(model_name, config, images_3d.shape)
    # z, mean_, std_dev = model.encode(images_3d, reuse_flag=False)
    # output = model.decode(z, reuse_flag=False)
    #
    # # Reshape input and output to flat vectors
    # flat_output = tf.reshape(output, [-1, get_flatten_shape(output.shape)])
    # flat_input = tf.reshape(images_3d, [-1, get_flatten_shape(images_3d.shape)])

    # with tf.name_scope('loss'):
    #     img_loss = tf.reduce_sum(flat_input * -tf.log(flat_output) + (1 - flat_input) * -tf.log(1 - flat_output), 1)
    #     latent_loss = 0.5 * tf.reduce_sum(tf.square(mean_) + tf.square(std_dev) - tf.log(tf.square(std_dev)) - 1, 1)
    #     loss = tf.reduce_mean(img_loss + latent_loss)


    x = 1

