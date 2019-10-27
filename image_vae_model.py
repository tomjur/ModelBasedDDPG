from reward_data_manager import get_image_cache
import time
import datetime
import numpy as np
import os
import yaml
import tensorflow as tf
from vae_network import VAENetwork


class VAEModel:

    def __init__(self, model_name, config, models_base_dir, tensorboard_dir):

        self.model_name = model_name
        self.config = config

        self.model_dir = os.path.join(models_base_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.train_summaries = []
        self.test_summaries = []

        self.epochs = config['general']['epochs']
        self.save_every_epochs = config['general']['save_every_epochs']
        self.train_vae = config['reward']['train_vae']

        inputs_example = tf.placeholder(tf.float32, (None, 55, 111), name='example')
        self.network = VAENetwork(config, self.model_dir, inputs_example.shape)

        self.global_step = 0
        self.global_step_var = tf.Variable(0, trainable=False)

        self.loss = self.init_loss()
        self.optimizer = self.init_optimizer()

        with open(os.path.join(self.model_dir, 'config.yml'), 'w') as fd:
            yaml.dump(config, fd)

        self.train_board = self.TensorBoard(tensorboard_dir, 'train_' + model_name, self.train_summaries)
        self.test_board = self.TensorBoard(tensorboard_dir, 'test_' + model_name, self.test_summaries)

    def load(self, session):
        self.network.load_weights(session)

    def make_feed(self, data_batch):
        return self.network.make_feed(*data_batch)

    def predict(self, data_batch, session):
        feed = self.make_feed(data_batch)
        return session.run([self.prediction], feed)[0]

    def init_loss(self):
        status_loss_scale = self.config['reward']['cross_entropy_coefficient']
        img_loss, latent_loss, total_loss = self.network.get_loss()

        image_loss_summary = tf.summary.scalar('Image_Loss', img_loss)
        latent_loss_summary = tf.summary.scalar('Latent_Loss', latent_loss)

        regularization_loss = tf.losses.get_regularization_loss()
        regularization_loss_summary = tf.summary.scalar('Regularization_Loss', regularization_loss)

        # total_loss = total_loss + regularization_loss
        total_loss_summary = tf.summary.scalar('Total_Loss', total_loss)

        self.train_summaries += [image_loss_summary, latent_loss_summary, regularization_loss_summary, total_loss_summary]
        self.test_summaries += [image_loss_summary, latent_loss_summary, regularization_loss_summary, total_loss_summary]

        return total_loss

    def init_optimizer(self):
        initial_learn_rate = self.config['reward']['initial_learn_rate']
        decrease_learn_rate_after = self.config['reward']['decrease_learn_rate_after']
        learn_rate_decrease_rate = self.config['reward']['learn_rate_decrease_rate']

        learning_rate = tf.train.exponential_decay(initial_learn_rate,
                                                   self.global_step_var,
                                                   decrease_learn_rate_after,
                                                   learn_rate_decrease_rate,
                                                   staircase=True)
        self.train_summaries.append(tf.summary.scalar('Learn_Rate', learning_rate))

        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(self.loss, tf.trainable_variables()))
        initial_gradients_norm = tf.global_norm(gradients)
        gradient_limit = self.config['reward']['gradient_limit']
        if gradient_limit > 0.0:
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_limit, use_norm=initial_gradients_norm)
        clipped_gradients_norm = tf.global_norm(gradients)
        initial_gradients_norm_summary = tf.summary.scalar('Gradients_Norm_Initial', initial_gradients_norm)
        clipped_gradients_norm_summary = tf.summary.scalar('Gradients_Norm_Clipped', clipped_gradients_norm)
        self.train_summaries += [initial_gradients_norm_summary, clipped_gradients_norm_summary]
        self.test_summaries += [initial_gradients_norm_summary, clipped_gradients_norm_summary]

        return optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step_var)

    def _train_batch(self, train_batch, session):
        train_feed = {self.network.workspace_image_inputs: train_batch}
        train_summary, self.global_step, img_loss, _ = session.run(
            [self.train_board.summaries, self.global_step_var, self.network.encoded, self.optimizer],
            train_feed)
        # print(img_loss)
        self.train_board.writer.add_summary(train_summary, self.global_step)

    def _test_batch(self, test_batch, session):
        test_feed = {self.network.workspace_image_inputs: test_batch}
        test_summary = session.run(
            [self.test_board.summaries],
            test_feed)[0]
        self.test_board.writer.add_summary(test_summary, self.global_step)
        self.test_board.writer.flush()

    def train(self, train_data, test_data, session):
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        test_every_batches = self.config['reward']['test_every_batches']

        total_train_batches = 0
        for epoch in range(self.epochs):

            train_batch_count = 1
            for train_batch in train_data:
                self._train_batch(train_batch, session)
                print("Finished epoch %d/%d batch %d/%d" % (epoch+1, self.epochs, train_batch_count, total_train_batches))
                train_batch_count += 1

                if train_batch_count % test_every_batches == 0:
                    test_batch = next(test_data.__iter__())  # random test batch
                    self._test_batch(test_batch, session)
                    # save the model
                    # self.network.save_weights(session, self.global_step)

            total_train_batches = train_batch_count - 1
            self.train_board.writer.flush()

            test_batch = next(test_data.__iter__()) # random test batch
            self._test_batch(test_batch, session)

            # save the model
            # if epoch == self.epochs - 1 or epoch % self.save_every_epochs == self.save_every_epochs - 1:
            #     self.network.save_weights(session, self.global_step)

            print('done epoch {} of {}, global step {}'.format(epoch, self.epochs, self.global_step))

    class TensorBoard:

        def __init__(self, tensorboard_path, board_name, summaries):
            self.writer = tf.summary.FileWriter(os.path.join(tensorboard_path, board_name))
            self.summaries = tf.summary.merge(summaries)


def count_weights():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)

if __name__ == '__main__':
    # read the config
    config_path = os.path.join(os.getcwd(), 'data/config/reward_config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        print('------------ Config ------------')
        print(yaml.dump(config))

    model_name = "vae" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

    image_cache = get_image_cache(config)
    batch_size = 1
    images_data = [image.np_array for image in image_cache.items.values()]
    images_batch_data = [images_data[i:i+batch_size] for i in range(0, len(images_data), batch_size)]

    train_data_count = int(len(images_batch_data) * 0.8)
    train_data = images_batch_data[:train_data_count]
    test_data = images_batch_data[train_data_count:]

    models_base_dir = os.path.join('data', 'reward', 'model')
    vae_model = VAEModel(model_name, config, models_base_dir, tensorboard_dir=models_base_dir)



    gpu_usage = config['general']['gpu_usage']
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage))
    with tf.Session(config=session_config) as session:
        count_weights()
        vae_model.train(train_data, test_data, session)
