from reward_collision_network import CollisionNetwork
import reward_data_manager
from reward_data_manager import get_train_and_test_datasets, get_batch_and_labels, get_image_cache
import time
import datetime
import numpy as np
import os
import yaml
import tensorflow as tf


class CollisionModel:

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

        self.network = CollisionNetwork(config, self.model_dir)
        self.net_output = self.network.status_softmax_logits
        self.status_input = tf.placeholder(tf.int32, [None, ])
        self.prediction = tf.argmax(tf.nn.softmax(self.net_output), axis=1, output_type=tf.int32)

        self.global_step = 0
        self.global_step_var = tf.Variable(0, trainable=False)

        self.loss = self.init_loss()
        self.test_measures = self.add_test_measures()
        self.optimizer = self.init_optimizer()

        with open(os.path.join(self.model_dir, 'config.yml'), 'w') as fd:
            yaml.dump(config, fd)

        self.train_board = self.TensorBoard(tensorboard_dir, 'train_' + model_name, self.train_summaries)
        self.test_board = self.TensorBoard(tensorboard_dir, 'test_' + model_name, self.test_summaries)

    def predict(self, data_batch, session):
        feed = self.network.make_feed(*data_batch)
        return session.run([self.prediction], feed)[0]

    def init_loss(self):
        status_loss_scale = config['reward']['cross_entropy_coefficient']
        status_loss = status_loss_scale * \
                           tf.losses.sparse_softmax_cross_entropy(labels=self.status_input - 1, logits=self.net_output)
        status_loss_summary = tf.summary.scalar('Status_Loss', status_loss)

        regularization_loss = tf.losses.get_regularization_loss()
        regularization_loss_summary = tf.summary.scalar('Regularization_Loss', regularization_loss)

        total_loss = status_loss + regularization_loss
        total_loss_summary = tf.summary.scalar('Total_Loss', total_loss)

        self.train_summaries += [status_loss_summary, regularization_loss_summary, total_loss_summary]
        self.test_summaries += [status_loss_summary, regularization_loss_summary, total_loss_summary]

        return total_loss

    def add_test_measures(self):
        labels = self.status_input - 1
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, self.prediction), tf.float32))
        TP = tf.count_nonzero((self.prediction) * (labels), dtype=tf.float32) + tf.Variable(0.001)
        TN = tf.count_nonzero((self.prediction - 1) * (labels - 1), dtype=tf.float32) + tf.Variable(0.001)
        FP = tf.count_nonzero(self.prediction * (labels - 1), dtype=tf.float32) + tf.Variable(0.001)
        FN = tf.count_nonzero((self.prediction - 1) * labels, dtype=tf.float32) + tf.Variable(0.001)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy_summary = tf.summary.scalar('Accuracy', accuracy)
        recall_summary = tf.summary.scalar('Recall', recall)
        precision_summary = tf.summary.scalar('Precision', precision)
        self.test_summaries += [accuracy_summary, recall_summary, precision_summary]
        return [accuracy, recall, precision]

    def init_optimizer(self):
        initial_learn_rate = self.config['reward']['initial_learn_rate']
        decrease_learn_rate_after = self.config['reward']['decrease_learn_rate_after']
        learn_rate_decrease_rate = self.config['reward']['learn_rate_decrease_rate']

        learning_rate = tf.train.exponential_decay(initial_learn_rate,
                                                   self.global_step,
                                                   decrease_learn_rate_after,
                                                   learn_rate_decrease_rate,
                                                   staircase=True)
        self.train_summaries.append(tf.summary.scalar('Learn_Rate', learning_rate))

        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(self.loss, tf.trainable_variables()))
        initial_gradients_norm = tf.global_norm(gradients)
        gradient_limit = config['reward']['gradient_limit']
        if gradient_limit > 0.0:
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_limit, use_norm=initial_gradients_norm)
        clipped_gradients_norm = tf.global_norm(gradients)
        initial_gradients_norm_summary = tf.summary.scalar('Gradients_Norm_Initial', initial_gradients_norm)
        clipped_gradients_norm_summary = tf.summary.scalar('Gradients_Norm_Clipped', clipped_gradients_norm)
        self.train_summaries += [initial_gradients_norm_summary, clipped_gradients_norm_summary]
        self.test_summaries += [initial_gradients_norm_summary, clipped_gradients_norm_summary]

        return optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step_var)

    def _train_batch(self, train_batch, train_status_batch, session):
        batch_start_joints, batch_actions, batch_images = train_batch
        train_feed = self.network.make_feed(batch_start_joints, batch_actions, batch_images)
        train_feed[self.status_input] = np.array(train_status_batch)
        train_summary, self.global_step, _ = session.run(
            [self.train_board.summaries, self.global_step_var, self.optimizer],
            train_feed)
        self.train_board.writer.add_summary(train_summary, self.global_step)

    def _test_batch(self, test_batch, test_status_batch, session):
        batch_start_joints, batch_actions, batch_images = test_batch
        test_feed = self.network.make_feed(batch_start_joints, batch_actions, batch_images)
        test_feed[self.status_input] = np.array(test_status_batch)
        test_summary = session.run(
            [self.test_board.summaries] + self.test_measures,
            test_feed)[0]
        self.test_board.writer.add_summary(test_summary, self.global_step)
        self.test_board.writer.flush()

    def train(self, train_data, test_data, image_cache, session):
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        total_train_batches = 0
        for epoch in range(self.epochs):

            train_batch_count = 1
            for train_batch in train_data:

                train_batch, train_status_batch = get_batch_and_labels(train_batch, image_cache)
                # assert if train_status contains goal status
                assert(np.all(np.array(train_status_batch) != reward_data_manager.GOAL_STATUS))

                self._train_batch(train_batch, train_status_batch, session)
                print("Finished epoch %d/%d batch %d/%d" % (epoch+1, self.epochs, train_batch_count, total_train_batches))
                train_batch_count += 1
            total_train_batches = train_batch_count
            self.train_board.writer.flush()

            test_batch = next(test_data.__iter__()) # random test batch
            test_batch, test_status_batch = get_batch_and_labels(test_batch, image_cache)
            self._test_batch(test_batch, test_status_batch, session)

            # save the model
            if epoch == self.epochs - 1 or epoch % self.save_every_epochs == self.save_every_epochs - 1:
                self.network.saver.save(session, self.model_dir, global_step=self.global_step)

            print('done epoch {} of {}, global step {}'.format(epoch, self.epochs, self.global_step))

    class TensorBoard:

        def __init__(self, tensorboard_path, board_name, summaries):
            self.writer = tf.summary.FileWriter(os.path.join(tensorboard_path, board_name))
            self.summaries = tf.summary.merge(summaries)


if __name__ == '__main__':
    # read the config
    config_path = os.path.join(os.getcwd(), 'data/config/reward_config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        print('------------ Config ------------')
        print(yaml.dump(config))

    model_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    models_base_dir = os.path.join('data', 'reward', 'model')
    collision_model = CollisionModel(model_name, config, models_base_dir, tensorboard_dir=models_base_dir)

    train_data, test_data = get_train_and_test_datasets(config)
    image_cache = get_image_cache(config)

    gpu_usage = config['general']['gpu_usage']
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage))
    with tf.Session(config=session_config) as session:
        collision_model.train(train_data, test_data, image_cache, session)
    train_data.stop()
    test_data.stop()
