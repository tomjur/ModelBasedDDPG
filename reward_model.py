from reward_collision_model import CollisionModel
import reward_data_manager
from reward_data_manager import get_train_and_test_datasets, get_batch_and_labels, get_image_cache
import time
import datetime
import numpy as np
import os
import yaml
import tensorflow as tf


class RewardModel:

    def __init__(self, model_name, config, models_base_dir, tensorboard_dir, trained_collision_model):

        self.model_name = model_name
        self.config = config

        self.model_dir = os.path.join(models_base_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.train_summaries = []
        self.test_summaries = []

        self.epochs = config['general']['epochs']
        self.save_every_epochs = config['general']['save_every_epochs']

        self.collision_model = trained_collision_model
        self.collision_prob = self.collision_model.collision_prob

        self.next_joints_inputs = tf.placeholder(tf.float32, (None, 4), name='next_joints_inputs')
        self.goal_joints_inputs = tf.placeholder(tf.float32, (None, 4), name='goal_joints_inputs')

        self.states_probabilities = self.create_reward_logic(self.next_joints_inputs, self.goal_joints_inputs, self.collision_prob)

        self.status_input = tf.placeholder(tf.int32, [None, ])
        self.prediction = tf.argmax(self.states_probabilities, axis=1, output_type=tf.int32)

        self.test_measures = self.add_test_measures()

        with open(os.path.join(self.model_dir, 'config.yml'), 'w') as fd:
            yaml.dump(config, fd)

        self.test_board = self.TensorBoard(tensorboard_dir, 'test_' + model_name, self.test_summaries)

    def predict(self, data_batch, session):
        feed = self.collision_model.make_feed(data_batch)
        return session.run([self.prediction], feed)[0]

    def create_reward_logic(self, next_joints, goal_joints, collision_prob):
        # close-to-goal sensitivity
        goal_sensitivity = self.config['openrave_rl']['goal_sensitivity']

        # margin parameter in which the model transitions from 100% close to goal to 0% close to goal
        alpha = 0.4

        # distance to goal
        delta = tf.norm(next_joints - goal_joints, ord='euclidean')

        # This expression is 0 if the distance between current and goal is below \epsilon.
        a = tf.maximum(delta - goal_sensitivity, 0)

        # When delta < epsilon: this is 0
        # When delta > epsilon + alpha: this is \alpha,
        # and in-between it is linear
        b = tf.minimum(a, alpha)

        is_close_to_goal = 1 - (b / alpha)

        p_goal = (1 - collision_prob) * is_close_to_goal
        p_free_space = (1 - collision_prob) * (1 - is_close_to_goal)

        return tf.concat((p_free_space, collision_prob, p_goal), axis=1)

    def add_test_measures(self):
        labels = self.status_input - 1
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, self.prediction), tf.float32))
        # # TP = tf.count_nonzero((self.prediction) * (labels), dtype=tf.float32) + tf.Variable(0.001)
        # # TN = tf.count_nonzero((self.prediction - 1) * (labels - 1), dtype=tf.float32) + tf.Variable(0.001)
        # # FP = tf.count_nonzero(self.prediction * (labels - 1), dtype=tf.float32) + tf.Variable(0.001)
        # # FN = tf.count_nonzero((self.prediction - 1) * labels, dtype=tf.float32) + tf.Variable(0.001)
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        accuracy_summary = tf.summary.scalar('Accuracy', accuracy)
        # recall_summary = tf.summary.scalar('Recall', recall)
        # precision_summary = tf.summary.scalar('Precision', precision)
        self.test_summaries += [accuracy_summary]
        return [accuracy]

    def _test_batch(self, test_batch, test_status_batch, session):
        batch_start_joints, batch_actions, batch_images, batch_next_joints, batch_goal_joints = test_batch
        test_feed = self.collision_model.make_feed((batch_start_joints, batch_actions, batch_images))
        test_feed[self.status_input] = np.array(test_status_batch)
        test_feed[self.next_joints_inputs] = batch_next_joints
        test_feed[self.goal_joints_inputs] = batch_goal_joints
        test_summary = session.run(
            [self.test_board.summaries] + self.test_measures,
            test_feed)[0]
        results = session.run(
            [self.test_measures, self.prediction, self.status_input - 1],
            test_feed)
        print(results)
        self.test_board.writer.add_summary(test_summary)
        self.test_board.writer.flush()

    def test(self, test_data, image_cache, session):
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        self.collision_model.load(session)

        test_batch = next(test_data.__iter__()) # random test batch
        test_batch, test_status_batch = get_batch_and_labels(test_batch, image_cache)
        self._test_batch(test_batch, test_status_batch, session)

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

    models_base_dir = os.path.join('data', 'reward', 'model')
    collision_model_name = "collision_simple_trained"
    collision_model = CollisionModel(collision_model_name, config, models_base_dir, tensorboard_dir=models_base_dir)

    model_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    reward_model = RewardModel(model_name, config, models_base_dir, models_base_dir, collision_model)

    train_data, test_data = get_train_and_test_datasets(config, is_collision_model=False)
    image_cache = get_image_cache(config)

    gpu_usage = config['general']['gpu_usage']
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage))
    with tf.Session(config=session_config) as session:
        reward_model.test(test_data, image_cache, session)
    train_data.stop()
    test_data.stop()
