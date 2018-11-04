import numpy as np
import random
import pickle
import os
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

from modeling_utils import get_activation
from openrave_manager import OpenraveManager
from potential_point import PotentialPoint


class PreTrainedReward:

    def __init__(self, model_name, config):
        self.config = config
        self.joints_inputs = tf.placeholder(tf.float32, (None, 4), name='joints_inputs')
        self.goal_joints_inputs = tf.placeholder(tf.float32, (None, 4), name='goal_joints_inputs')
        self.workspace_image_inputs = None
        self.goal_pose_inputs = None
        if config['model']['consider_goal_pose']:
            self.goal_pose_inputs = tf.placeholder(tf.float32, (None, 2), name='goal_pose_inputs')
        self.action_inputs = tf.placeholder(tf.float32, (None, 4), name='action_inputs')
        current_variables_count = len(tf.trainable_variables())
        self.reward_prediction, self.status_softmax_logits = self._create_reward_network()
        reward_variables = tf.trainable_variables()[current_variables_count:]

        # model path to load
        self.model_name = model_name
        self.saver_dir = os.path.join('reward', 'model', model_name)
        assert os.path.exists(self.saver_dir)
        self.saver = tf.train.Saver(reward_variables, max_to_keep=4, save_relative_paths=self.saver_dir)

    def _generate_goal_features(self):
        features = [self.goal_joints_inputs]
        if self.goal_pose_inputs is not None:
            features.append(self.goal_pose_inputs)
        return tf.concat(features, axis=1)

    def _next_state_model(self):
        # next step is a deterministic computation
        action_step_size = self.config['openrave_rl']['action_step_size']
        step = self.action_inputs * action_step_size
        unclipped_result = self.joints_inputs + step
        # we initiate an openrave manager to get the robot, to get the joint bounds and the safety
        joint_safety = 0.0001
        lower_bounds = [-2.617, -1.571, -1.571, -1.745, -2.617]
        lower_bounds = [b + joint_safety for b in lower_bounds[1:]]
        upper_bounds = [-b for b in lower_bounds]

        # clip the result
        clipped_result = tf.maximum(unclipped_result, lower_bounds)
        clipped_result = tf.minimum(clipped_result, upper_bounds)
        return clipped_result, unclipped_result

    def _create_reward_network(self):
        name_prefix = 'reward'
        # get the next joints
        clipped_next_joints, unclipped_next_joints = self._next_state_model()

        # predict the transition classification
        layers = self.config['reward']['layers'] + [3]
        scale = self.config['reward']['l2_regularization_coefficient']
        current = tf.concat((clipped_next_joints, self._generate_goal_features()), axis=1)
        for i, layer_size in enumerate(layers):
            _activation = None if i == len(layers) - 1 else get_activation(self.config['reward']['activation'])
            current = tf.layers.dense(
                current, layer_size, activation=_activation, name='{}_layers_{}'.format(name_prefix, i),
                kernel_regularizer=tf_layers.l2_regularizer(scale)
            )
        softmax_logits = current
        softmax_res = tf.nn.softmax(softmax_logits)

        # get the classification reward
        classification_reward = tf.layers.dense(
                softmax_res, 1, activation=None, use_bias=False, name='{}_classification_reward'.format(name_prefix),
            )

        # get the clipping-related reward
        clipped_difference = tf.expand_dims(tf.sqrt(
                tf.reduce_sum(tf.square(unclipped_next_joints - clipped_next_joints), axis=1)
            ), axis=1)
        clipping_reward = tf.layers.dense(
            clipped_difference, 1, activation=None, use_bias=False, name='{}_clipping_weight'.format(name_prefix)
        )

        total_reward = classification_reward + clipping_reward
        return total_reward, softmax_logits

    def load_weights(self, sess):
        self.saver.restore(sess, tf.train.latest_checkpoint(self.saver_dir))

    def make_prediction(self, sess, all_start_joints, all_goal_joints, all_actions, all_goal_poses):
        feed = self.make_feed(all_start_joints, all_goal_joints, all_actions, all_goal_poses)
        return sess.run([self.reward_prediction, self.status_softmax_logits], feed)

    def make_feed(self, all_start_joints, all_goal_joints, all_actions, all_goal_poses):
        feed = {
            self.joints_inputs: all_start_joints,
            self.goal_joints_inputs: all_goal_joints,
            self.action_inputs: all_actions,
        }
        if self.goal_pose_inputs is not None:
            feed[self.goal_pose_inputs] = all_goal_poses
        return feed


def oversample_batch(data_collection, data_index, batch_size, oversample_large_magnitude=False):
    current_batch = data_collection[data_index:data_index + batch_size]
    if not oversample_large_magnitude:
        return current_batch
    rewards = [b[5] for b in current_batch]
    success_reward_indices = [i for i, r in enumerate(rewards) if r > 0.8]
    if len(success_reward_indices) < 3:
        return None
    collision_reward_indices = [i for i, r in enumerate(rewards) if r < -0.8]
    if len(collision_reward_indices) < 3:
        return None
    other_reward_indices = [i for i, r in enumerate(rewards) if np.abs(r) <= 0.8]
    assert len(success_reward_indices) + len(collision_reward_indices) + len(other_reward_indices) == len(current_batch)
    sample_size = len(other_reward_indices)
    batch_indices = other_reward_indices
    # sample_size = min(100, len(other_reward_indices))
    # batch_indices = list(np.random.choice(other_reward_indices, sample_size))
    success_super_sample = list(np.random.choice(success_reward_indices, sample_size))
    batch_indices.extend(success_super_sample)
    collision_super_sample = list(np.random.choice(collision_reward_indices, sample_size))
    batch_indices.extend(collision_super_sample)
    return [current_batch[i] for i in batch_indices]


def get_batch_and_labels(batch, openrave_manager):
    all_start_joints = []
    all_goal_joints = []
    all_actions = []
    all_rewards = []
    all_goal_poses = []
    all_status = []
    for i in range(len(batch)):
        start_joints, goal_joints, image, action, next_joints, reward, terminated, status = batch[i]
        goal_pose = openrave_manager.get_target_pose(goal_joints)
        all_start_joints.append(start_joints[1:])
        all_goal_joints.append(goal_joints[1:])
        all_actions.append(action[1:])
        all_rewards.append(reward)
        all_status.append(status)
        all_goal_poses.append(goal_pose)
    return [all_start_joints, all_goal_joints, all_actions, all_goal_poses], all_rewards, all_status


def compute_stats_single_class(real_status, real_reward, status_prediction, reward_prediction, class_indicator):
    class_indices = [i for i, s in enumerate(real_status) if s == class_indicator]
    if len(class_indices) == 0:
        accuracy = 0.0
        class_average_absolute_error = 0.0
        class_max_absolute_error = 0.0
    else:
        my_status_prediction = [status_prediction[i] for i in class_indices]
        best_label = np.argmax(my_status_prediction, axis=1)
        best_label += 1
        hit_cont = len([i for i, b in enumerate(best_label) if b == class_indicator])
        accuracy = float(hit_cont) / len(class_indices)
        difference = [np.abs(real_reward[i] - reward_prediction[i]) for i in class_indices]
        class_average_absolute_error = np.mean(difference)
        class_max_absolute_error = np.max(difference)
    return class_indices, [class_average_absolute_error, class_max_absolute_error, accuracy]


def compute_stats_per_class(real_status, real_reward, status_prediction, reward_prediction):
    goal_rewards_indices, goal_stats = compute_stats_single_class(
        real_status, real_reward, status_prediction, reward_prediction, 3)
    collision_rewards_indices, collision_stats = compute_stats_single_class(
        real_status, real_reward, status_prediction, reward_prediction, 2)
    other_rewards_indices, other_stats = compute_stats_single_class(
        real_status, real_reward, status_prediction, reward_prediction, 1)
    assert len(goal_rewards_indices) + len(collision_rewards_indices) + len(other_rewards_indices) == len(real_reward)
    return goal_stats, collision_stats, other_stats


def load_data_from(data_dir, max_read=None):
    assert os.path.exists(data_dir)
    files = [file for file in os.listdir(data_dir) if file.endswith(".pkl")]
    assert len(files) > 0
    random.shuffle(files)
    total_buffer = []
    for file in files:
        if max_read is not None and len(total_buffer) > max_read:
            break
        current_buffer = pickle.load(open(os.path.join(data_dir, file)))
        total_buffer.extend(current_buffer)
    return total_buffer


def print_model_stats(pre_trained_reward_network, test_batch_size, sess):
    # read the data
    test = load_data_from(os.path.join('supervised_data', 'test'), max_read=10 * test_batch_size)
    print len(test)

    # partition to train and test
    random.shuffle(test)

    openrave_manager = OpenraveManager(0.001, PotentialPoint.from_config(pre_trained_reward_network.config))

    sess.run(tf.global_variables_initializer())

    # run test for one (random) batch
    random.shuffle(test)
    test_batch = oversample_batch(test, 0, test_batch_size)
    test_batch, test_rewards, test_status = get_batch_and_labels(test_batch, openrave_manager)
    reward_prediction, status_prediction = pre_trained_reward_network.make_prediction(*([sess] + test_batch))
    # see what happens for different reward classes:
    goal_rewards_stats, collision_rewards_stats, other_rewards_stats = compute_stats_per_class(
        test_status, test_rewards, status_prediction, reward_prediction)
    print 'before loading weights'
    print 'goal mean_error {} max_error {} accuracy {}'.format(*goal_rewards_stats)
    print 'collision mean_error {} max_error {} accuracy {}'.format(*collision_rewards_stats)
    print 'other mean_error {} max_error {} accuracy {}'.format(*other_rewards_stats)

    # load weights
    pre_trained_reward_network.load_weights(sess)
    # run test for one (random) batch
    random.shuffle(test)

    test_batch = oversample_batch(test, 0, test_batch_size)
    test_batch, test_rewards, test_status = get_batch_and_labels(test_batch, openrave_manager)
    reward_prediction, status_prediction = pre_trained_reward_network.make_prediction(*([sess] + test_batch))
    # see what happens for different reward classes:
    goal_rewards_stats, collision_rewards_stats, other_rewards_stats = compute_stats_per_class(
        test_status, test_rewards, status_prediction, reward_prediction)
    print 'after loading weights'
    print 'goal mean_error {} max_error {} accuracy {}'.format(*goal_rewards_stats)
    print 'collision mean_error {} max_error {} accuracy {}'.format(*collision_rewards_stats)
    print 'other mean_error {} max_error {} accuracy {}'.format(*other_rewards_stats)
