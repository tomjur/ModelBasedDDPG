import bz2
import numpy as np
import random
import pickle
import os
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

from dqn_model import DqnModel
from modeling_utils import get_activation


class PreTrainedReward:

    def __init__(self, model_name, config):
        self._reuse_flag = False

        self.config = config
        self.is_vision_enabled = config['general']['scenario'] == 'vision'

        self.joints_inputs = tf.placeholder(tf.float32, (None, 4), name='joints_inputs')
        self.goal_joints_inputs = tf.placeholder(tf.float32, (None, 4), name='goal_joints_inputs')
        self.workspace_image_inputs, self.images_3d = None, None
        if self.is_vision_enabled:
            self.workspace_image_inputs = tf.placeholder(tf.float32, (None, 55, 111), name='workspace_image_inputs')
            self.images_3d = tf.expand_dims(self.workspace_image_inputs, axis=-1)
        self.goal_pose_inputs = tf.placeholder(tf.float32, (None, 2), name='goal_pose_inputs')
        self.action_inputs = tf.placeholder(tf.float32, (None, 4), name='action_inputs')
        self.transition_label = tf.placeholder_with_default([[0.0]*3], (None, 3), name='labeled_transition')
        current_variables_count = len(tf.trainable_variables())
        self.reward_prediction, self.status_softmax_logits = self.create_reward_network(
            self.joints_inputs, self.action_inputs, self.goal_joints_inputs, self.goal_pose_inputs, self.images_3d
        )
        reward_variables = tf.trainable_variables()[current_variables_count:]

        # model path to load
        self.model_name = model_name
        self.saver_dir = os.path.join('reward', 'model', model_name)
        assert os.path.exists(self.saver_dir)
        self.saver = tf.train.Saver(reward_variables, max_to_keep=4, save_relative_paths=self.saver_dir)

    @staticmethod
    def _generate_goal_features(goal_joints_inputs, goal_pose_inputs):
        features = [goal_joints_inputs]
        if goal_pose_inputs is not None:
            features.append(goal_pose_inputs)
        return tf.concat(features, axis=1)

    def _next_state_model(self, joints_inputs, action_inputs):
        # next step is a deterministic computation
        action_step_size = self.config['openrave_rl']['action_step_size']
        step = action_inputs * action_step_size
        unclipped_result = joints_inputs + step
        # we initiate an openrave manager to get the robot, to get the joint bounds and the safety
        joint_safety = 0.0001
        lower_bounds = [-2.617, -1.571, -1.571, -1.745, -2.617]
        lower_bounds = [b + joint_safety for b in lower_bounds[1:]]
        upper_bounds = [-b for b in lower_bounds]

        # clip the result
        clipped_result = tf.maximum(unclipped_result, lower_bounds)
        clipped_result = tf.minimum(clipped_result, upper_bounds)
        return clipped_result, unclipped_result

    def create_reward_network(
            self, joints_inputs, action_inputs, goal_joints_inputs, goal_pose_inputs, images_3d):
        name_prefix = 'reward'
        # get the next joints
        clipped_next_joints, unclipped_next_joints = self._next_state_model(joints_inputs, action_inputs)

        # predict the transition classification
        layers = self.config['reward']['layers'] + [3]
        scale = 0.0
        if 'l2_regularization_coefficient' in self.config['reward']:
            scale = self.config['reward']['l2_regularization_coefficient']
        current = tf.concat(
            (clipped_next_joints, self._generate_goal_features(goal_joints_inputs, goal_pose_inputs)), axis=1)
        # add vision if needed
        if self.is_vision_enabled:
            visual_inputs = DqnModel(name_prefix).predict(images_3d, self._reuse_flag)
            current = tf.concat((current, visual_inputs), axis=1)
        for i, layer_size in enumerate(layers):
            _activation = None if i == len(layers) - 1 else get_activation(self.config['reward']['activation'])
            current = tf.layers.dense(
                current, layer_size, activation=_activation, name='{}_layers_{}'.format(name_prefix, i),
                kernel_regularizer=tf_layers.l2_regularizer(scale), reuse=self._reuse_flag
            )
        softmax_logits = current
        softmax_res = tf.nn.softmax(softmax_logits)

        # if the one-hot input is fed, is labeled will be 1.0 otherwise it will be zero
        is_labeled = tf.expand_dims(tf.reduce_max(self.transition_label, axis=1), axis=1)
        reward_calculation_input = self.transition_label + tf.multiply(1.0 - is_labeled, softmax_res)

        # get the classification reward
        classification_reward = tf.layers.dense(
            reward_calculation_input, 1, activation=None, use_bias=False,
            name='{}_classification_reward'.format(name_prefix), reuse=self._reuse_flag
            )

        # get the clipping-related reward
        # clipped_difference = tf.expand_dims(tf.norm(unclipped_next_joints - clipped_next_joints, axis=1), axis=1)  # this is the original
        # clipped_difference = tf.expand_dims(tf.reduce_sum(tf.zeros_like(clipped_next_joints), axis=1), axis=1)  # this will have no gradient backlash
        clipped_difference = tf.expand_dims(tf.reduce_sum(tf.abs(unclipped_next_joints - clipped_next_joints), axis=1), axis=1)

        clipping_reward = tf.layers.dense(
            clipped_difference, 1, activation=None, use_bias=False, name='{}_clipping_weight'.format(name_prefix),
            reuse=self._reuse_flag
        )

        total_reward = classification_reward + clipping_reward
        self._reuse_flag = True
        return total_reward, softmax_logits

    def load_weights(self, sess):
        self.saver.restore(sess, tf.train.latest_checkpoint(self.saver_dir))

    def make_prediction(self, sess, all_start_joints, all_goal_joints, all_actions, all_goal_poses,
                        all_transition_labels=None, images=None):
        feed = self.make_feed(all_start_joints, all_goal_joints, all_actions, all_goal_poses, images=images,
                              all_transition_labels=all_transition_labels)
        return sess.run([self.reward_prediction, self.status_softmax_logits], feed)

    def make_feed(self, all_start_joints, all_goal_joints, all_actions, all_goal_poses, images=None,
                  all_transition_labels=None):
        feed = {
            self.joints_inputs: all_start_joints,
            self.goal_joints_inputs: all_goal_joints,
            self.action_inputs: all_actions,
        }
        if self.goal_pose_inputs is not None:
            feed[self.goal_pose_inputs] = all_goal_poses
        if self.is_vision_enabled:
            assert images is not None
            assert images[0] is not None
            feed[self.workspace_image_inputs] = images
        if all_transition_labels is not None:
            feed[self.transition_label] = all_transition_labels
        return feed


def oversample_batch(current_batch, oversample_large_magnitude=None):
    if oversample_large_magnitude is None:
        return current_batch
    oversample_success = oversample_large_magnitude[0]
    oversample_collision = oversample_large_magnitude[1]
    status = [b[-1] for b in current_batch]
    success_reward_indices = [i for i, s in enumerate(status) if s == 3]
    if len(success_reward_indices) < 3:
        return None
    collision_reward_indices = [i for i, s in enumerate(status) if s == 2]
    if len(collision_reward_indices) < 3:
        return None
    other_reward_indices = [i for i, s in enumerate(status) if s == 1]
    assert len(success_reward_indices) + len(collision_reward_indices) + len(other_reward_indices) == len(current_batch)
    sample_size = len(other_reward_indices)
    batch_indices = other_reward_indices
    # sample_size = min(100, len(other_reward_indices))
    # batch_indices = list(np.random.choice(other_reward_indices, sample_size))
    success_sample_size = int(oversample_success * sample_size)
    success_super_sample = list(np.random.choice(success_reward_indices, success_sample_size))
    batch_indices.extend(success_super_sample)
    collision_sample_size = int(oversample_collision * sample_size)
    collision_super_sample = list(np.random.choice(collision_reward_indices, collision_sample_size))
    batch_indices.extend(collision_super_sample)
    return [current_batch[i] for i in batch_indices]


def get_batch_and_labels(batch, openrave_manager, image_cache):
    all_start_joints = []
    all_goal_joints = []
    all_actions = []
    all_rewards = []
    all_goal_poses = []
    all_status = []
    all_images = None
    if image_cache is not None:
        all_images = []
    for i in range(len(batch)):
        if image_cache is None:
            workspace_id = None
            start_joints, goal_joints, action, next_joints, reward, terminated, status = batch[i]
        else:
            workspace_id, start_joints, goal_joints, action, next_joints, reward, terminated, status = batch[i]
        goal_pose = openrave_manager.get_target_pose(goal_joints)
        all_start_joints.append(start_joints[1:])
        all_goal_joints.append(goal_joints[1:])
        all_actions.append(action[1:])
        all_rewards.append(reward)
        all_status.append(status)
        all_goal_poses.append(goal_pose)
        if image_cache is not None:
            image = image_cache.items[workspace_id].np_array
            all_images.append(image)
    return [all_start_joints, all_goal_joints, all_actions, all_goal_poses, all_images], all_rewards, all_status


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


def load_data_from(data_dir, max_read=None, is_vision=False):
    assert os.path.exists(data_dir)
    files = [file for file in os.listdir(data_dir) if file.endswith(".pkl")]
    assert len(files) > 0
    random.shuffle(files)
    total_buffer = []
    for file in files:
        if max_read is not None and len(total_buffer) > max_read:
            break
        compressed_file = bz2.BZ2File(os.path.join(data_dir, file), 'r')
        current_buffer = pickle.load(compressed_file)
        compressed_file.close()
        if is_vision:
            parts = file.split('_')
            workspace_id = '{}_{}.pkl'.format(parts[0], parts[1])
            current_buffer = [tuple([workspace_id] + list(t)) for t in current_buffer]
        total_buffer.extend(current_buffer)
    return total_buffer
