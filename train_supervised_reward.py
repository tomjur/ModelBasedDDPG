from image_cache import ImageCache
from pre_trained_reward import *
from openrave_manager import OpenraveManager
from potential_point import PotentialPoint

import time
import datetime
import numpy as np
import random
import os
import yaml
import tensorflow as tf


class RewardDataLoader:
    def __init__(self, data_dir, status_to_read, max_read=None):
        assert os.path.exists(data_dir)
        files = [f for f in os.listdir(data_dir) if f.endswith(".pkl") and f.startswith('{}_'.format(status_to_read))]
        assert len(files) > 0
        self.data_dir = data_dir
        self.files = files
        self.max_read = max_read

        self.first_iteration_done = False

    def __iter__(self):
        random.shuffle(self.files)
        files_to_keep = []
        row_counter = 0
        result = []
        for f in self.files:
            with bz2.BZ2File(os.path.join(self.data_dir, f), 'r') as compressed_file:
                result = pickle.load(compressed_file)
            row_counter += len(result)
            if not self.first_iteration_done and self.max_read is not None:
                files_to_keep.append(f)
                if row_counter > self.max_read:
                    break
            yield result
            result = []
        if not self.first_iteration_done:
            self.first_iteration_done = True
            if self.max_read is not None:
                self.files = files_to_keep
        yield result


class Batcher:
    def __init__(self, input_iterator, batch_size, shuffle_before_yield):
        self.input_iterator = input_iterator
        self.batch_size = batch_size
        self.shuffle_before_yield = shuffle_before_yield

    def __iter__(self):
        current_batch = []
        for tuple_list in self.input_iterator:
            for t in tuple_list:
                current_batch.append(t)
                if len(current_batch) == self.batch_size:
                    if self.shuffle_before_yield:
                        random.shuffle(current_batch)
                    yield current_batch
                    current_batch = []
        if self.shuffle_before_yield:
            random.shuffle(current_batch)
        yield current_batch


class Oversampler:
    def __init__(self, data_dir, free_class_batch_size, oversample_goal, oversample_collision, max_read=None, shuffle_batch_multiplier=2):
        self.data_dir = data_dir
        self.oversample_goal = oversample_goal
        self.oversample_collision = oversample_collision

        # load data
        free_transitions_iterator = RewardDataLoader(data_dir, 1, max_read)
        self.all_collisions = self._load_all(RewardDataLoader(data_dir, 2, max_read))
        self.all_goals = self._load_all(RewardDataLoader(data_dir, 3, max_read))
        self._describe_data(free_transitions_iterator)

        inner_batcher = Batcher(free_transitions_iterator, free_class_batch_size * shuffle_batch_multiplier, True)
        self.batcher = Batcher(inner_batcher, free_class_batch_size, False)

    def _describe_data(self, free_transitions_iterator):
        free_count = 0
        for tuple_list in free_transitions_iterator:
            free_count += len(tuple_list)
        collision_count = len(self.all_collisions)
        goal_count = len(self.all_goals)
        all_count = free_count + collision_count + goal_count

        print 'data dir: {}'.format(self.data_dir)
        print 'free: {} ({})'.format(free_count, float(free_count) / all_count)
        print 'collision: {} ({})'.format(collision_count, float(collision_count) / all_count)
        print 'goal: {} ({})'.format(goal_count, float(goal_count) / all_count)
        print ''

    @staticmethod
    def _load_all(files_iterator):
        all_transitions = []
        for tuple_list in files_iterator:
            all_transitions.extend(tuple_list)
        return all_transitions

    def _oversample_result(self, free_transition_batch):
        batch_size = len(free_transition_batch)
        goal_sample_size = int(self.oversample_goal * batch_size)
        goal_indices = list(np.random.choice(len(self.all_goals), goal_sample_size))
        goal_current_batch = [self.all_goals[i] for i in goal_indices]
        collision_sample_size = int(self.oversample_collision * batch_size)
        collision_indices = list(np.random.choice(len(self.all_collisions), collision_sample_size))
        collision_current_batch = [self.all_collisions[i] for i in collision_indices]
        return free_transition_batch + goal_current_batch + collision_current_batch

    def __iter__(self):
        for free_transition_batch in self.batcher:
            yield self._oversample_result(free_transition_batch)


model_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

# read the config
config_path = os.path.join(os.getcwd(), 'config/reward_config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)
    print('------------ Config ------------')
    print(yaml.dump(config))


epochs = config['general']['epochs']
save_every_epochs = config['general']['save_every_epochs']
batch_size = config['model']['batch_size']

initial_learn_rate = config['reward']['initial_learn_rate']
decrease_learn_rate_after = config['reward']['decrease_learn_rate_after']
learn_rate_decrease_rate = config['reward']['learn_rate_decrease_rate']
oversample_goal = config['reward']['oversample_goal']
oversample_collision = config['reward']['oversample_collision']

scenario = config['general']['scenario']
# base_data_dir = os.path.join('supervised_data', scenario)
base_data_dir = os.path.join('supervised_data', scenario + '_by_status')
image_cache = None
if scenario == 'vision':
    params_dir = os.path.abspath(os.path.expanduser('~/ModelBasedDDPG/scenario_params/vision/'))
    image_cache = ImageCache(params_dir)
train_data_dir = os.path.join(base_data_dir, 'train')
test_data_dir = os.path.join(base_data_dir, 'test')

train = Oversampler(train_data_dir, batch_size, oversample_goal, oversample_collision, max_read=10000)
test = Oversampler(train_data_dir, batch_size, oversample_goal, oversample_collision, max_read=10000)
# train = Oversampler(train_data_dir, batch_size, oversample_goal, oversample_collision)
# test = Oversampler(train_data_dir, batch_size, oversample_goal, oversample_collision)

# get openrave manager
openrave_manager = OpenraveManager(0.001, PotentialPoint.from_config(config))

# set summaries and saver dir
summaries_dir = os.path.join('reward', 'tensorboard')
train_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train_' + model_name))
test_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'test_' + model_name))
saver_dir = os.path.join('reward', 'model', model_name)
if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)

# save the config
config_copy_path = os.path.join(saver_dir, 'config.yml')
yaml.dump(config, open(config_copy_path, 'w'))

# create the network
pre_trained_reward = PreTrainedReward(model_name, config)
reward_prediction = pre_trained_reward.reward_prediction
reward_input = tf.placeholder(tf.float32, [None, 1])
# reward_loss = tf.reduce_mean(tf.square(reward_input - reward_prediction))
reward_loss = tf.losses.mean_squared_error(labels=reward_input, predictions=reward_prediction)
status_prediction = pre_trained_reward.status_softmax_logits
status_input = tf.placeholder(tf.int32, [None, ])
# status_loss = tf.reduce_mean(
#     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=status_input-1, logits=status_prediction)
# ) * config['reward']['cross_entropy_coefficient']
status_loss = tf.losses.sparse_softmax_cross_entropy(labels=status_input-1, logits=status_prediction) * config[
    'reward']['cross_entropy_coefficient']
regularization_loss = tf.losses.get_regularization_loss()
total_loss = reward_loss + status_loss + regularization_loss


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    initial_learn_rate, global_step, decrease_learn_rate_after, learn_rate_decrease_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, variables = zip(*optimizer.compute_gradients(total_loss, tf.trainable_variables()))
initial_gradients_norm = tf.global_norm(gradients)
gradient_limit = config['reward']['gradient_limit']
if gradient_limit > 0.0:
    gradients, _ = tf.clip_by_global_norm(gradients, gradient_limit, use_norm=initial_gradients_norm)
clipped_gradients_norm = tf.global_norm(gradients)
optimize_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

reward_loss_summary = tf.summary.scalar('reward_loss', reward_loss)
status_loss_summary = tf.summary.scalar('status_loss', status_loss)
regularization_loss_summary = tf.summary.scalar('regularization_loss_summary', regularization_loss)
total_loss_summary = tf.summary.scalar('total_loss_summary', total_loss)

# summaries for the reward optimization
train_optimization_summaries = tf.summary.merge([
    reward_loss_summary,
    status_loss_summary,
    regularization_loss_summary,
    total_loss_summary,
    tf.summary.scalar('gradients_norm_initial', initial_gradients_norm),
    tf.summary.scalar('gradients_norm_clipped', clipped_gradients_norm),
    tf.summary.scalar('learn_rate', learning_rate)
])

test_optimization_summaries = tf.summary.merge([
    reward_loss_summary,
    status_loss_summary,
    regularization_loss_summary,
    total_loss_summary,
])

goal_mean_rewards_error_input = tf.placeholder(tf.float32, name='goal_mean_rewards_error_input')
collision_mean_rewards_error_input = tf.placeholder(tf.float32, name='collision_mean_rewards_error_input')
other_mean_rewards_error_input = tf.placeholder(tf.float32, name='other_mean_rewards_error_input')
goal_max_rewards_error_input = tf.placeholder(tf.float32, name='goal_max_rewards_error_input')
collision_max_rewards_error_input = tf.placeholder(tf.float32, name='collision_max_rewards_error_input')
other_max_rewards_error_input = tf.placeholder(tf.float32, name='other_max_rewards_error_input')
goal_accuracy_input = tf.placeholder(tf.float32, name='goal_accuracy_input')
collision_accuracy_input = tf.placeholder(tf.float32, name='collision_accuracy_input')
other_accuracy_input = tf.placeholder(tf.float32, name='other_accuracy_input')

test_summaries = tf.summary.merge([
    tf.summary.scalar('goal_mean_rewards_error', goal_mean_rewards_error_input),
    tf.summary.scalar('collision_mean_rewards_error', collision_mean_rewards_error_input),
    tf.summary.scalar('other_mean_rewards_error', other_mean_rewards_error_input),
    tf.summary.scalar('goal_max_rewards_error', goal_max_rewards_error_input),
    tf.summary.scalar('collision_max_rewards_error', collision_max_rewards_error_input),
    tf.summary.scalar('other_max_rewards_error', other_max_rewards_error_input),
    tf.summary.scalar('goal_accuracy', goal_accuracy_input),
    tf.summary.scalar('collision_accuracy', collision_accuracy_input),
    tf.summary.scalar('other_accuracy', other_accuracy_input),
])


with tf.Session(
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
        )
) as sess:
    sess.run(tf.global_variables_initializer())
    current_global_step = 0
    for epoch in range(epochs):
        # run train for one epoch
        for train_batch in train:
            train_batch, train_rewards, train_status = get_batch_and_labels(train_batch, openrave_manager, image_cache)
            train_status_one_hot = np.zeros((len(train_rewards), 3), dtype=np.float32)
            train_status_one_hot[np.arange(len(train_rewards)), np.array(train_status)-1] = 1.0
            train_feed = pre_trained_reward.make_feed(*train_batch, all_transition_labels=train_status_one_hot)
            train_feed[reward_input] = np.expand_dims(np.array(train_rewards), axis=1)
            train_feed[status_input] = np.array(train_status)
            train_total_loss, train_summary, current_global_step, _ = sess.run(
                [total_loss, train_optimization_summaries, global_step, optimize_op], train_feed)
            # if np.isnan(train_loss):
            #     for b in train_batch:
            #         print b
            #     print train_rewards
            #     exit()
            train_summary_writer.add_summary(train_summary, current_global_step)
        train_summary_writer.flush()

        if current_global_step > 0:
            # run test for one (random) batch
            test_batch = None
            for test_batch in test:
                break
            test_batch, test_rewards, test_status = get_batch_and_labels(test_batch, openrave_manager, image_cache)
            test_feed = pre_trained_reward.make_feed(*test_batch)
            test_feed[reward_input] = np.expand_dims(np.array(test_rewards), axis=1)
            test_feed[status_input] = np.array(test_status)
            test_reward_prediction, test_status_prediction, test_total_loss, test_summary = sess.run(
                [reward_prediction, status_prediction, total_loss, test_optimization_summaries], test_feed)
            test_summary_writer.add_summary(test_summary, current_global_step)
            # see what happens for different reward classes:
            goal_stats, collision_stats, other_stats = compute_stats_per_class(
                test_status, test_rewards, test_status_prediction, test_reward_prediction)
            test_summary_writer.add_summary(sess.run(test_summaries, {
                goal_mean_rewards_error_input: goal_stats[0],
                collision_mean_rewards_error_input: collision_stats[0],
                other_mean_rewards_error_input: other_stats[0],
                goal_max_rewards_error_input: goal_stats[1],
                collision_max_rewards_error_input: collision_stats[1],
                other_max_rewards_error_input: other_stats[1],
                goal_accuracy_input: goal_stats[2],
                collision_accuracy_input: collision_stats[2],
                other_accuracy_input: other_stats[2],
            }), current_global_step)
            test_summary_writer.flush()
            # save the model
            if epoch % save_every_epochs == save_every_epochs-1:
                pre_trained_reward.saver.save(sess, os.path.join(saver_dir, 'reward'), global_step=current_global_step)
        print 'done epoch {} of {}, global step {}'.format(epoch, epochs, current_global_step)
