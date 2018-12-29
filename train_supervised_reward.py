from image_cache import ImageCache
from pre_trained_reward import *

import time
import datetime
import numpy as np
import random
import os
import yaml
import tensorflow as tf

from openrave_manager import OpenraveManager
from potential_point import PotentialPoint


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

test_batch_size = batch_size * 10

scenario = config['general']['scenario']
base_data_dir = os.path.join('supervised_data', scenario)
image_cache = None
if scenario == 'vision':
    params_dir = os.path.abspath(os.path.expanduser('~/ModelBasedDDPG/scenario_params/vision/'))
    image_cache = ImageCache(params_dir)
# train = load_data_from(os.path.join(base_data_dir, 'train'), 100000)
# test = load_data_from(os.path.join(base_data_dir, 'test'), 100000)
train = load_data_from(os.path.join(base_data_dir, 'train'))
test = load_data_from(os.path.join(base_data_dir, 'test'))


def describe_data(data_collection):
    data_status = [t[-1] for t in data_collection]
    free_status = len([s for s in data_status if s == 1])
    collision_status = len([s for s in data_status if s == 2])
    goal_status = len([s for s in data_status if s == 3])
    print 'free: {} ({})'.format(free_status, float(free_status)/len(data_collection))
    print 'collision: {} ({})'.format(collision_status, float(collision_status)/len(data_collection))
    print 'goal: {} ({})'.format(goal_status, float(goal_status)/len(data_collection))
    print ''


print 'train description'
describe_data(train)
print 'test description'
describe_data(test)

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
        random.shuffle(train)
        data_index = 0
        while data_index < len(train):
            train_batch = oversample_batch(train, data_index, batch_size, oversample_large_magnitude=True)
            data_index += batch_size
            if train_batch is None:
                continue
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
            random.shuffle(test)
            test_batch = oversample_batch(test, 0, test_batch_size, oversample_large_magnitude=False)
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
