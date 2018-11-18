import pickle
import time
import datetime
import numpy as np
import random
import os
import yaml
import tensorflow as tf

from network import Network
from rollout_manager import RolloutManager
from trajectory_eval import TrajectoryEval

model_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

# read the config
config_path = os.path.join(os.getcwd(), 'config/imitation_config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)
    print('------------ Config ------------')
    print(yaml.dump(config))


epochs = config['general']['epochs']
save_every_epochs = config['general']['save_every_epochs']
batch_size = config['model']['batch_size']

test_batch_size = batch_size * 10


def load_data_from(data_dir, max_read=None):
    assert os.path.exists(data_dir)
    files = [file for file in os.listdir(data_dir) if file.endswith(".pkl")]
    assert len(files) > 0
    random.shuffle(files)
    paths = []
    for file in files:
        if max_read is not None and len(paths) > max_read:
            break
        current_buffer = pickle.load(open(os.path.join(data_dir, file)))
        paths.extend(current_buffer)

    # assert length
    step_size = config['openrave_rl']['action_step_size'] + 0.00001
    for (traj, image, goal_pose) in paths:
        for i in range(len(traj)-1):
            assert np.linalg.norm(np.array(traj[i]) - np.array(traj[i+1])) < step_size

    # partition traj to buffer (current joints, goal, next_joints)
    transitions = []
    for (traj, image, goal_pose) in paths:
        # goal_pose = openrave_manager.get_target_pose(traj[0])
        goal_joints = traj[-1]
        for i in range(len(traj)-1):
            joints = traj[i]
            next_joints = traj[i+1]
            transition = (joints[1:], next_joints[1:], goal_joints[1:], goal_pose)
            transitions.append(transition)
    return paths, transitions


train_paths, train_transitions = load_data_from(os.path.join('imitation_data', 'train'), 50000)
test_paths, test_transitions = load_data_from(os.path.join('imitation_data', 'test'), 50000)
# train_paths, train_transitions = load_data_from(os.path.join('imitation_data', 'train'))
# test_paths, test_transitions = load_data_from(os.path.join('imitation_data', 'test'))


# set summaries and saver dir
summaries_dir = os.path.join('imitation', 'tensorboard')
train_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train_' + model_name))
test_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'test_' + model_name))
saver_dir = os.path.join('imitation', 'model', model_name)
if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)
completed_trajectories_dir = os.path.join('imitation', 'trajectories', model_name)


# save the config
config_copy_path = os.path.join(saver_dir, 'config.yml')
yaml.dump(config, open(config_copy_path, 'w'))

# create the network
network = Network(config, True)
next_joints_inputs = tf.placeholder(tf.float32, (None, 4), name='next_joints')

# compute loss
relative_target = tf.nn.l2_normalize(next_joints_inputs - network.joints_inputs, 1)
loss = tf.losses.cosine_distance(network.online_action, relative_target, axis=1, reduction=tf.losses.Reduction.MEAN)

# optimize
global_step = tf.Variable(0, trainable=False)
initial_learn_rate = config['imitation']['initial_learn_rate']
decrease_learn_rate_after = config['imitation']['decrease_learn_rate_after']
learn_rate_decrease_rate = config['imitation']['learn_rate_decrease_rate']
learning_rate = tf.train.exponential_decay(
    initial_learn_rate, global_step, decrease_learn_rate_after, learn_rate_decrease_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, variables = zip(*optimizer.compute_gradients(loss))
current_gradients_norm = tf.global_norm(gradients)
gradients, _ = tf.clip_by_global_norm(gradients, config['imitation']['gradient_limit'], use_norm=current_gradients_norm)
clipped_gradients_norm = tf.global_norm(gradients)
train_step = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


# summaries
loss_summary = tf.summary.scalar('loss', loss)
model_summary_step = tf.summary.merge([
    loss_summary,
    tf.summary.scalar('learn_rate', learning_rate),
    tf.summary.scalar('gradients_norm_initial', current_gradients_norm),
    tf.summary.scalar('gradients_norm_clipped', clipped_gradients_norm),
])

train_success_rate_input = tf.placeholder(tf.float32, name='train_success_rate_input')
train_collision_rate_input = tf.placeholder(tf.float32, name='train_collision_rate_input')
train_max_len_rate_input = tf.placeholder(tf.float32, name='train_max_len_rate_input')
train_rate_summary = tf.summary.merge([
    tf.summary.scalar('train_success_rate', train_success_rate_input),
    tf.summary.scalar('train_collision_rate', train_collision_rate_input),
    tf.summary.scalar('train_max_len_rate', train_max_len_rate_input),
])

test_success_rate_input = tf.placeholder(tf.float32, name='test_success_rate_input')
test_collision_rate_input = tf.placeholder(tf.float32, name='test_collision_rate_input')
test_max_len_rate_input = tf.placeholder(tf.float32, name='test_max_len_rate_input')
test_rate_summary = tf.summary.merge([
    tf.summary.scalar('test_success_rate', test_success_rate_input),
    tf.summary.scalar('test_collision_rate', test_collision_rate_input),
    tf.summary.scalar('test_max_len_rate', test_max_len_rate_input),
])

random_success_rate_input = tf.placeholder(tf.float32, name='random_success_rate_input')
random_collision_rate_input = tf.placeholder(tf.float32, name='random_collision_rate_input')
random_max_len_rate_input = tf.placeholder(tf.float32, name='random_max_len_rate_input')
random_rate_summary = tf.summary.merge([
    tf.summary.scalar('random_success_rate', random_success_rate_input),
    tf.summary.scalar('random_collision_rate', random_collision_rate_input),
    tf.summary.scalar('random_max_len_rate', random_max_len_rate_input),
])



def get_batch(data_collection, data_index, batch_size):
    batch = data_collection[data_index:data_index + batch_size]
    return zip(*batch)


def print_state(prefix, episodes, successful_episodes, collision_episodes, max_len_episodes):
    print '{}: {}: finished: {}, successful: {} ({}), collision: {} ({}), max length: {} ({})'.format(
        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), prefix, episodes,
        successful_episodes, float(successful_episodes) / episodes, collision_episodes,
        float(collision_episodes) / episodes, max_len_episodes, float(max_len_episodes) / episodes
    )


# evaluate train paths
train_rollout_manager = RolloutManager(config, collector_processes=0, fixed_queries=train_paths)
train_trajectory_eval = TrajectoryEval(config, train_rollout_manager, os.path.join(completed_trajectories_dir, 'train'))

# evaluate test paths
test_rollout_manager = RolloutManager(config, collector_processes=0, fixed_queries=test_paths)
test_trajectory_eval = TrajectoryEval(config, test_rollout_manager, os.path.join(completed_trajectories_dir, 'test'))

# evaluate random paths
random_rollout_manager = RolloutManager(config)
random_trajectory_eval = TrajectoryEval(
    config, random_rollout_manager, os.path.join(completed_trajectories_dir, 'random'))

with tf.Session(
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
        )
) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        # run train for one epoch
        random.shuffle(train_transitions)
        data_index = 0
        while data_index < len(train_transitions):
            train_batch = get_batch(train_transitions, data_index, batch_size)
            data_index += batch_size
            train_feed = {
                network.joints_inputs: train_batch[0],
                next_joints_inputs: train_batch[1],
                network.goal_joints_inputs: train_batch[2],
                network.goal_pose_inputs: train_batch[3]
            }

            train_total_loss, train_summary, current_global_step, _ = sess.run(
                [loss, model_summary_step, global_step, train_step], train_feed)
            train_summary_writer.add_summary(train_summary, current_global_step)
        train_summary_writer.flush()

        # run test for one (random) batch
        random.shuffle(test_transitions)
        test_batch = get_batch(test_transitions, 0, test_batch_size)
        test_feed = {
            network.joints_inputs: test_batch[0],
            next_joints_inputs: test_batch[1],
            network.goal_joints_inputs: test_batch[2],
            network.goal_pose_inputs: test_batch[3]
        }
        test_action_prediction, test_total_loss, test_summary = sess.run(
            [network.online_action, loss, loss_summary], test_feed)
        test_summary_writer.add_summary(test_summary, current_global_step)

        # test
        # set current weights
        actor_trained_weights = network.get_actor_online_weights(sess)
        train_rollout_manager.set_policy_weights(actor_trained_weights)
        test_rollout_manager.set_policy_weights(actor_trained_weights)
        random_rollout_manager.set_policy_weights(actor_trained_weights)
        # do trajectory evaluations
        eval_result = train_trajectory_eval.eval(current_global_step, None)
        test_episodes = eval_result[0]
        test_successful_episodes = eval_result[1]
        test_collision_episodes = eval_result[2]
        test_max_len_episodes = eval_result[3]
        test_mean_reward = eval_result[4]
        print_state('train episodes', test_episodes, test_successful_episodes, test_collision_episodes,
                        test_max_len_episodes)
        print('train episodes mean total reward {}'.format(test_mean_reward))
        test_summary_writer.add_summary(sess.run(train_rate_summary, {
            train_success_rate_input: test_successful_episodes / float(test_episodes),
            train_collision_rate_input: test_collision_episodes / float(test_episodes),
            train_max_len_rate_input: test_max_len_episodes / float(test_episodes),
        }), current_global_step)
        
        eval_result = test_trajectory_eval.eval(current_global_step, None)
        test_episodes = eval_result[0]
        test_successful_episodes = eval_result[1]
        test_collision_episodes = eval_result[2]
        test_max_len_episodes = eval_result[3]
        test_mean_reward = eval_result[4]
        print_state('test episodes', test_episodes, test_successful_episodes, test_collision_episodes,
                        test_max_len_episodes)
        print('test episodes mean total reward {}'.format(test_mean_reward))
        test_summary_writer.add_summary(sess.run(test_rate_summary, {
            test_success_rate_input: test_successful_episodes / float(test_episodes),
            test_collision_rate_input: test_collision_episodes / float(test_episodes),
            test_max_len_rate_input: test_max_len_episodes / float(test_episodes),
        }), current_global_step)
        
        eval_result = random_trajectory_eval.eval(current_global_step, None)
        test_episodes = eval_result[0]
        test_successful_episodes = eval_result[1]
        test_collision_episodes = eval_result[2]
        test_max_len_episodes = eval_result[3]
        test_mean_reward = eval_result[4]
        print_state('random episodes', test_episodes, test_successful_episodes, test_collision_episodes,
                        test_max_len_episodes)
        print('random episodes mean total reward {}'.format(test_mean_reward))
        test_summary_writer.add_summary(sess.run(random_rate_summary, {
            random_success_rate_input: test_successful_episodes / float(test_episodes),
            random_collision_rate_input: test_collision_episodes / float(test_episodes),
            random_max_len_rate_input: test_max_len_episodes / float(test_episodes),
        }), current_global_step)

        test_summary_writer.flush()
        # # save the model
        # if epoch % save_every_epochs == save_every_epochs-1:
        #     pre_trained_reward.saver.save(sess, os.path.join(saver_dir, 'reward'), global_step=current_global_step)
        print 'done epoch {} of {}'.format(epoch, epochs)
    train_rollout_manager.end()
    test_rollout_manager.end()
    random_rollout_manager.end()
