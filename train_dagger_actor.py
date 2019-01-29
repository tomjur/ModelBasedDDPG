import bz2
import copy
import cPickle as pickle
import time
import datetime
import numpy as np
import random
import os
import yaml
import tensorflow as tf
import multiprocessing
import Queue

from network import Network
from openrave_rl_interface import OpenraveRLInterface
from potential_point import PotentialPoint
from rollout_manager import ActorProcess
from trajectory_eval import TrajectoryEval


model_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

# read the config
config_path = os.path.join(os.getcwd(), 'config/dagger_config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)
    print('------------ Config ------------')
    print(yaml.dump(config))


epochs = config['general']['epochs']
scenario = config['general']['scenario']
save_every_epochs = config['general']['save_every_epochs']
batch_size = config['model']['batch_size']

config['general']['params_file'] = os.path.abspath(
    os.path.expanduser(os.path.join('~/ModelBasedDDPG/scenario_params', scenario, 'params.pkl')))

test_batch_size = batch_size * 10


def produce_transitions(data_dir, cache_dir):
    print 'producing transition data from original trajectories at {}'.format(data_dir)
    assert os.path.exists(data_dir)

    if os.path.exists(cache_dir):
        print 'found cache dir at {}, assuming all transitions are present there (if not delete the directory)'.format(
            cache_dir)
        return

    print 'cache not found, creating cache at: {}'.format(cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    files = [file for file in os.listdir(data_dir) if file.endswith(".path_pkl")]
    assert len(files) > 0
    target_point = PotentialPoint.from_config(config)[-1]
    for file in files:
        print 'loading file {}'.format(file)
        with bz2.BZ2File(os.path.join(data_dir, file), 'r') as compressed_file:
            paths = pickle.load(compressed_file)

        print 'asserting step sizes match'
        step_size = config['openrave_rl']['action_step_size'] + 0.00001
        for (traj, _) in paths:
            for i in range(len(traj) - 1):
                assert np.linalg.norm(np.array(traj[i]) - np.array(traj[i + 1])) < step_size

        print 'creating transitions'
        transitions = []
        for (traj, poses_trajectory) in paths:
            goal_joints = traj[-1]
            goal_pose = poses_trajectory[-1][target_point.tuple]
            for i in range(len(traj) - 1):
                joints = traj[i]
                next_joints = traj[i + 1]
                transition = (joints[1:], next_joints[1:], goal_joints[1:], goal_pose)
                transitions.append(transition)

        transition_file = os.path.join(cache_dir, file + '.transitions_cache')
        print 'writing transitions file {}'.format(transition_file)
        with open(transition_file, 'w') as pickle_file:
            pickle.dump(transitions, pickle_file)
        # with bz2.BZ2File(transition_file, 'w') as compressed_file:
        #     pickle.dump(transitions, compressed_file)

    print 'cache created at {}'.format(cache_dir)


def produce_paths(data_dir, cache_dir):
    print 'producing paths data from original trajectories at {}'.format(data_dir)
    assert os.path.exists(data_dir)

    if os.path.exists(cache_dir):
        print 'found cache dir at {}, assuming all paths are present there (if not delete the directory)'.format(
            cache_dir)
        return

    print 'cache not found, creating cache at: {}'.format(cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    files = [file for file in os.listdir(data_dir) if file.endswith(".path_pkl")]
    assert len(files) > 0
    for file in files:
        print 'loading file {}'.format(file)
        with bz2.BZ2File(os.path.join(data_dir, file), 'r') as compressed_file:
            paths = pickle.load(compressed_file)

        print 'asserting step sizes match'
        step_size = config['openrave_rl']['action_step_size'] + 0.00001
        for (traj, _) in paths:
            for i in range(len(traj) - 1):
                assert np.linalg.norm(np.array(traj[i]) - np.array(traj[i + 1])) < step_size

        paths_file = os.path.join(cache_dir, file + '.paths_cache')
        print 'writing paths file {}'.format(paths_file)
        with open(paths_file, 'w') as pickle_file:
            pickle.dump(paths, pickle_file)

    print 'cache created at {}'.format(cache_dir)


train_original_dir = os.path.join('imitation_data', scenario, 'train')
train_transitions_dir = os.path.join('imitation_data_transitions', scenario, 'train')
train_transitions_dir = os.path.join(train_transitions_dir, PotentialPoint.from_config(config)[-1].str)
produce_transitions(train_original_dir, train_transitions_dir)
train_paths_dir = os.path.join('imitation_data_paths', scenario, 'train')
produce_paths(train_original_dir, train_paths_dir)

test_original_dir = os.path.join('imitation_data', scenario, 'test')
test_transitions_dir = os.path.join('imitation_data_transitions', scenario, 'test')
test_transitions_dir = os.path.join(test_transitions_dir, PotentialPoint.from_config(config)[-1].str)
produce_transitions(test_original_dir, test_transitions_dir)
test_paths_dir = os.path.join('imitation_data_paths', scenario, 'test')
produce_paths(test_original_dir, test_paths_dir)


def get_files(paths_dir, transitions_dir, max_files=None):
    print 'loading from paths {} transitions {}. max files {}'.format(paths_dir, transitions_dir, max_files)
    assert os.path.exists(paths_dir)
    assert os.path.exists(transitions_dir)
    files = [file for file in os.listdir(paths_dir) if file.endswith(".paths_cache")]
    assert len(files) > 0
    for file in files:
        assert os.path.exists(os.path.join(transitions_dir, file).replace(".paths_cache", '.transitions_cache'))
    random.shuffle(files)
    files = files[:max_files]
    path_files = [os.path.join(paths_dir, f) for f in files]
    transition_files = [os.path.join(transitions_dir, f.replace(".paths_cache", '.transitions_cache')) for f in files]
    return path_files, transition_files


train_path_files, train_transition_files = get_files(train_paths_dir, train_transitions_dir, config[
    'general']['train_files'])
test_path_files, test_transition_files = get_files(test_paths_dir, test_transitions_dir, config[
    'general']['test_files'])

# set summaries and saver dir
base_dir = os.path.join('imitation_dagger', scenario)
summaries_dir = os.path.join(base_dir, 'tensorboard')
train_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train_' + model_name))
test_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'test_' + model_name))
saver_dir = os.path.join(base_dir, 'model', model_name)
if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)
train_completed_trajectories_dir = os.path.join(base_dir, 'trajectories', 'train', model_name)
test_completed_trajectories_dir = os.path.join(base_dir, 'trajectories', 'test', model_name)

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


def print_state(prefix, episodes, successful_episodes, collision_episodes, max_len_episodes):
    print '{}: {}: finished: {}, successful: {} ({}), collision: {} ({}), max length: {} ({})'.format(
        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), prefix, episodes,
        successful_episodes, float(successful_episodes) / episodes, collision_episodes,
        float(collision_episodes) / episodes, max_len_episodes, float(max_len_episodes) / episodes
    )


class ImitationRolloutManager:
    def __init__(self, config, train_trajectories_files, test_trajectories_files):
        self._train_index = 0
        self._test_index = 0
        self.train_trajectories = self.files_to_trajectories(train_trajectories_files)
        self.test_trajectories = self.files_to_trajectories(test_trajectories_files)

        self.episode_generation_queue = multiprocessing.JoinableQueue()
        self.episode_results_queue = multiprocessing.Queue()

        actor_processes = config['general']['actor_processes']
        if actor_processes is None:
            actor_processes = multiprocessing.cpu_count()

        self.actor_specific_queues = [multiprocessing.JoinableQueue() for _ in range(actor_processes)]

        self.actors = [
            ActorProcess(copy.deepcopy(config), self.episode_generation_queue, self.episode_results_queue,
                         self.actor_specific_queues[i], None)
            for i in range(actor_processes)
        ]
        # start all the actor processes
        for a in self.actors:
            a.start()
        # for every actor process, post a message to initialize the actor network
        for actor_queue in self.actor_specific_queues:
            actor_queue.put((0, ))
            actor_queue.join()

    @staticmethod
    def files_to_trajectories(files):
        paths = []
        for f in files:
            with open(f, 'r') as pickle_file:
                current_buffer = pickle.load(pickle_file)
                paths.extend([t[0] for t in current_buffer])
        return paths

    def generate_episodes(self, number_of_episodes, is_train):
        # use collectors to generate queries
        for i in range(number_of_episodes):
            if is_train:
                traj = self.train_trajectories[self._train_index]
                self._train_index += 1
                if self._train_index == len(self.train_trajectories):
                    self._train_index = 0
                    random.shuffle(self.train_trajectories)
            else:
                traj = self.test_trajectories[self._test_index]
                self._test_index += 1
                if self._test_index == len(self.test_trajectories):
                    self._test_index = 0
                    random.shuffle(self.test_trajectories)
            # get a query
            message = ((traj, None, None), False)  # poses and workspace image are not required
            # message = ((traj, None, None), True)  # poses and workspace image are not required
            # place in queue
            self.episode_generation_queue.put(message)

        self.episode_generation_queue.join()

        episodes = []
        while number_of_episodes:
            number_of_episodes -= 1
            episodes.append(self.episode_results_queue.get())

        return episodes

    def set_policy_weights(self, weights, is_online):
        message = (2, weights, is_online)
        self._post_private_message(message, self.actor_specific_queues)

    def end(self):
        message = (1, )
        self._post_private_message(message, self.actor_specific_queues)
        time.sleep(10)
        for a in self.actors:
            a.terminate()
        time.sleep(10)

    @staticmethod
    def _post_private_message(message, queues):
        for queue in queues:
            queue.put(message)
        for queue in queues:
            queue.join()


# evaluate train paths
rollout_manager = ImitationRolloutManager(config, train_path_files, test_path_files)
train_trajectory_eval = TrajectoryEval(config, rollout_manager, train_completed_trajectories_dir)

# evaluate test paths
test_trajectory_eval = TrajectoryEval(config, rollout_manager, test_completed_trajectories_dir)


test_results = []
best_epoch = -1
best_success_rate = -1.0
best_model_path = None

latest_saver = tf.train.Saver(max_to_keep=2, save_relative_paths=saver_dir)
best_saver = tf.train.Saver(max_to_keep=2, save_relative_paths=saver_dir)


class TransitionDataLoader:
    def __init__(self, files, limit_files=None):
        self.files = files
        if limit_files is not None:
            self.files = self.files[:limit_files]
        self.files = self.files + [None]
        self.extra_transitions = []

    def __iter__(self):
        total_transitions_counter = 0
        random.shuffle(self.files)
        for f in self.files:
            if f is None:
                if len(self.extra_transitions) > 0:
                    random.shuffle(self.extra_transitions)
                    total_transitions_counter += len(self.extra_transitions)
                    yield self.extra_transitions
            else:
                with open(f, 'r') as pickle_file:
                    file_transitions = pickle.load(pickle_file)
                    total_transitions_counter += len(file_transitions)
                    yield file_transitions
        print 'total transitions seen in epoch {}'.format(total_transitions_counter)

    def add_extra_transitions(self, transitions):
        self.extra_transitions.extend(transitions)


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


train_transition_data_loader = TransitionDataLoader(train_transition_files, config['general']['train_transition_files'])
train_batcher = Batcher(train_transition_data_loader, batch_size, True)
test_batcher = Batcher(TransitionDataLoader(test_transition_files), test_batch_size, True)


class OpenravePlannerProcess(multiprocessing.Process):
    def __init__(self, config, generate_plan_queue, result_queue, planner_specific_queue):
        multiprocessing.Process.__init__(self)
        self.generate_plan_queue = generate_plan_queue
        self.result_queue = result_queue
        self.planner_specific_queue = planner_specific_queue
        self.config = config
        # members to set at runtime
        self.openrave_interface = None

    def _run_query(self, query_params):
        transitions_list = []
        start_state = query_params[0]
        goal_state = query_params[1]
        goal_pose = self.openrave_interface.openrave_manager.get_target_pose(goal_state)
        traj = self.openrave_interface.openrave_manager.plan(start_state, goal_state, config['openrave_rl']['planner_iterations'])
        if traj is not None:
            split_traj = OpenraveRLInterface.split_trajectory(traj, config['openrave_rl']['action_step_size'])
            for i in range(len(split_traj)-1):
                current_state = split_traj[i]
                next_state = split_traj[i+1]
                t = [current_state[1:], next_state[1:], goal_state[1:], goal_pose]
                transitions_list.append(t)
        return transitions_list

    def _run_main_loop(self):
        while True:
            try:
                # wait 1 second for a trajectory request
                query_params = self.generate_plan_queue.get(block=True, timeout=1)
                t = self._run_query(query_params)
                self.result_queue.put(t)
                self.generate_plan_queue.task_done()
            except Queue.Empty:
                pass
            try:
                next_actor_specific_task = self.planner_specific_queue.get(block=True, timeout=0.001)
                self.planner_specific_queue.task_done()
                break
            except Queue.Empty:
                pass

    def run(self):
        params_file = os.path.abspath(os.path.expanduser(self.config['general']['params_file']))
        self.openrave_interface = OpenraveRLInterface(self.config)
        self.openrave_interface.openrave_manager.set_params(params_file)
        self._run_main_loop()


class OpenravePlannerManager:
    def __init__(self, config):
        self.episode_generation_queue = multiprocessing.JoinableQueue()
        self.episode_results_queue = multiprocessing.Queue()

        actor_processes = config['general']['openrave_processes']

        self.planner_specific_queues = [multiprocessing.JoinableQueue() for _ in range(actor_processes)]

        self.planner_processes = [
            OpenravePlannerProcess(copy.deepcopy(config), self.episode_generation_queue, self.episode_results_queue,
                                   self.planner_specific_queues [i])
            for i in range(actor_processes)
        ]

        # start all the actor processes
        for p in self.planner_processes:
            p.start()

    def generate_transitions(self, queries):
        # use collectors to generate queries
        for q in queries:
            self.episode_generation_queue.put(q)

        self.episode_generation_queue.join()

        transitions = []
        for i in range(len(queries)):
            episode_transitions_list = self.episode_results_queue.get()
            transitions.extend(episode_transitions_list)

        return transitions

    def end(self):
        message = (1, )
        self._post_private_message(message, self.planner_specific_queues)
        time.sleep(10)
        for p in self.planner_processes:
            p.terminate()
        time.sleep(10)

    @staticmethod
    def _post_private_message(message, queues):
        for queue in queues:
            queue.put(message)
        for queue in queues:
            queue.join()


planner_manager = OpenravePlannerManager(config)


def create_dagger_transitions(all_train_episodes):
    plans_counter = sum([len(e[0][1])-1 for e in all_train_episodes])
    print 'need to plan for {} states'.format(plans_counter)
    queries = []
    for e in all_train_episodes:
        status = e[0][0]
        states = [[0.0] + list(s[0]) for s in e[0][1]]
        goal_state = e[2][0][-1]
        if status == 1:
            # max length - plan for the middle state
            middle_state_index = (len(states)-1) / 2
            queries.append((states[middle_state_index], goal_state))
        elif status == 2:
            # collision - plan for next to last state
            queries.append((states[len(states) - 2], goal_state))
    result = planner_manager.generate_transitions(queries)
    return result


with tf.Session(
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
        )
) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        # run train for one epoch
        print 'starting epoch {}'.format(epoch)
        for raw_train_batch in train_batcher:
            train_batch = zip(*raw_train_batch)
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
        for raw_test_batch in test_batcher:
            test_batch = zip(*raw_test_batch)
            break
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
        # copy the online weights to the offline policy
        actor_trained_weights = network.get_actor_weights(sess, is_online=True)
        rollout_manager.set_policy_weights(actor_trained_weights, is_online=False)
        # do trajectory evaluations
        eval_result = train_trajectory_eval.eval(current_global_step, config['model']['dagger_episodes_per_epoch'], is_train=True, return_episodes=True)
        train_transition_data_loader.add_extra_transitions(create_dagger_transitions(eval_result[-1]))
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
        
        eval_result = test_trajectory_eval.eval(current_global_step, config['test']['number_of_episodes'], is_train=False)
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
        
        test_summary_writer.flush()
        test_results.append((epoch, test_episodes, test_successful_episodes))
        # save the model
        if epoch % config['general']['save_every_epochs'] == 0:
            latest_saver.save(sess, os.path.join(saver_dir, 'last_iteration'), global_step=epoch)

        rate = test_successful_episodes / float(test_episodes)
        if rate > best_success_rate:
            best_model_path = best_saver.save(sess, os.path.join(saver_dir, 'best'), global_step=global_step)
            print 'old best rate: {} new best rate: {}'.format(best_success_rate, rate)
            best_success_rate = rate
            best_epoch = epoch
        else:
            print 'current rate is: {}, best model is still at epoch {} with rate: {}'.format(
                rate, best_epoch, best_success_rate)
        print 'done epoch {} of {}'.format(epoch, epochs)

    # load best model, without training and save the test results
    best_saver.restore(sess, best_model_path)
    eval_result = test_trajectory_eval.eval(-1, config['test']['number_of_episodes'], is_train=False)
    test_episodes = eval_result[0]
    test_successful_episodes = eval_result[1]
    test_collision_episodes = eval_result[2]
    test_max_len_episodes = eval_result[3]
    print_state('validation episodes', test_episodes, test_successful_episodes, test_collision_episodes,
                test_max_len_episodes)

    with open(os.path.join(test_completed_trajectories_dir, 'final_status.txt'), 'w') as final_message_file:
        validation_rate = test_successful_episodes / float(test_episodes)
        final_message_file.write('final validation rate is {}'.format(validation_rate))
        final_message_file.flush()

    test_results.append((-1, test_episodes, test_successful_episodes))

    rollout_manager.end()

test_results_file = os.path.join(test_completed_trajectories_dir, 'test_results.test_results_pkl')
with bz2.BZ2File(test_results_file, 'w') as compressed_file:
    pickle.dump(test_results, compressed_file)
