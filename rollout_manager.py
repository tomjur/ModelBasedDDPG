import os
import copy
import random
import numpy as np
import tensorflow as tf
import multiprocessing
import Queue
import datetime

from network import Network
from openrave_rl_interface import OpenraveRLInterface
from workspace_generation_utils import WorkspaceParams


class QueryCollectorProcess(multiprocessing.Process):
    def __init__(self, config, result_queue, collector_specific_queue):
        multiprocessing.Process.__init__(self)
        self.result_queue = result_queue
        self.collector_specific_queue = collector_specific_queue
        self.config = config
        self.openrave_interface = None

    def _run_main_loop(self):
        episodes_per_update = self.config['general']['episodes_per_update']
        required_trajectories = episodes_per_update * 10
        while True:
            try:
                next_collector_specific_task = self.collector_specific_queue.get(block=True, timeout=0.1)
                task_type = next_collector_specific_task[0]
                # can only terminate
                self.collector_specific_queue.task_done()
                break
            except Queue.Empty:
                pass
            if self.result_queue.qsize() < required_trajectories:
                trajectory, start_joints, goal_joints = self.openrave_interface.find_random_trajectory()
                trajectory_poses = [
                    self.openrave_interface.openrave_manager.get_potential_points_poses(step) for step in trajectory]
                assert len(trajectory) == len(trajectory_poses)
                result = trajectory, start_joints, goal_joints, trajectory_poses
                self.result_queue.put(result)

    def run(self):
        # write pid to file
        collector_id = os.getpid()
        collector_file = os.path.join(os.getcwd(), 'collector_{}.sh'.format(collector_id))
        with open(collector_file, 'w') as f:
            f.write('kill -9 {}'.format(collector_id))

        params_file = self.config['general']['params_file']
        workspace_params = None
        if params_file is not None:
            workspace_params = WorkspaceParams.load_from_file(params_file)
        self.openrave_interface = OpenraveRLInterface(self.config, workspace_params)
        self._run_main_loop()


class ActorProcess(multiprocessing.Process):
    def __init__(self, config, generate_episode_queue, result_queue, actor_specific_queue):
        multiprocessing.Process.__init__(self)
        self.generate_episode_queue = generate_episode_queue
        self.result_queue = result_queue
        self.actor_specific_queue = actor_specific_queue
        self.config = config
        # members to set at runtime
        self.openrave_interface = None
        self.actor = None

    def _get_sampled_action(self, action):
        totally_random = np.random.binomial(1, self.config['model']['random_action_probability'], 1)[0]
        if totally_random:
            # take a completely random action
            result = np.random.uniform(-1.0, 1.0, np.shape(action))
        else:
            # modify existing step
            result = action + np.random.normal(0.0, self.config['model']['random_noise_std'], np.shape(action))
        # normalize
        result /= np.linalg.norm(result)
        return result

    def _compute_state(self, joints):
        openrave_manager = self.openrave_interface.openrave_manager
        # get the poses
        poses = openrave_manager.get_potential_points_poses(joints)
        # get the jacobians
        jacobians = None
        # preprocess the joints (remove first joint value)
        joints = joints[1:]
        return joints, poses, jacobians

    def _run_episode(self, sess, query_params, is_train):
        trajctory, start_joints, goal_joints, trajectory_poses = query_params
        # the trajectory data structures to return
        start_episode_time = datetime.datetime.now()
        states = []
        actions = []
        rewards = []
        # start the new query
        current_joints, goal_joints, workspace_image, steps_required_for_motion_plan = \
            self.openrave_interface.start_specific(trajctory, start_joints, goal_joints)
        goal_pose = self.openrave_interface.openrave_manager.get_target_pose(goal_joints)
        goal_joints = goal_joints[1:]
        # set the start state
        current_state = self._compute_state(current_joints)
        states.append(current_state)
        start_rollout_time = datetime.datetime.now()
        # the result of the episode
        status = None
        # compute the maximal number of steps to execute
        max_steps = int(steps_required_for_motion_plan * self.config['general']['max_path_slack'])
        for j in range(max_steps):
            # do a single step prediction
            # current_poses = None if current_state[1] is None else {
            #     p.tuple: [current_state[1][p.tuple]] for p in self.actor.potential_points
            # }
            # current_jacobians = None if current_state[2] is None else {
            #     p.tuple: [current_state[2][p.tuple]] for p in self.actor.potential_points
            # }
            action_mean = self.actor.predict_action(
                [current_state[0]], [workspace_image], [goal_pose], [goal_joints], sess, use_online_network=True
            )[0]
            sampled_action = self._get_sampled_action(action_mean) if is_train else action_mean
            # make an environment step
            openrave_step = np.insert(sampled_action, 0, [0.0])
            next_joints, current_reward, is_terminal, status = self.openrave_interface.step(openrave_step)
            # set a new current state
            current_state = self._compute_state(next_joints)
            # update return data structures
            states.append(current_state)
            actions.append(sampled_action)
            rewards.append(current_reward)
            # break if needed
            if is_terminal:
                break
        # return the trajectory along with query info
        assert len(states) == len(actions) + 1
        assert len(states) == len(rewards) + 1
        end_episode_time = datetime.datetime.now()
        find_trajectory_time = start_rollout_time - start_episode_time
        rollout_time = end_episode_time-start_rollout_time
        return status, states, actions, rewards, goal_pose, goal_joints, workspace_image, find_trajectory_time, \
               rollout_time, trajctory, trajectory_poses

    def _run_main_loop(self, sess):
        while True:
            try:
                # wait 1 second for a trajectory request
                next_episode_request = self.generate_episode_queue.get(block=True, timeout=1)
                query_params = next_episode_request[0]
                is_train = next_episode_request[1]
                path = self._run_episode(sess, query_params, is_train)
                self.result_queue.put(path)
                self.generate_episode_queue.task_done()
            except Queue.Empty:
                pass
            try:
                next_actor_specific_task = self.actor_specific_queue.get(block=True, timeout=0.001)
                task_type = next_actor_specific_task[0]
                if task_type == 0:
                    # need to init the actor, called once.
                    assert self.actor is None
                    # on init, we only create a part of the graph (online actor model)
                    self.actor = Network(self.config, is_rollout_agent=True)
                    sess.run(tf.global_variables_initializer())
                    self.actor_specific_queue.task_done()
                elif task_type == 1:
                    # need to terminate
                    self.actor_specific_queue.task_done()
                    break
                elif task_type == 2:
                    # update the weights
                    new_weights = next_actor_specific_task[1]
                    self.actor.set_actor_online_weights(sess, new_weights)
                    self.actor_specific_queue.task_done()
            except Queue.Empty:
                pass

    def run(self):
        # write pid to file
        actor_id = os.getpid()
        actor_file = os.path.join(os.getcwd(), 'actor_{}.sh'.format(actor_id))
        with open(actor_file, 'w') as f:
            f.write('kill -9 {}'.format(actor_id))

        params_file = self.config['general']['params_file']
        workspace_params = None
        if params_file is not None:
            workspace_params = WorkspaceParams.load_from_file(params_file)
        self.openrave_interface = OpenraveRLInterface(self.config, workspace_params)

        # config = tf.ConfigProto(
        #     device_count = {'GPU': 0}
        # )
        # self.session = tf.Session(config=config)
        # self.session.run(tf.initialize_all_variables())

        with tf.Session(
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.config['general']['actor_gpu_usage'])
                )
        ) as sess:
            self._run_main_loop(sess)


class RolloutManager:
    def __init__(self, config, collector_processes=None, actor_processes=None, fixed_queries=None):
        self.episode_generation_queue = multiprocessing.JoinableQueue()
        self.episode_results_queue = multiprocessing.Queue()
        self.query_results_queue = multiprocessing.Queue()
        if collector_processes is None:
            collector_processes = config['general']['collector_processes']
        if actor_processes is None:
            actor_processes = config['general']['actor_processes']
        self.fixed_queries = fixed_queries

        self.collector_specific_queue = [multiprocessing.JoinableQueue() for _ in range(collector_processes)]
        self.actor_specific_queues = [multiprocessing.JoinableQueue() for _ in range(actor_processes)]

        self.collectors = [
            QueryCollectorProcess(
                copy.deepcopy(config), self.query_results_queue, self.collector_specific_queue[i]
            )
            for i in range(collector_processes)
        ]
        self.actors = [
            ActorProcess(copy.deepcopy(config), self.episode_generation_queue, self.episode_results_queue,
                         self.actor_specific_queues[i])
            for i in range(actor_processes)
        ]
        # start all the collector processes
        for c in self. collectors:
            c.start()

        # start all the actor processes
        for a in self.actors:
            a.start()
        # for every actor process, post a message to initialize the actor network
        for actor_queue in self.actor_specific_queues:
            actor_queue.put((0, ))
            actor_queue.join()

    def generate_episodes(self, number_of_episodes, is_train):
        if self.fixed_queries is None:
            # use collectors to generate queries
            for i in range(number_of_episodes):
                # get a query
                query = self.query_results_queue.get()
                message = (query, is_train)
                # place in queue
                self.episode_generation_queue.put(message)
        else:
            #  use the fixed queries
            random.shuffle(self.fixed_queries)
            for (traj, image, goal_pose) in self.fixed_queries[:number_of_episodes]:
                start_joints = traj[0]
                goal_joints = traj[-1]
                query = (traj, start_joints, goal_joints)
                message = (query, is_train)
                # place in queue
                self.episode_generation_queue.put(message)

        self.episode_generation_queue.join()

        episodes = []
        while number_of_episodes:
            number_of_episodes -= 1
            episodes.append(self.episode_results_queue.get())

        return episodes

    def set_policy_weights(self, weights):
        message = (2, weights)
        self._post_private_message(message, self.actor_specific_queues)

    def end(self):
        message = (1, )
        self._post_private_message(message, self.actor_specific_queues)
        self._post_private_message(message, self.collector_specific_queue)

    @staticmethod
    def _post_private_message(message, queues):
        for queue in queues:
            queue.put(message)
        for queue in queues:
            queue.join()
