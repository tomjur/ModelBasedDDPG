import os
import copy
import numpy as np
import tensorflow as tf
import multiprocessing
import Queue

from network import Network
from openrave_rl_interface import OpenraveRLInterface
from workspace_generation_utils import WorkspaceParams


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

    def _run_episode(self, sess, allowed_size, is_train):
        # the trajectory data structures to return
        states = []
        actions = []
        rewards = []
        # get a new query
        current_joints, goal_joints, workspace_image, steps_required_for_motion_plan = \
            self.openrave_interface.start_new_random(allowed_size)
        goal_pose = self.openrave_interface.openrave_manager.get_target_pose(goal_joints)
        goal_joints = goal_joints[1:]
        # set the start state
        current_state = self._compute_state(current_joints)
        states.append(current_state)
        # the result of the episode
        status = None
        # compute the maximal number of steps to execute
        max_steps = int(steps_required_for_motion_plan * self.config['general']['max_path_slack'])
        for j in range(max_steps):
            # do a single step prediction
            current_poses = None if current_state[1] is None else {
                p.tuple: [current_state[1][p.tuple]] for p in self.actor.potential_points
            }
            current_jacobians = None if current_state[2] is None else {
                p.tuple: [current_state[2][p.tuple]] for p in self.actor.potential_points
            }
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
        return status, states, actions, rewards, goal_pose, goal_joints, workspace_image

    def _run_main_loop(self, sess):
        while True:
            try:
                # wait 1 second for a trajectory request
                next_episode_request = self.generate_episode_queue.get(block=True, timeout=1)
                allowed_size = next_episode_request[0]
                is_train = next_episode_request[1]
                path = self._run_episode(sess, allowed_size, is_train)
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

        workspace_params = WorkspaceParams.load_from_file('params.pkl')
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
    def __init__(self, config):
        self.episode_generation_queue = multiprocessing.JoinableQueue()
        self.results_queue = multiprocessing.Queue()
        self.actor_specific_queues = [
            multiprocessing.JoinableQueue() for _ in range(config['general']['actor_processes'])
        ]

        self.actors = [
            ActorProcess(
                copy.deepcopy(config), self.episode_generation_queue, self.results_queue, self.actor_specific_queues[i]
            )
            for i in range(config['general']['actor_processes'])
        ]
        # start all the processes
        for a in self.actors:
            a.start()
        # for every actor process, post a message to initialize the actor network
        for actor_queue in self.actor_specific_queues:
            actor_queue.put((0, ))
            actor_queue.join()

    def generate_episodes(self, number_of_episodes, allowed_size, is_train):
        # create a tuple that describes the episode generation request
        message = (allowed_size, is_train)
        # place in queue
        for i in range(number_of_episodes):
            self.episode_generation_queue.put(message)

        self.episode_generation_queue.join()

        episodes = []
        while number_of_episodes:
            number_of_episodes -= 1
            episodes.append(self.results_queue.get())

        return episodes

    def set_policy_weights(self, weights):
        message = (2, weights)
        self._post_private_message(message)

    def end(self):
        message = (1, )
        self._post_private_message(message)

    def _post_private_message(self, message):
        for actor_queue in self.actor_specific_queues:
            actor_queue.put(message)
        for actor_queue in self.actor_specific_queues:
            actor_queue.join()
