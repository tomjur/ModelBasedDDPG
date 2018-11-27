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
    def __init__(self, config, generate_data_queue, result_queue, actor_specific_queue):
        multiprocessing.Process.__init__(self)
        self.generate_data_queue = generate_data_queue
        self.result_queue = result_queue
        self.actor_specific_queue = actor_specific_queue
        self.config = config
        # members to set at runtime
        self.openrave_interface = None

    def _get_tuple(self):
        start_joints, goal_joints, image, _, trajectory = self.openrave_interface.start_new_random(
            None, return_traj=True)
        trajectory_poses = [
            self.openrave_interface.openrave_manager.get_potential_points_poses(step) for step in trajectory]
        goal_pose = self.openrave_interface.openrave_manager.get_target_pose(goal_joints)
        return trajectory, image, goal_pose, trajectory_poses

    def _run_main_loop(self):
        while True:
            try:
                # wait 1 second for a trajectory request
                _ = self.generate_data_queue.get(block=True, timeout=1)
                self.result_queue.put(self._get_tuple())
                self.generate_data_queue.task_done()
            except Queue.Empty:
                pass
            try:
                next_actor_specific_task = self.actor_specific_queue.get(block=True, timeout=0.001)
                task_type = next_actor_specific_task[0]
                if task_type == 1:
                    # need to terminate
                    self.actor_specific_queue.task_done()
                    break
            except Queue.Empty:
                pass

    def run(self):
        # write pid to file
        actor_id = os.getpid()
        actor_file = os.path.join(os.getcwd(), 'actor_{}.sh'.format(actor_id))
        with open(actor_file, 'w') as f:
            f.write('kill -9 {}'.format(actor_id))

        workspace_params = WorkspaceParams.load_from_file('params_simple.pkl')
        self.openrave_interface = OpenraveRLInterface(self.config, workspace_params)
        self._run_main_loop()


class ImitationDataCollector:
    def __init__(self, config, number_of_threads, ):
        self.number_of_threads = number_of_threads
        self.data_generation_queue = multiprocessing.JoinableQueue()
        self.results_queue = multiprocessing.Queue()
        self.actor_specific_queues = [
            multiprocessing.JoinableQueue() for _ in range(self.number_of_threads)
        ]

        self.actors = [
            ActorProcess(
                copy.deepcopy(config), self.data_generation_queue, self.results_queue, self.actor_specific_queues[i]
            )
            for i in range(self.number_of_threads)
        ]
        # start all the processes
        for a in self.actors:
            a.start()

    def generate_samples(self, number_of_samples):
        # place in queue
        for i in range(number_of_samples):
            self.data_generation_queue.put(i)

        self.data_generation_queue.join()

        result_buffer = []
        while number_of_samples:
            number_of_samples -= 1
            result_buffer.append(self.results_queue.get())

        return result_buffer

    def end(self):
        message = (1, )
        self._post_private_message(message)

    def _post_private_message(self, message):
        for actor_queue in self.actor_specific_queues:
            actor_queue.put(message)
        for actor_queue in self.actor_specific_queues:
            actor_queue.join()
