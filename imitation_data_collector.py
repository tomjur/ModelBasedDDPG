import copy
import multiprocessing
import Queue
import time

from openrave_rl_interface import OpenraveRLInterface
from workspace_generation_utils import WorkspaceParams


class ActorProcess(multiprocessing.Process):
    def __init__(self, config, required_trajectories, result_queue, actor_specific_queue, params_file=None):
        multiprocessing.Process.__init__(self)
        self.result_queue = result_queue
        self.actor_specific_queue = actor_specific_queue
        self.config = config
        self.params_file = params_file
        # members to set at runtime
        self.openrave_interface = None
        self.required_trajectories = required_trajectories

    def _get_tuple(self):
        start_joints, goal_joints, _, trajectory = self.openrave_interface.start_new_random(None, return_traj=True)
        trajectory_poses = [
            self.openrave_interface.openrave_manager.get_potential_points_poses(step) for step in trajectory]
        return trajectory, trajectory_poses

    def _run_main_loop(self):
        while True:
            try:
                next_actor_specific_task = self.actor_specific_queue.get(block=True, timeout=0.001)
                task_type = next_actor_specific_task[0]
                if task_type == 1:
                    # need to terminate
                    self.actor_specific_queue.task_done()
                    break
            except Queue.Empty:
                pass
            if self.result_queue.qsize() < self.required_trajectories:
                self.result_queue.put(self._get_tuple())

    def run(self):
        workspace_params = None
        if self.params_file is not None:
            workspace_params = WorkspaceParams.load_from_file(self.params_file)
        self.openrave_interface = OpenraveRLInterface(self.config, workspace_params)
        self._run_main_loop()


class ImitationDataCollector:
    def __init__(self, config, number_of_threads, params_file=None):
        self.number_of_threads = number_of_threads
        self.results_queue = multiprocessing.Queue()
        self.actor_specific_queues = [
            multiprocessing.JoinableQueue() for _ in range(self.number_of_threads)
        ]

        self.actors = [
            ActorProcess(
                copy.deepcopy(config), self.number_of_threads*2, self.results_queue, self.actor_specific_queues[i],
                params_file
            )
            for i in range(self.number_of_threads)
        ]
        # start all the processes
        for a in self.actors:
            a.start()

    def generate_samples(self, number_of_samples):
        result_buffer = []
        while number_of_samples:
            number_of_samples -= 1
            result_buffer.append(self.results_queue.get())

        return result_buffer

    def end(self):
        message = (1, )
        self._post_private_message(message)
        time.sleep(10)

    def _post_private_message(self, message):
        for actor_queue in self.actor_specific_queues:
            actor_queue.put(message)
        for actor_queue in self.actor_specific_queues:
            actor_queue.join()
