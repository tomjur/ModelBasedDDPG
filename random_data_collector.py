import copy
import numpy as np
import multiprocessing
import Queue

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
        start_joints, goal_joints, _ = self.openrave_interface.start_new_random(None)
        random_action = np.random.uniform(-1.0, 1.0, len(start_joints) - 1)
        random_action /= np.linalg.norm(random_action)
        random_action = np.array([0.0] + list(random_action))
        next_joints, reward, terminated, status = self.openrave_interface.step(random_action)
        return start_joints, goal_joints, random_action, next_joints, reward, terminated, status

    def _run_main_loop(self):
        while True:
            try:
                # wait 1 second for a trajectory request
                _ = self.generate_data_queue.get(block=True, timeout=1)
                transition = self._get_tuple()
                self.result_queue.put(transition)
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
        workspace_params = WorkspaceParams.load_from_file('params.pkl')
        self.openrave_interface = OpenraveRLInterface(self.config, workspace_params)
        self._run_main_loop()


class RandomDataCollector:
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
