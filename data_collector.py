import copy
import multiprocessing
import Queue
import time

from openrave_rl_interface import OpenraveRLInterface
from workspace_generation_utils import WorkspaceParams


class CollectorProcess(multiprocessing.Process):
    def __init__(self, config, queued_data_points, result_queue, collector_specific_queue, params_file=None,
                 query_parameters_queue=None):
        multiprocessing.Process.__init__(self)
        self.result_queue = result_queue
        self.collector_specific_queue = collector_specific_queue
        self.config = config
        self.params_file = params_file
        self.query_parameters_queue = query_parameters_queue
        # members to set at runtime
        self.openrave_interface = None
        self.queued_data_points = queued_data_points

    def _get_tuple(self, query_params=None):
        pass

    def _run_main_loop(self):
        while True:
            try:
                next_collector_specific_task = self.collector_specific_queue.get(block=True, timeout=0.001)
                task_type = next_collector_specific_task[0]
                if task_type == 1:
                    # need to terminate
                    self.collector_specific_queue.task_done()
                    break
            except Queue.Empty:
                pass
            if self.result_queue.qsize() < self.queued_data_points:
                if self.query_parameters_queue is None:
                    self.result_queue.put(self._get_tuple())
                else:
                    try:
                        query_parameters = self.query_parameters_queue.get(block=True, timeout=0.001)
                        self.result_queue.put(self._get_tuple(query_parameters))
                    except Queue.Empty:
                        pass
                time.sleep(1.0)

    def run(self):
        workspace_params = None
        if self.params_file is not None:
            workspace_params = WorkspaceParams.load_from_file(self.params_file)
        self.openrave_interface = OpenraveRLInterface(self.config, workspace_params)
        self._run_main_loop()


class DataCollector:
    def __init__(self, config, number_of_threads, params_file=None, query_parameters=None):
        self.number_of_threads = number_of_threads
        self.results_queue = multiprocessing.Queue()
        self.query_parameters_queue = None
        if query_parameters is not None:
            # put all the query parameters in the designated queue
            self.query_parameters_queue = multiprocessing.Queue(maxsize=len(query_parameters))
            for t in query_parameters:
                self.query_parameters_queue.put(t, True)
        self.collector_specific_queues = [
            multiprocessing.JoinableQueue() for _ in range(self.number_of_threads)
        ]

        queue_size = self._get_queue_size(number_of_threads)
        self.collectors = [
            self._get_collector(
                copy.deepcopy(config), queue_size, self.collector_specific_queues[i], params_file
            )
            for i in range(self.number_of_threads)
        ]
        # start all the processes
        for c in self.collectors:
            c.start()

    def _get_queue_size(self, number_of_threads):
        pass

    def _get_collector(self, config, queued_data_points, collector_specific_queue, params_file=None):
        pass

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
        for c in self.collectors:
            c.terminate()
        time.sleep(10)

    def _post_private_message(self, message):
        for collector_queue in self.collector_specific_queues:
            collector_queue.put(message)
        for collector_queue in self.collector_specific_queues:
            collector_queue.join()
