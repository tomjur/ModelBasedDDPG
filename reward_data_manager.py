import time
import datetime
import numpy as np
import random
import os
import multiprocessing
import Queue
import pickle
import bz2
from image_cache import ImageCache


class UnzipperProcess(multiprocessing.Process):
    def __init__(self, number_of_unzippers, files_queue, result_queue, unzipper_specific_queue):
        multiprocessing.Process.__init__(self)
        self.number_of_unzippers = number_of_unzippers
        self.files_queue = files_queue
        self.result_queue = result_queue
        self.unzipper_specific_queue = unzipper_specific_queue

    def run(self):
        while True:
            try:
                if self.result_queue.empty():
                    next_file = self.files_queue.get(block=True, timeout=1)
                    with open(next_file, 'r') as source_file:
                        result = pickle.load(source_file)
                    self.result_queue.put(result)
                    self.result_queue.task_done()
                else:
                    time.sleep(1)
            except Queue.Empty:
                time.sleep(1)
                pass
            try:
                # only option is to break
                task_type = self.unzipper_specific_queue.get(block=True, timeout=0.001)
                break
            except Queue.Empty:
                pass


class UnzipperIterator:
    def __init__(self, number_of_unzippers, files):
        self.number_of_unzippers = number_of_unzippers
        self.files = files

        self.unzipper_specific_queues = [multiprocessing.JoinableQueue() for _ in range(number_of_unzippers)]
        self.files_queue = multiprocessing.JoinableQueue()
        self.results_queue = multiprocessing.JoinableQueue()

        self.unzippers = [UnzipperProcess(
            number_of_unzippers, self.files_queue, self.results_queue, self.unzipper_specific_queues[i])
            for i in range(number_of_unzippers)]

        for u in self.unzippers:
            u.start()

    def __iter__(self):
        random.shuffle(self.files)
        # put all the files
        for f in self.files:
            self.files_queue.put(f)
        # when ready - collect the zipped files
        for i in range(len(self.files)):
            unzipped = self.results_queue.get()
            yield unzipped

    def end(self):
        for u in self.unzippers:
            u.terminate()
        time.sleep(10)


class RewardDataLoader:
    def __init__(self, data_dir, status_to_read, number_of_unzippers=None):
        assert os.path.exists(data_dir)
        self.cache_dir = data_dir.replace('supervised_data', 'supervised_data_cache')
        if not os.path.exists(self.cache_dir):
            self._create_cache(data_dir, self.cache_dir)

        self.files = [os.path.join(self.cache_dir, f)
                      for f in os.listdir(self.cache_dir)
                      if f.endswith(".pkl") and f.startswith('{}_'.format(status_to_read))]

        self.files_iterator = None
        if number_of_unzippers is not None:
            self.files_iterator = UnzipperIterator(number_of_unzippers, self.files)

    @staticmethod
    def _create_cache(data_dir, cache_dir):
        print('creating cache for {} in {}'.format(data_dir, cache_dir))
        os.makedirs(cache_dir)
        files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
        for f in files:
            destination_file = os.path.join(cache_dir, f)
            print('caching {}'.format(f))
            with bz2.BZ2File(os.path.join(data_dir, f), 'r') as compressed_file:
                data = pickle.load(compressed_file)
                with open(destination_file, 'w') as cache_file:
                    pickle.dump(data, cache_file)
        print('done creating cache for {} in {}'.format(data_dir, cache_dir))

    def __iter__(self):
        random.shuffle(self.files)
        if self.files_iterator is None:
            for f in self.files:
                with open(f, 'r') as source_file:
                    yield pickle.load(source_file)
        else:
            for content in self.files_iterator:
                yield content

    def stop(self):
        if self.files_iterator is not None:
            self.files_iterator.end()


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
    def __init__(self, data_dir, free_class_batch_size, oversample_goal, oversample_collision,
                 shuffle_batch_multiplier=2, number_of_unzippers=None):
        self.data_dir = data_dir
        self.oversample_goal = oversample_goal
        self.oversample_collision = oversample_collision

        # load data
        self.all_collisions = self._load_all(RewardDataLoader(data_dir, 2, None))
        self.all_goals = self._load_all(RewardDataLoader(data_dir, 3, None))
        self.free_transitions_iterator = RewardDataLoader(data_dir, 1, number_of_unzippers=number_of_unzippers)
        # self._describe_data(self.free_transitions_iterator) # TODO: UNCOMMENT THIS

        inner_batcher = Batcher(self.free_transitions_iterator, free_class_batch_size * shuffle_batch_multiplier, True)
        self.batcher = Batcher(inner_batcher, free_class_batch_size, False)

    def _describe_data(self, free_transitions_iterator):
        free_count = 0
        for tuple_list in free_transitions_iterator:
            free_count += len(tuple_list)
        collision_count = len(self.all_collisions)
        goal_count = len(self.all_goals)
        all_count = free_count + collision_count + goal_count

        print('data dir: {}'.format(self.data_dir))
        print('free: {} ({})'.format(free_count, float(free_count) / all_count))
        print('collision: {} ({})'.format(collision_count, float(collision_count) / all_count))
        print('goal: {} ({})'.format(goal_count, float(goal_count) / all_count))
        print('')

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

    def stop(self):
        self.free_transitions_iterator.stop()


def get_image_cache(config):
    image_cache = None
    scenario = config['general']['scenario']
    if 'vision' in scenario:
        params_dir = os.path.abspath(os.path.join(os.getcwd(), "scenario_params", scenario))
        image_cache = ImageCache(params_dir)
    return image_cache


def get_train_and_test_datasets(config):
    number_of_unzippers = config['general']['number_of_unzippers']
    batch_size = config['model']['batch_size']
    oversample_goal = config['reward']['oversample_goal']
    oversample_collision = config['reward']['oversample_collision']
    scenario = config['general']['scenario']
    base_data_dir = os.path.join('data', 'supervised_data', scenario + '_by_status')
    train_data_dir = os.path.join(base_data_dir, 'train')
    test_data_dir = os.path.join(base_data_dir, 'test')
    train = Oversampler(train_data_dir, batch_size, oversample_goal, oversample_collision,
                        number_of_unzippers=number_of_unzippers)
    print("Loaded Train")
    test = Oversampler(test_data_dir, batch_size, oversample_goal, oversample_collision,
                       number_of_unzippers=number_of_unzippers)
    print("Loaded Test")
    return train, test


def get_batch_and_labels(batch, image_cache):
    all_start_joints = []
    all_actions = []
    all_status = []
    all_images = None
    if image_cache is not None:
        all_images = []
    for i in range(len(batch)):
        if image_cache is None:
            workspace_id = None
            start_joints, goal_joints, action, next_joints, reward, terminated, status = batch[i]
        else:
            workspace_id, start_joints, goal_joints, action, next_joints, reward, terminated, status = batch[i]
        all_start_joints.append(start_joints[1:])
        all_actions.append(action[1:])
        all_status.append(status)
        if image_cache is not None:
            image = image_cache.items[workspace_id].np_array
            all_images.append(image)
    return [all_start_joints, all_actions, all_images], all_status