import pickle
import os
import yaml
import datetime
import bz2
import numpy as np

from data_collector import CollectorProcess, DataCollector


class RandomStepCollectorProcess(CollectorProcess):
    def _get_tuple(self, query_params=None):
        openrave_manager = self.openrave_interface.openrave_manager

        # find free start and goal joints
        start_joints = openrave_manager.get_random_joints({0: 0.0})
        while not openrave_manager.is_valid(start_joints):
            start_joints = openrave_manager.get_random_joints({0: 0.0})

        goal_joints = openrave_manager.get_random_joints({0: 0.0})
        while not openrave_manager.is_valid(goal_joints):
            goal_joints = openrave_manager.get_random_joints({0: 0.0})

        # set fake trajectory with just start and goal, make sure the interface does not verify
        traj = [start_joints, goal_joints]
        self.openrave_interface.start_specific(traj, verify_traj=False)

        # take a random action
        random_action = np.random.uniform(-1.0, 1.0, len(start_joints) - 1)
        random_action /= np.linalg.norm(random_action)
        random_action = np.array([0.0] + list(random_action))
        next_joints, reward, terminated, status = self.openrave_interface.step(random_action)

        # the result contains also the workspace used
        return start_joints, goal_joints, random_action, next_joints, reward, terminated, status


class RandomStepDataCollector(DataCollector):
    def _get_queue_size(self, number_of_threads):
        return 100*number_of_threads

    def _get_collector(self, config, queued_data_points, collector_specific_queue, params_file=None):
        return RandomStepCollectorProcess(
            config, queued_data_points, self.results_queue, collector_specific_queue, params_file,
            init_rl_interface=True)


def print_status_dist(current_buffer):
    status = [t[-1] for t in current_buffer]
    total = len(status)
    for i in range(1, 4):
        count = sum([s == i for s in status])
        print '{}: {} ({})'.format(i, count, float(count) / total)


# read the config
config_path = os.path.join(os.getcwd(), 'config/config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)
    print('------------ Config ------------')
    print(yaml.dump(config))

# number_of_samples = 30
# samples_per_file = 10
# threads = 10
# results_dir = 'supervised_data_temp_to_delete'
# scenario = 'hard'

number_of_samples = 1000000
samples_per_file = 5000
threads = 100
results_dir = 'supervised_data'
scenario = 'hard'

params_file = os.path.abspath(os.path.expanduser(
        os.path.join('~/ModelBasedDDPG/scenario_params', scenario, 'params.pkl')))

data_collector = RandomStepDataCollector(config, threads, params_file)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
collected = 0
while collected < number_of_samples:
    a = datetime.datetime.now()
    current_buffer = data_collector.generate_samples(samples_per_file)
    b = datetime.datetime.now()
    print 'data collection took: {}'.format(b - a)
    print_status_dist(current_buffer)
    dump_path = os.path.join(results_dir, 'temp_data_{}.pkl'.format(collected))
    compressed_file = bz2.BZ2File(dump_path, 'w')
    pickle.dump(current_buffer, compressed_file)
    compressed_file.close()
    collected += len(current_buffer)
data_collector.end()
