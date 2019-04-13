import cPickle as pickle
import bz2
import os
import yaml
import datetime

from data_collector import CollectorProcess, DataCollector


class TrajectoryCollectorProcess(CollectorProcess):
    def _get_tuple(self, query_params=None):
        trajectory = self.openrave_trajectory_generator.find_random_trajectory()
        trajectory_poses = [
            self.openrave_trajectory_generator.openrave_manager.get_potential_points_poses(step) for step in trajectory]
        return trajectory, trajectory_poses


class ImitationDataCollector(DataCollector):
    def _get_queue_size(self, number_of_threads):
        return 2 * number_of_threads

    def _get_collector(self, config, queued_data_points, collector_specific_queue, params_file=None):
        return TrajectoryCollectorProcess(
            config, queued_data_points, self.results_queue, collector_specific_queue, params_file,
            init_trajectory_collector=True
        )


# read the config
config_path = os.path.join(os.getcwd(), 'config/config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)
    print('------------ Config ------------')
    print(yaml.dump(config))

config['openrave_planner'] = {
    'challenging_trajectories_only': True,
    'planner_iterations_start': 100,
    'planner_iterations_increase': 10,
    'planner_iterations_decrease': 1,
}

# scenario = 'simple'
scenario = 'hard'
params_file = os.path.abspath(os.path.expanduser(
    os.path.join('~/ModelBasedDDPG/scenario_params', scenario, 'params.pkl')))

# number_of_trajectories = 12
# trajectories_per_file = 4
# threads = 2
# results_dir = 'imitation_data_to_delete'

number_of_trajectories = 100000
trajectories_per_file = 1000
threads = 100
results_dir = 'imitation_data'

data_collector = ImitationDataCollector(config, threads, params_file)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
collected = 0
while collected < number_of_trajectories:
    a = datetime.datetime.now()
    current_buffer = data_collector.generate_samples(trajectories_per_file)
    b = datetime.datetime.now()
    print 'data collection took: {}'.format(b - a)
    dump_path = os.path.join(results_dir, 'temp_data_{}.path_pkl'.format(collected))
    compressed_file = bz2.BZ2File(dump_path, 'w')
    pickle.dump(current_buffer, compressed_file)
    compressed_file.close()
    collected += len(current_buffer)
data_collector.end()
