import pickle
import os
import yaml
import datetime
import bz2
import numpy as np

from data_collector import CollectorProcess, DataCollector
from image_cache import ImageCache


class VisionRandomStepCollectorProcess(CollectorProcess):
    def _get_tuple(self, query_params=None):
        assert query_params is not None
        workspace_id = query_params[0]
        full_workspace_path = query_params[1]

        openrave_manager = self.openrave_interface.openrave_manager
        #  set the obstacles
        openrave_manager.set_params(full_workspace_path)

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
        return workspace_id, start_joints, goal_joints, random_action, next_joints, reward, terminated, status


class VisionRandomStepDataCollector(DataCollector):
    def _get_queue_size(self, number_of_threads):
        return 100*number_of_threads

    def _get_collector(self, config, queued_data_points, collector_specific_queue, params_file=None):
        return VisionRandomStepCollectorProcess(
            config, queued_data_points, self.results_queue, collector_specific_queue,
            query_parameters_queue=self.query_parameters_queue, init_rl_interface=True
        )


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

# scenario = 'vision'
scenario = 'vision_harder'

# number_of_samples_per_workspace = 50
# samples_per_file = 10
# threads = 10
# results_dir = 'supervised_data_vision_temp_to_delete'
# scenario = 'vision_harder_small'

number_of_samples_per_workspace = 1000
samples_per_file = 1000
# threads = 100
threads = 10
results_dir = 'supervised_data_vision_harder'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

params_dir = os.path.abspath(os.path.expanduser('~/ModelBasedDDPG/scenario_params/{}/'.format(scenario)))
image_cache = ImageCache(params_dir)
collection_queries = []
workspace_ids = []
for cache_item in image_cache.items.values():
    collection_queries.extend([(cache_item.workspace_id, cache_item.full_filename)] * number_of_samples_per_workspace)
    workspace_ids.append(cache_item.workspace_id)

data_collector = VisionRandomStepDataCollector(config, threads, query_parameters=collection_queries)
collected = 0

aa = datetime.datetime.now()

params_ids_to_tuples = {workspace_id: [] for workspace_id in workspace_ids}
params_ids_to_offset = {workspace_id: 0 for workspace_id in workspace_ids}
while collected < len(collection_queries):
    a = datetime.datetime.now()
    current_buffer = data_collector.generate_samples(samples_per_file)
    b = datetime.datetime.now()
    print 'data collection took: {}'.format(b - a)
    print_status_dist(current_buffer)

    for t in current_buffer:
        workspace_id = t[0]
        real_tuple = t[1:]
        params_ids_to_tuples[workspace_id].append(real_tuple)

    collected += len(current_buffer)

    for workspace_id in params_ids_to_tuples:
        workspace_buffer = params_ids_to_tuples[workspace_id]
        if len(workspace_buffer) >= samples_per_file:
            current_buffer = workspace_buffer[:samples_per_file]
            current_offset = params_ids_to_offset[workspace_id]
            dump_path = os.path.join(results_dir, '{}_{}.pkl'.format(workspace_id, current_offset))
            compressed_file = bz2.BZ2File(dump_path, 'w')
            pickle.dump(current_buffer, compressed_file)
            compressed_file.close()
            params_ids_to_offset[workspace_id] = current_offset + samples_per_file
            params_ids_to_tuples[workspace_id] = params_ids_to_tuples[workspace_id][samples_per_file:]
        assert len(params_ids_to_tuples[workspace_id]) < samples_per_file

bb = datetime.datetime.now()
print 'collection took: {}'.format(bb - aa)

for workspace_id in params_ids_to_tuples:
    assert len(params_ids_to_tuples[workspace_id]) == 0

data_collector.end()
