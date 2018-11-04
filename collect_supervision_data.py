import pickle
import os
import yaml
import datetime

from random_data_collector import RandomDataCollector


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

config['openrave_rl']['challenging_trajectories_only'] = False


number_of_samples = 1000000
samples_per_file = 5000
threads = 5

data_collector = RandomDataCollector(config, threads)
results_dir = 'supervised_data'
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
    pickle.dump(current_buffer, open(dump_path, 'w'))
    collected += len(current_buffer)
data_collector.end()



