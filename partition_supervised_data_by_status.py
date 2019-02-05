import os
import bz2
import cPickle as pickle

# source_dir = '/home/tom/ModelBasedDDPG/supervised_data/vision/train'
# source_dir = '/home/tom/ModelBasedDDPG/supervised_data/vision/test'
# source_dir = '/home/tom/ModelBasedDDPG/supervised_data/vision_harder_shuffled/test'
source_dir = '/home/tom/ModelBasedDDPG/supervised_data/vision_harder_shuffled/train'
# target_dir = '/home/tom/ModelBasedDDPG/supervised_data/vision_by_status/train'
# target_dir = '/home/tom/ModelBasedDDPG/supervised_data/vision_by_status/test'
# target_dir = '/home/tom/ModelBasedDDPG/supervised_data/vision_harder_by_status/test'
target_dir = '/home/tom/ModelBasedDDPG/supervised_data/vision_harder_by_status/train'
file_max_size = 1000

assert os.path.isdir(source_dir)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)


data_by_status = {1: [], 2: [], 3: []}
index_counters = {1: 0, 2: 0, 3: 0}


def write_data(force_write=False):
    for status in data_by_status:
        if len(data_by_status[status]) > file_max_size or force_write:
            target_file_path = os.path.join(target_dir, '{}_{}.pkl'.format(status, index_counters[status]))
            print 'writing status file for {} to {}'.format(status, target_file_path)
            data_to_write = data_by_status[status][:file_max_size]
            with bz2.BZ2File(target_file_path, 'w') as compressed_file:
                pickle.dump(data_to_write, compressed_file)
            print 'done writing {}'.format(target_file_path)
            # advance the data to not write the same data twice
            data_by_status[status] = data_by_status[status][file_max_size:]
            index_counters[status] += 1


for dirpath, dirnames, filenames in os.walk(source_dir):
    data_files = [f for f in filenames if f.endswith('.pkl')]

    for i, data_file in enumerate(data_files):
        full_path = os.path.join(source_dir, data_file)
        print 'starting {}: {}'.format(i, full_path)
        with bz2.BZ2File(full_path, 'r') as compressed_file:
            content = pickle.load(compressed_file)
        for transition in content:
            data_by_status[transition[-1]].append(transition)
        print 'done reading {}: {}'.format(i, full_path)

        write_data()
    write_data(True)


