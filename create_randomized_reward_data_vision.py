import bz2
import os
import random
import cPickle as pickle


source_dir = os.path.expanduser('~/ModelBasedDDPG/supervised_data/vision/train')
new_dir = os.path.expanduser('~/ModelBasedDDPG/supervised_data/vision_shuffled/train')
samples_per_new_file = 1000

log_file = open(os.path.join(new_dir, 'creation_log.txt'), 'w')


def write_chunk(entire_buffer, file_index):
    random.shuffle(entire_buffer)
    data_to_write = entire_buffer[:samples_per_new_file]
    entire_buffer = entire_buffer[samples_per_new_file:]
    filename = '{}.pkl'.format(file_index)
    compressed_file = bz2.BZ2File(os.path.join(new_dir, filename), 'w')
    pickle.dump(data_to_write, compressed_file)
    compressed_file.close()
    return entire_buffer


def write_to_log(message):
    print message
    log_file.write('{}{}'.format(message, os.linesep))
    log_file.flush()


if not os.path.exists(new_dir):
    os.makedirs(new_dir)

output_file_index = 0
input_file_index = 0
data = []
entire_data_length = 0
workspace_serial_to_count = {}
for dirpath, dirnames, filenames in os.walk(source_dir):
    for filename in filenames:
        name_parts = filename.split('_')
        workspace_id = '{}_{}.pkl'.format(name_parts[0], name_parts[1])
        compressed_file = bz2.BZ2File(os.path.join(source_dir, filename), 'r')
        new_data = pickle.load(compressed_file)
        compressed_file.close()
        new_data = [tuple([workspace_id] + list(t))for t in new_data]
        data.extend(new_data)
        current_data_len = len(new_data)
        entire_data_length += current_data_len
        workspace_serial = int(name_parts[0])
        if workspace_serial not in workspace_serial_to_count:
            workspace_serial_to_count[workspace_serial] = current_data_len
        else:
            workspace_serial_to_count[workspace_serial] += current_data_len
        while len(data) > samples_per_new_file * samples_per_new_file:
            data = write_chunk(data, output_file_index)
            output_file_index += 1
        input_file_index += 1
        print 'done with {}. input files processed {} output files produced {}'.format(
            filename, input_file_index, output_file_index)

while len(data) > 0:
    if len(data) < samples_per_new_file:
        write_to_log(
            'warning: last file is not complete, choose a file size that is a multiplier of the number of elements!')
        write_to_log('number of elements: {}, file size {}'.format(entire_data_length, samples_per_new_file))
    data = write_chunk(data, output_file_index)
    output_file_index += 1
    print 'writing leftover data. output files produced {}'.format(output_file_index)

for workspace_serial in workspace_serial_to_count:
    write_to_log('workspace {} has {} entries'.format(workspace_serial, workspace_serial_to_count[workspace_serial]))

write_to_log('number of workspaces: {}'.format(len(workspace_serial_to_count)))
write_to_log('minimal workspace serial is {}'.format(min(workspace_serial_to_count.keys())))
write_to_log('maximal workspace serial is {}'.format(max(workspace_serial_to_count.keys())))
write_to_log('avg count per workspace is {}'.format(
    sum(workspace_serial_to_count.values()) / len(workspace_serial_to_count)))
log_file.close()
print 'done'
