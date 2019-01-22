import cPickle as pickle
import bz2
import numpy as np
import matplotlib.pyplot as plt

groups_to_test_results_files = {
    'DDPG': [
        '/home/tom/paper_results/simple_scenario/trajectories/ddpg1/test_results.test_results_pkl',
        '/home/tom/paper_results/simple_scenario/trajectories/ddpg2/test_results.test_results_pkl',
        '/home/tom/paper_results/simple_scenario/trajectories/ddpg3/test_results.test_results_pkl',
    ]
}
title = 'simple scenario'
colors = ['blue', 'green', 'red', 'yellow', 'teal']


def load_file_as_series(test_results_file):
    with bz2.BZ2File(test_results_file, 'r') as compressed_file:
        test_results = pickle.load(compressed_file)
    episodes_res = []
    success_rate_res = []
    for t in test_results:
        global_step, episodes, test_successful_episodes, test_collision_episodes,test_max_len_episodes, test_mean_reward = t
        if global_step == -1:
            continue
        episodes_res.append(episodes)
        success_rate_res.append(float(test_successful_episodes) / (test_successful_episodes + test_collision_episodes + test_max_len_episodes))
    return episodes_res, success_rate_res


def load_several_files(test_results_files):
    episode_axis = None
    data = None
    for f in test_results_files:
        episodes_res, success_rate_res = load_file_as_series(f)
        if episode_axis is None:
            episode_axis = episodes_res
            data = np.expand_dims(np.array(success_rate_res), axis=0)
        else:
            assert episodes_res == episode_axis
            new_data = np.expand_dims(np.array(success_rate_res), axis=0)
            data = np.concatenate((data, new_data), axis=0)
    return episode_axis, data


def plot_group(episode_axis, data, ax, label, color):
    # get data bounds
    data_mean = np.mean(data, axis=0)
    ax.plot(episode_axis, data_mean, lw=2, label=label, color=color)
    if data.shape[0] > 1:
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        ax.fill_between(episode_axis, data_max, data_min, facecolor=color, alpha=0.5)


fig, ax = plt.subplots(1)
for i, label in enumerate(groups_to_test_results_files.keys()):
    group_axis, group_data = load_several_files(groups_to_test_results_files[label])
    plot_group(group_axis, group_data, ax, label, colors[i])
ax.set_title(title)
ax.legend(loc='lower right')
ax.set_xlabel('train episodes')
ax.set_ylabel('success rate')
plt.show()
