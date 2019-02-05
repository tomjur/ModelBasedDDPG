import cPickle as pickle
import bz2
import numpy as np
import matplotlib.pyplot as plt

# groups_to_test_results_files = {
#     'DDPG': [
#         '/home/tom/paper_results/simple_scenario/trajectories/ddpg4/test_results.test_results_pkl',
#         '/home/tom/paper_results/simple_scenario/trajectories/ddpg5/test_results.test_results_pkl',
#         '/home/tom/paper_results/simple_scenario/trajectories/ddpg6/test_results.test_results_pkl',
#     ],
#     'DDPG-MP (no expert)': [
#         '/home/tom/paper_results/simple_scenario/trajectories/ddpgmp4/test_results.test_results_pkl',
#         '/home/tom/paper_results/simple_scenario/trajectories/ddpgmp5/test_results.test_results_pkl',
#         '/home/tom/paper_results/simple_scenario/trajectories/ddpgmp6/test_results.test_results_pkl',
#     ],
# }
# title = 'Simple scenario'

# groups_to_test_results_files = {
#     'DDPG': [
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg1/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg2/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg3/test_results.test_results_pkl',
#     ],
#     'DDPG+HER': [
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg_her1/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg_her2/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg_her3/test_results.test_results_pkl',
#     ],
#     'DDPG-MP (full)': [
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp1/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp2/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp3/test_results.test_results_pkl',
#     ],
#     'DDPG-MP (no expert)': [
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score1/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score2/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score3/test_results.test_results_pkl',
#     ],
#     'DDPG-MP+HER (no expert)': [
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score_her_k1_1/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score_her_k1_2/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score_her_k1_3/test_results.test_results_pkl',
#     ],
# }
# title = 'Hard scenario'

# groups_to_test_results_files = {
#     'DDPG': [
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg1/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg2/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg3/test_results.test_results_pkl',
#     ],
#     'DDPG+HER': [
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg_her1/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg_her2/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpg_her3/test_results.test_results_pkl',
#     ],
#     'DDPG-MP (our method)': [
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp1/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp2/test_results.test_results_pkl',
#         '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp3/test_results.test_results_pkl',
#     ],
#     # 'DDPG-MP (no expert)': [
#     #     '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score1/test_results.test_results_pkl',
#     #     '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score2/test_results.test_results_pkl',
#     #     '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score3/test_results.test_results_pkl',
#     # ],
#     # 'DDPG-MP+HER (no expert)': [
#     #     '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score_her_k1_1/test_results.test_results_pkl',
#     #     '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score_her_k1_2/test_results.test_results_pkl',
#     #     '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score_her_k1_3/test_results.test_results_pkl',
#     # ],
# }
# title = 'Hard scenario - comparing DDPG-MP to baselines'

groups_to_test_results_files = {
    # 'DDPG': [
    #     '/home/tom/paper_results/hard_scenario/trajectories/ddpg1/test_results.test_results_pkl',
    #     '/home/tom/paper_results/hard_scenario/trajectories/ddpg2/test_results.test_results_pkl',
    #     '/home/tom/paper_results/hard_scenario/trajectories/ddpg3/test_results.test_results_pkl',
    # ],
    # 'DDPG+HER': [
    #     '/home/tom/paper_results/hard_scenario/trajectories/ddpg_her1/test_results.test_results_pkl',
    #     '/home/tom/paper_results/hard_scenario/trajectories/ddpg_her2/test_results.test_results_pkl',
    #     '/home/tom/paper_results/hard_scenario/trajectories/ddpg_her3/test_results.test_results_pkl',
    # ],
    'DDPG-MP (our method)': [
        '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp1/test_results.test_results_pkl',
        '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp2/test_results.test_results_pkl',
        '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp3/test_results.test_results_pkl',
    ],
    'DDPG-MP (no expert)': [
        '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score1/test_results.test_results_pkl',
        '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score2/test_results.test_results_pkl',
        '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score3/test_results.test_results_pkl',
    ],
    'DDPG-MP+HER (no expert)': [
        '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score_her_k1_1/test_results.test_results_pkl',
        '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score_her_k1_2/test_results.test_results_pkl',
        '/home/tom/paper_results/hard_scenario/trajectories/ddpgmp_no_score_her_k1_3/test_results.test_results_pkl',
    ],
}
title = 'Hard scenario - comparing exploration strategies'


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
    # collect from each file
    all_results = {}
    longest_res = None
    for f in test_results_files:
        episodes_res, success_rate_res = load_file_as_series(f)
        # once 1.0 reached consider the suffix as 1.0 also
        i = 0
        for i, r in enumerate(success_rate_res):
            if r == 1.0:
                break
        prefix_size = i + 1
        episodes_res = episodes_res[:prefix_size]
        success_rate_res = success_rate_res[:prefix_size]
        all_results[f] = success_rate_res
        # see which is the longest:
        if longest_res is None or episodes_res[-1] > longest_res[-1]:
            longest_res = episodes_res
    # merge
    data = None
    for f in all_results:
        success_rate_res = all_results[f]
        # need to add 1.0 elements
        count = len(longest_res) - len(success_rate_res)
        if count > 0:
            success_rate_res = success_rate_res + [1.0] * count
        new_data = np.expand_dims(np.array(success_rate_res), axis=0)
        if data is None:
            data = new_data
        else:
            data = np.concatenate((data, new_data), axis=0)

    return longest_res, data


def plot_group(episode_axis, data, ax, label, color):
    # get data bounds
    data_mean = np.mean(data, axis=0)
    ax.plot(episode_axis, data_mean, lw=2, label=label, color=color)
    if data.shape[0] > 1:
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        ax.fill_between(episode_axis, data_max, data_min, facecolor=color, alpha=0.5)
    # if data.shape[0] > 1:
    #     data_std = np.std(data, axis=0)
    #     data_min = data_mean - data_std
    #     data_max = data_mean + data_std
    #     ax.fill_between(episode_axis, data_max, data_min, facecolor=color, alpha=0.5)


fig, ax = plt.subplots(1)
for i, label in enumerate(groups_to_test_results_files.keys()):
    group_axis, group_data = load_several_files(groups_to_test_results_files[label])
    plot_group(group_axis, group_data, ax, label, colors[i])
ax.set_title(title)
ax.legend(loc='lower right')
ax.set_xlabel('train episodes')
ax.set_ylabel('success rate')
plt.show()
