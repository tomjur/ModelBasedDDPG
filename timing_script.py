import numpy as np
import cPickle as pickle
import bz2
import os
import yaml
import datetime
import tensorflow as tf


from ddpg_main import overload_config_by_scenario
from image_cache import ImageCache
from openrave_manager import OpenraveManager
from openrave_rl_interface import OpenraveRLInterface
from potential_point import PotentialPoint
from pre_trained_reward import PreTrainedReward
from network import Network

file_to_use = '/home/tom/ModelBasedDDPG/imitation_data/vision/test/0.path_pkl'
repeat = 2
query_limit = 100

vision_model_directory = 'vision_speed_model/'
vision_model_file = os.path.join(vision_model_directory, 'best-386040')
network_name_prefix = '1922'


# read the config
config_path = os.path.join(os.getcwd(), 'config/config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)
    assert config['general']['scenario'] == 'vision'
    overload_config_by_scenario(config)
    print('------------ Config ------------')
    print(yaml.dump(config))


def get_queries():
    with bz2.BZ2File(file_to_use, 'r') as compressed_file:
        loaded_data = pickle.load(compressed_file)
    result = []
    for traj, pose_traj, workspace_id in loaded_data:
        start_joints = traj[0]
        goal_joints = traj[-1]
        t = [start_joints, goal_joints, workspace_id, traj]
        result.append(t)
    return result


queries = get_queries()
queries = queries[:query_limit]
image_cache = ImageCache(config['general']['params_file'], create_images=True)


def run_network_single(sess, openrave_rl_interface, network, trajectory, workspace_image):
    # start the new query
    current_joints, goal_joints, steps_required_for_motion_plan = openrave_rl_interface.start_specific(trajectory)
    current_joints = current_joints[1:]
    # compute the maximal number of steps to execute
    max_steps = int(steps_required_for_motion_plan * config['general']['max_path_slack'])
    goal_pose = openrave_rl_interface.openrave_manager.get_target_pose(goal_joints)
    goal_joints = goal_joints[1:]
    # set the start state
    status = None
    # the trajectory data structures to return
    total_episode_time = None
    for j in range(max_steps):
        # do a single step prediction
        start_transition_time = datetime.datetime.now()
        action = network.predict_action(
            [current_joints], [workspace_image], [goal_pose], [goal_joints], sess, use_online_network=False
        )[0]
        end_transition_time = datetime.datetime.now()
        time_diff = end_transition_time - start_transition_time
        if total_episode_time is None:
            total_episode_time = time_diff
        else:
            total_episode_time += time_diff
        # make an environment step
        openrave_step = np.insert(action, 0, [0.0])
        next_joints, current_reward, is_terminal, status = openrave_rl_interface.step(openrave_step)
        current_joints = next_joints[1:]
        # break if needed
        if is_terminal:
            break

    # return the trajectory along with query info
    return total_episode_time, status


def run_network():
    result = None
    # load pretrained model
    reward_model_name = config['model']['reward_model_name']
    pre_trained_reward = PreTrainedReward(reward_model_name, config)
    openrave_rl_interface = OpenraveRLInterface(config)

    # generate graph:
    network = Network(config, is_rollout_agent=False, pre_trained_reward=pre_trained_reward,
                      name_prefix=network_name_prefix)
    best_saver = tf.train.Saver(max_to_keep=2, save_relative_paths=vision_model_directory)
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))) as sess:
        pre_trained_reward.load_weights(sess)
        best_saver.restore(sess, vision_model_file)

        for _, _, workspace_id, trajectory in queries:
            cache_item = image_cache.items[workspace_id]
            workspace_image = cache_item.np_array
            openrave_rl_interface.openrave_manager.set_params(cache_item.full_filename)
            for i in range(repeat):
                time_diff, status = run_network_single(sess, openrave_rl_interface, network, trajectory, workspace_image)
                if result is None:
                    result = time_diff
                else:
                    result += time_diff
    return result


network_result = run_network()
print network_result


def run_motion_planner():
    result = None
    openrave_manager = OpenraveManager(config['openrave_rl']['segment_validity_step'], PotentialPoint.from_config(config))
    for start_joints, goal_joints, workspace_id, _ in queries:
        params_file_path = image_cache.items[workspace_id].full_filename
        openrave_manager.set_params(params_file_path)
        for i in range(repeat):
            start_time = datetime.datetime.now()
            traj = openrave_manager.plan(start_joints, goal_joints, None)
            # assert traj is not None
            end_time = datetime.datetime.now()
            time_diff = end_time - start_time
            if result is None:
                result = time_diff
            else:
                result += time_diff
    return result


planner_result = run_motion_planner()
print planner_result