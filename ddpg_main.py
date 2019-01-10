import os
import random
import datetime
import bz2
import tensorflow as tf
import yaml
import time
import numpy as np

from episode_editor import EpisodeEditor
from hindsight_policy import HindsightPolicy
from network import Network
from replay_buffer import ReplayBuffer
from rollout_manager import FixedRolloutManager
from summaries_collector import SummariesCollector
from trajectory_eval import TrajectoryEval
from pre_trained_reward import PreTrainedReward
from workspace_generation_utils import *


def run_for_config(config, print_messages):
    # set the name of the model
    model_name = config['general']['name']
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    model_name = now + '_' + model_name if model_name is not None else now

    # openrave_interface = OpenraveRLInterface(config, None)
    random_seed = config['general']['random_seed']
    np.random.seed(random_seed)
    random.seed(random_seed)
    tf.set_random_seed(random_seed)

    # where we save all the outputs
    working_dir = os.getcwd()
    saver_dir = os.path.join(working_dir, 'models', model_name)
    if not os.path.exists(saver_dir):
        os.makedirs(saver_dir)
    config_copy_path = os.path.join(working_dir, 'models', model_name, 'config.yml')
    summaries_dir = os.path.join(working_dir, 'tensorboard', model_name)
    completed_trajectories_dir = os.path.join(working_dir, 'trajectories', model_name)

    # load pretrained model if required
    pre_trained_reward = None
    if config['model']['use_reward_model'] is not None:
        reward_model_name = config['model']['reward_model_name']
        pre_trained_reward = PreTrainedReward(reward_model_name, config)

    # generate graph:
    network = Network(config, is_rollout_agent=False, pre_trained_reward=pre_trained_reward)

    def unpack_state_batch(state_batch):
        joints = [state[0] for state in state_batch]
        poses = {p.tuple: [state[1][p.tuple] for state in state_batch] for p in network.potential_points}
        jacobians = None
        return joints, poses, jacobians

    def score_for_hindsight(augmented_buffer):
        # unzip
        goal_pose_list, goal_joints_list, workspace_image_list, current_state_list, action_used_list, _, is_goal_list,\
        __ = zip(*augmented_buffer)
        # make one hot status vector:
        is_goal_one_hot_list = np.zeros((len(is_goal_list), 3), dtype=np.float32)
        for i in range(len(is_goal_list)):
            if is_goal_list[i]:
                is_goal_one_hot_list[i, 2] = 1.0  # mark as goal transition
            else:
                is_goal_one_hot_list[i, 0] = 1.0  # mark as free transition
        # unpack current and next state
        current_joints, _, __ = unpack_state_batch(current_state_list)

        fake_rewards, _ = pre_trained_reward.make_prediction(
            sess, current_joints, goal_joints_list, action_used_list, goal_pose_list,
            all_transition_labels=is_goal_one_hot_list
        )
        return list(fake_rewards)

    # initialize replay memory
    replay_buffer = ReplayBuffer(config)
    hindsight_policy = HindsightPolicy(config, replay_buffer, score_for_hindsight)

    # save model
    saver = tf.train.Saver(max_to_keep=4, save_relative_paths=saver_dir)
    yaml.dump(config, open(config_copy_path, 'w'))
    summaries_collector = SummariesCollector(summaries_dir, model_name)
    rollout_manager = FixedRolloutManager(config)
    trajectory_eval = TrajectoryEval(config, rollout_manager, completed_trajectories_dir)

    test_results = []

    def update_model(sess, global_step):
        batch_size = config['model']['batch_size']
        gamma = config['model']['gamma']
        replay_buffer_batch = replay_buffer.sample_batch(batch_size)

        goal_pose, goal_joints, workspace_image, current_state, action, reward, terminated, next_state = \
            replay_buffer_batch

        current_joints, current_poses, current_jacobians = unpack_state_batch(current_state)
        next_joints, next_poses, next_jacobians = unpack_state_batch(next_state)

        # get the predicted q value of the next state (action is taken from the target policy)
        next_state_action_target_q = network.predict_policy_q(
            next_joints, workspace_image, goal_pose, goal_joints, sess, use_online_network=False
        )

        # compute critic label
        q_label = np.expand_dims(np.squeeze(np.array(reward)) + np.multiply(
            np.multiply(1 - np.array(terminated), gamma),
            np.squeeze(next_state_action_target_q)
        ), 1)
        max_label = np.max(q_label)
        min_label = np.min(q_label)
        limit = 1.0 / (1.0 - gamma)
        if max_label > limit:
            print 'out of range max label: {} limit: {}'.format(max_label, limit)
        if min_label < -limit:
            print 'out of range min label: {} limit: {}'.format(min_label, limit)

        # # step to use for debug:
        # network.debug_all(current_joints, workspace_image, goal_pose, goal_joints, action, q_label, sess)

        # train critic given the targets
        critic_optimization_summaries, _ = network.train_critic(
            current_joints, workspace_image, goal_pose, goal_joints, action, q_label, sess
        )

        # train actor
        actor_optimization_summaries, _ = network.train_actor(
            current_joints, workspace_image, goal_pose, goal_joints, sess
        )

        # update target networks
        network.update_target_networks(sess)

        result = [critic_optimization_summaries, actor_optimization_summaries, ]
        return result

    def print_state(prefix, episodes, successful_episodes, collision_episodes, max_len_episodes):
        if not print_messages:
            return
        print '{}: {}: finished: {}, successful: {} ({}), collision: {} ({}), max length: {} ({})'.format(
            datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), prefix, episodes,
            successful_episodes, float(successful_episodes) / episodes, collision_episodes,
            float(collision_episodes) / episodes, max_len_episodes, float(max_len_episodes) / episodes
        )

    def process_example_trajectory(episode_example_trajectory, episode_agent_trajectory):
        _, __, ___, ____, goal_pose, goal_joints, workspace_image = episode_agent_trajectory
        example_trajectory, example_trajectory_poses = episode_example_trajectory
        example_trajectory = [j[1:] for j in example_trajectory]
        # goal reached always
        status = 3
        # get the states (joints, poses, jacobians), for now, ignore the jacobians.
        states = [(example_trajectory[i], example_trajectory_poses[i], None) for i in range(len(example_trajectory))]
        # compute the actions by normalized difference between steps
        actions = [np.array(example_trajectory[i+1]) - np.array(example_trajectory[i])
                   for i in range(len(example_trajectory)-1)]
        actions = [a / max(np.linalg.norm(a), 0.00001) for a in actions]
        rewards = [None] * len(actions)
        return status, states, actions, rewards, goal_pose, goal_joints, workspace_image

    def do_test(sess, best_model_global_step, best_model_test_success_rate):
        rollout_manager.set_policy_weights(network.get_actor_weights(sess, is_online=False), is_online=False)
        eval_result = trajectory_eval.eval(global_step)
        test_episodes = eval_result[0]
        test_successful_episodes = eval_result[1]
        test_collision_episodes = eval_result[2]
        test_max_len_episodes = eval_result[3]
        test_mean_reward = eval_result[4]
        if print_messages:
            print_state('test', test_episodes, test_successful_episodes, test_collision_episodes,
                        test_max_len_episodes)
            print('test mean total reward {}'.format(test_mean_reward))
        summaries_collector.write_test_episode_summaries(
            sess, global_step, test_episodes, test_successful_episodes, test_collision_episodes,
            test_max_len_episodes
        )
        test_results.append((global_step, episodes, test_successful_episodes, test_collision_episodes,
                             test_max_len_episodes, test_mean_reward))
        # see if best
        rate = test_successful_episodes / float(test_episodes)
        if best_model_test_success_rate < rate:
            if print_messages:
                print 'new best model found at step {}'.format(global_step)
                print 'old success rate {} new success rate {}'.format(best_model_test_success_rate, rate)
            is_best = True
            best_model_global_step = global_step
            best_model_test_success_rate = rate
        else:
            is_best = False
            if print_messages:
                print 'best model still at step {}'.format(best_model_global_step)
        return is_best, best_model_global_step, best_model_test_success_rate

    regular_episode_editor = EpisodeEditor(config['model']['alter_episode'], pre_trained_reward)
    motion_planner_episode_editor = EpisodeEditor(2, pre_trained_reward)

    with tf.Session(
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
            )
    ) as sess:
        sess.run(tf.global_variables_initializer())
        if pre_trained_reward is not None:
            pre_trained_reward.load_weights(sess)
        network.update_target_networks(sess)

        global_step = 0
        total_episodes = episodes = successful_episodes = collision_episodes = max_len_episodes = 0
        best_model_global_step, best_model_test_success_rate = -1, -1.0
        for update_index in range(config['general']['updates_cycle_count']):
            # collect data
            a = datetime.datetime.now()
            rollout_manager.set_policy_weights(network.get_actor_weights(sess, is_online=True), is_online=True)
            episodes_per_update = config['general']['episodes_per_update']
            episode_results = rollout_manager.generate_episodes(episodes_per_update, True)
            episodes_agent_trajectory, episodes_times, episodes_example_trajectory = zip(*episode_results)

            # alter the episodes based on reward model
            altered_episodes = regular_episode_editor.process_episodes(episodes_agent_trajectory, sess)

            # process example episodes for failed interactions
            altered_motion_planner_episodes = []
            failed_motion_planner_trajectories = config['model']['failed_motion_planner_trajectories']
            if failed_motion_planner_trajectories > 0:
                # take a small number of failed motion plans
                failed_episodes_indices = [i for i in range(len(altered_episodes)) if altered_episodes[i][0] != 3]
                failed_episodes_indices = failed_episodes_indices[:failed_motion_planner_trajectories]
                motion_planner_episodes = [
                    process_example_trajectory(episodes_example_trajectory[i], altered_episodes[i])
                    for i in failed_episodes_indices
                ]
                altered_motion_planner_episodes = motion_planner_episode_editor.process_episodes(
                    motion_planner_episodes, sess)

            # add to replay buffer
            hindsight_policy.append_to_replay_buffer(list(altered_episodes) + list(altered_motion_planner_episodes))

            # compute times
            total_find_trajectory_time = None
            total_rollout_time = None
            for episode_times in episodes_times:
                # update the times
                find_trajectory_time, rollout_time = episode_times
                if total_find_trajectory_time is None:
                    total_find_trajectory_time = find_trajectory_time
                else:
                    total_find_trajectory_time += find_trajectory_time
                if total_rollout_time is None:
                    total_rollout_time = rollout_time
                else:
                    total_rollout_time += rollout_time

            # compute counters
            for altered_episode in altered_episodes:
                status = altered_episode[0]
                total_episodes += 1
                episodes += 1
                if status == 1:
                    max_len_episodes += 1
                elif status == 2:
                    collision_episodes += 1
                elif status == 3:
                    successful_episodes += 1

            b = datetime.datetime.now()
            print 'data collection took: {}'.format(b-a)
            print 'find trajectory took: {}'.format(total_find_trajectory_time)
            print 'rollout time took: {}'.format(total_rollout_time)
            print_state('train', episodes, successful_episodes, collision_episodes, max_len_episodes)

            # do updates
            if replay_buffer.size() > config['model']['batch_size']:
                a = datetime.datetime.now()
                for _ in range(config['general']['model_updates_per_cycle']):
                    summaries = update_model(sess, global_step)
                    if global_step % config['general']['write_train_summaries'] == 0:
                        summaries_collector.write_train_episode_summaries(
                            sess, global_step, episodes, successful_episodes, collision_episodes, max_len_episodes
                        )
                        summaries_collector.write_train_optimization_summaries(summaries, global_step)
                    global_step += 1
                b = datetime.datetime.now()
                print 'update took: {}'.format(b - a)

            # test if needed
            if update_index % config['test']['test_every_cycles'] == 0:
                is_best, best_model_global_step, best_model_test_success_rate = do_test(
                    sess, best_model_global_step, best_model_test_success_rate)
                if is_best:
                    saver.save(sess, os.path.join(saver_dir, 'best'), global_step=global_step)
            if update_index % config['general']['save_model_every_cycles'] == 0:
                saver.save(sess, os.path.join(saver_dir, 'last_iteration'), global_step=global_step)

        # final test at the end
        is_best, best_model_global_step, best_model_test_success_rate = do_test(
            sess, best_model_global_step, best_model_test_success_rate)
        if is_best:
            saver.save(sess, os.path.join(saver_dir, 'best'), global_step=global_step)
    last_message = 'best model stats at step {} has success rate of {}'.format(
        best_model_global_step, best_model_test_success_rate)
    print last_message

    with open(os.path.join(completed_trajectories_dir, 'final_status.txt'), 'w') as f:
        f.write(last_message)
        f.flush()

    test_results_file = os.path.join(completed_trajectories_dir, 'test_results.test_results_pkl')
    with bz2.BZ2File(test_results_file, 'w') as compressed_file:
        pickle.dump(test_results, compressed_file)

    rollout_manager.end()
    return test_results


def overload_config_by_scenario(config):
    scenario = config['general']['scenario']
    config['general']['trajectory_directory'] = os.path.abspath(os.path.expanduser(
        os.path.join('~/ModelBasedDDPG/imitation_data/', scenario)))
    config['general']['params_file'] = os.path.abspath(os.path.expanduser(
        os.path.join('~/ModelBasedDDPG/scenario_params', scenario, 'params.pkl')))
    config['model']['consider_image'] = scenario is 'vision'
    config['model']['reward_model_name'] = scenario


if __name__ == '__main__':
    # disable tf warning
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # read the config
    config_path = os.path.join(os.getcwd(), 'config/config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        overload_config_by_scenario(config)
        print('------------ Config ------------')
        print(yaml.dump(config))

    run_for_config(config, print_messages=True)
