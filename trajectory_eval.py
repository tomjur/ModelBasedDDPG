import pickle
import os
from potential_point import PotentialPoint


class TrajectoryEval:

    def __init__(self, config, rollout_manager, results_directory):
        self.config = config
        self.rollout_manager = rollout_manager
        self.results_directory = results_directory
        self._make_dir(self.results_directory)
        potential_points_path = os.path.join(self.results_directory, 'potential_points.p')
        pickle.dump(PotentialPoint.from_config(config), open(potential_points_path, 'w'))

    def eval(self, global_step, allowed_distance=None):
        successful_episodes = 0
        collision_episodes = 0
        max_len_episodes = 0
        episodes = 0
        mean_total_reward = 0.0
        number_of_episodes = self.config['test']['number_of_episodes']
        episode_results = self.rollout_manager.generate_episodes(number_of_episodes, allowed_distance, False)
        for episode_result in episode_results:
            status = episode_result[0]
            states = episode_result[1]
            rewards = episode_result[3]
            goal_pose = episode_result[4]
            mean_total_reward += sum(rewards)
            # at the end of episode
            episodes += 1
            if status == 1:
                self.save_trajectory(states, goal_pose, max_len_episodes, 'max_len', global_step)
                max_len_episodes += 1
            elif status == 2:
                self.save_trajectory(states, goal_pose, collision_episodes, 'collision', global_step)
                collision_episodes += 1
            elif status == 3:
                self.save_trajectory(states, goal_pose, successful_episodes, 'success', global_step)
                successful_episodes += 1
        mean_total_reward /= number_of_episodes
        return episodes, successful_episodes, collision_episodes, max_len_episodes, mean_total_reward

    def save_trajectory(self, trajectory, goal_pose, path_index, header, global_step):
        # get the joints
        joints = [state[0] for state in trajectory]
        to_save = (goal_pose, joints)
        step_dir = os.path.join(self.results_directory, str(global_step))
        self._make_dir(step_dir)
        filename = '{}_{}.p'.format(header, path_index)
        trajectory_path = os.path.join(step_dir, filename)
        pickle.dump(to_save, open(trajectory_path, 'w'))

    @staticmethod
    def _make_dir(dir_location):
        if not os.path.exists(dir_location):
            os.makedirs(dir_location)