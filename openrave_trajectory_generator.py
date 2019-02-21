import numpy as np
from openrave_manager import OpenraveManager
from potential_point import PotentialPoint


class OpenraveTrajectoryGenerator:

    def __init__(self, config):
        self.action_step_size = config['openrave_rl']['action_step_size']
        self.goal_sensitivity = config['openrave_rl']['goal_sensitivity']
        self.challenging_trajectories_only = config['openrave_planner']['challenging_trajectories_only']
        self.planner_iterations_start = config['openrave_planner']['planner_iterations_start']
        self.planner_iterations_increase = config['openrave_planner']['planner_iterations_increase']
        self.planner_iterations_decrease = config['openrave_planner']['planner_iterations_decrease']
        self.max_planner_iterations = self.planner_iterations_start

        self.openrave_manager = OpenraveManager(
            config['openrave_rl']['segment_validity_step'], PotentialPoint.from_config(config))

    def is_below_goal_sensitivity(self, start_joints, goal_joints):
        start_pose = self.openrave_manager.get_target_pose(start_joints)
        goal_pose = self.openrave_manager.get_target_pose(goal_joints)
        pose_distance = np.linalg.norm(np.array(start_pose) - np.array(goal_pose))
        return pose_distance < self.goal_sensitivity

    def find_random_trajectory_single_try(self):
        # select at random
        start_joints = self.openrave_manager.get_random_joints({0: 0.0})
        goal_joints = self.openrave_manager.get_random_joints({0: 0.0})
        # if the start and goal are too close, re-sample
        if self.is_below_goal_sensitivity(start_joints, goal_joints):
            return None
        start_pose = self.openrave_manager.get_target_pose(start_joints)
        goal_pose = self.openrave_manager.get_target_pose(goal_joints)
        # valid region:
        if not self._is_valid_region(start_pose, goal_pose):
            return None
        # trajectories that must cross an obstacle
        if self.challenging_trajectories_only and not self._is_challenging(start_pose, goal_pose):
            return None
        traj = self.openrave_manager.plan(start_joints, goal_joints, self.max_planner_iterations)
        return traj

    def find_random_trajectory(self):
        # lower_size = 0.0  # when doing curriculum, this this is the lowest possible distance between start and goal
        while True:
            traj = self.find_random_trajectory_single_try()
            if traj is None:
                # if failed to plan, give more power
                self.max_planner_iterations += self.planner_iterations_increase
            elif self.max_planner_iterations > self.planner_iterations_start + self.planner_iterations_decrease:
                # if plan was found, maybe we need less iterations
                self.max_planner_iterations -= self.planner_iterations_decrease
                return self.split_trajectory(traj, self.action_step_size)

    @staticmethod
    def _is_valid_region(start_pose, goal_pose):
        return start_pose[1] > 0.0 and goal_pose[1] > 0.0

    def _is_challenging(self, start_pose, goal_pose):
        workspace_params = self.openrave_manager.loaded_params
        if workspace_params is None or workspace_params.number_of_obstacles == 0:
            return True
        # check if the distance from any obstacle is smaller that the start-goal-distance
        start = np.array(start_pose)
        goal = np.array(goal_pose)
        start_goal_distance = np.linalg.norm(start - goal)
        for i in range(workspace_params.number_of_obstacles):
            obstacle = np.array(
                [workspace_params.centers_position_x[i], workspace_params.centers_position_z[i]]
            )
            start_obstacle_distance = np.linalg.norm(start - obstacle)
            goal_obstacle_distance = np.linalg.norm(goal - obstacle)
            if start_obstacle_distance < start_goal_distance and goal_obstacle_distance < start_goal_distance:
                return True
        # all tests failed
        return False

    @staticmethod
    def split_trajectory(trajectory, action_step_size):
        res = [tuple(trajectory[0])]
        for i in range(len(trajectory) - 1):
            current_step = np.array(trajectory[i])
            next_step = np.array(trajectory[i + 1])
            difference = next_step - current_step
            difference_norm = np.linalg.norm(difference)
            if difference_norm < action_step_size:
                # if smaller than allowed step just append the next step
                res.append(tuple(trajectory[i + 1]))
                continue
            scaled_step = (action_step_size / difference_norm) * difference
            steps = []
            for alpha in range(int(np.floor(difference_norm / action_step_size))):
                processed_step = current_step + (1 + alpha) * scaled_step
                steps.append(processed_step)
            # we probably have a leftover section, append it to res
            last_step_difference = np.linalg.norm(steps[-1] - next_step)
            if last_step_difference > 0.0:
                steps.append(next_step)
            # append to path
            res += [tuple(s) for s in steps]
        return res
