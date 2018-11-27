import numpy as np
from openrave_manager import OpenraveManager
from potential_point import PotentialPoint
import scipy.spatial


class OpenraveRLInterface:
    # class StepResult(enum.Enum):
    #     free_space = 1
    #     collision = 2
    #     close_to_goal = 3

    def __init__(self, config, workspace_params=None):
        self.action_step_size = config['openrave_rl']['action_step_size']
        self.goal_sensitivity = config['openrave_rl']['goal_sensitivity']
        self.planner_iterations_start = config['openrave_rl']['planner_iterations_start']
        self.planner_iterations_increase = config['openrave_rl']['planner_iterations_increase']
        self.planner_iterations_decrease = config['openrave_rl']['planner_iterations_decrease']
        self.max_planner_iterations = self.planner_iterations_start
        self.keep_alive_penalty = config['openrave_rl']['keep_alive_penalty']
        self.truncate_penalty = config['openrave_rl']['truncate_penalty']
        self.challenging_trajectories_only = config['openrave_rl']['challenging_trajectories_only']

        self.openrave_manager = OpenraveManager(
            config['openrave_rl']['segment_validity_step'], PotentialPoint.from_config(config))
        self.workspace_params = workspace_params
        if workspace_params is not None:
            self.openrave_manager.load_params(workspace_params)

        self.current_joints = None
        self.goal_joints = None
        self.start_joints = None
        self.traj = None
        self.current_shaping_index = None

    def is_below_goal_sensitivity(self, start_joints, goal_joints):
        start_pose = self.openrave_manager.get_target_pose(start_joints)
        goal_pose = self.openrave_manager.get_target_pose(goal_joints)
        pose_distance = np.linalg.norm(np.array(start_pose) - np.array(goal_pose))
        return pose_distance < self.goal_sensitivity

    def find_random_trajectory(self, allowed_start_goal_difference=None):
        lower_size = 0.0  # when doing curriculum, this this is the lowest possible distance between start and goal
        while True:
            # select at random
            start_joints = self.openrave_manager.get_random_joints({0: 0.0})
            goal_joints = self.openrave_manager.get_random_joints({0: 0.0})
            # if curriculum is enabled, change goal to match
            if allowed_start_goal_difference is not None:
                direction = np.array(goal_joints) - np.array(start_joints)
                direction_size = np.linalg.norm(direction)
                direction /= direction_size  # direction is size 1.0 now
                direction *= np.random.uniform(lower_size, allowed_start_goal_difference)  # now in valid range
                goal_joints = list(np.array(start_joints) + np.array(direction))
                goal_joints = self.openrave_manager.truncate_joints(goal_joints)
                distance = np.linalg.norm(np.array(start_joints) - np.array(goal_joints))
                assert distance <= allowed_start_goal_difference + 0.001
            # if the start and goal are too close, re-sample
            if self.is_below_goal_sensitivity(start_joints, goal_joints):
                if allowed_start_goal_difference is not None:
                    lower_size = (lower_size + allowed_start_goal_difference) / 2.0
                continue
            start_pose = self.openrave_manager.get_target_pose(start_joints)
            goal_pose = self.openrave_manager.get_target_pose(goal_joints)
            # valid region:
            if not self._is_valid_region(start_pose, goal_pose):
                continue
            # trajectories that must cross an obstacle
            if self.challenging_trajectories_only and not self._is_challenging(start_pose, goal_pose):
                continue
            traj = self.openrave_manager.plan(start_joints, goal_joints, self.max_planner_iterations)
            if traj is None:
                # if failed to plan, give more power
                self.max_planner_iterations += self.planner_iterations_increase
            elif self.max_planner_iterations > self.planner_iterations_start + self.planner_iterations_decrease:
                # if plan was found, maybe we need less iterations
                self.max_planner_iterations -= self.planner_iterations_decrease
                return self._split_trajectory(traj)

    def start_new_random(self, allowed_start_goal_difference=None, return_traj=False):
        traj = self.find_random_trajectory(allowed_start_goal_difference)
        return self.start_specific(traj, return_traj=return_traj)

    def start_specific(self, traj, return_traj=False):
        self.traj = traj
        start_joints = traj[0]
        goal_joints = traj[-1]
        # assert path is legal
        step_size = self.action_step_size + 0.00001
        for i in range(len(traj)-1):
            step_i_size = np.linalg.norm(np.array(traj[i]) - np.array(traj[i+1]))
            assert step_i_size < step_size, 'step_i_size {}'.format(step_i_size)
        steps_required_for_motion_plan = len(traj)
        self.current_joints = np.array(start_joints)
        self.start_joints = np.array(start_joints)
        self.goal_joints = np.array(goal_joints)
        self.current_shaping_index = 0
        # the agent gets the staring joints and the goal joints and the workspace image
        if return_traj:
            return start_joints, goal_joints, steps_required_for_motion_plan, traj
        else:
            return start_joints, goal_joints, steps_required_for_motion_plan

    @staticmethod
    def _is_valid_region(start_pose, goal_pose):
        return start_pose[1] > 0.0 and goal_pose[1] > 0.0

    def _is_challenging(self, start_pose, goal_pose):
        if self.workspace_params is None or self.workspace_params.number_of_obstacles == 0:
            return True
        # check if the distance from any obstacle is smaller that the start-goal-distance
        start = np.array(start_pose)
        goal = np.array(goal_pose)
        start_goal_distance = np.linalg.norm(start - goal)
        for i in range(self.workspace_params.number_of_obstacles):
            obstacle = np.array(
                [self.workspace_params.centers_position_x[i], self.workspace_params.centers_position_z[i]]
            )
            start_obstacle_distance = np.linalg.norm(start - obstacle)
            goal_obstacle_distance = np.linalg.norm(goal - obstacle)
            if start_obstacle_distance < start_goal_distance and goal_obstacle_distance < start_goal_distance:
                return True
        # all tests failed
        return False

    def step(self, joints_action):
        # compute next joints
        step = joints_action * self.action_step_size
        next_joints_before_truncate = self.current_joints + step
        next_joints = self.openrave_manager.truncate_joints(next_joints_before_truncate)

        reward = 0.0
        if self.truncate_penalty > 0.0:
            reward -= self.truncate_penalty * np.linalg.norm(next_joints_before_truncate - next_joints) / \
                      self.action_step_size

        # if segment not valid return collision result
        if not self.openrave_manager.check_segment_validity(self.current_joints, next_joints):
            return self._get_step_result(next_joints, -1.0 + reward, True, 2)
        # if close enough to goal, return positive reward
        if self.is_below_goal_sensitivity(next_joints, self.goal_joints):
            return self._get_step_result(next_joints, 1.0 + reward, True, 3)
        # else, just a normal step...
        return self._get_step_result(next_joints, -self.keep_alive_penalty + reward, False, 1)

    def _get_step_result(self, next_joints, reward, is_terminal, enum_res):
        if is_terminal:
            self.current_joints = None
        else:
            self.current_joints = next_joints
        return list(next_joints), reward, is_terminal, enum_res

    def _split_trajectory(self, trajectory):
        max_step = self.action_step_size
        res = [tuple(trajectory[0])]
        for i in range(len(trajectory) - 1):
            current_step = np.array(trajectory[i])
            next_step = np.array(trajectory[i + 1])
            difference = next_step - current_step
            difference_norm = np.linalg.norm(difference)
            if difference_norm < max_step:
                # if smaller than allowed step just append the next step
                res.append(tuple(trajectory[i + 1]))
                continue
            scaled_step = (max_step / difference_norm) * difference
            steps = []
            for alpha in range(int(np.floor(difference_norm / max_step))):
                processed_step = current_step + (1 + alpha) * scaled_step
                steps.append(processed_step)
            # we probably have a leftover section, append it to res
            last_step_difference = np.linalg.norm(steps[-1] - next_step)
            if last_step_difference > 0.0:
                steps.append(next_step)
            # append to path
            res += [tuple(s) for s in steps]
        return res
