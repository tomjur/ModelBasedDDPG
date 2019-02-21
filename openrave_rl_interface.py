import numpy as np
from openrave_manager import OpenraveManager
from potential_point import PotentialPoint


class OpenraveRLInterface:
    # class StepResult(enum.Enum):
    #     free_space = 1
    #     collision = 2
    #     close_to_goal = 3

    def __init__(self, config):
        self.action_step_size = config['openrave_rl']['action_step_size']
        self.goal_sensitivity = config['openrave_rl']['goal_sensitivity']
        self.keep_alive_penalty = config['openrave_rl']['keep_alive_penalty']
        self.truncate_penalty = config['openrave_rl']['truncate_penalty']

        self.openrave_manager = OpenraveManager(
            config['openrave_rl']['segment_validity_step'], PotentialPoint.from_config(config))

        self.current_joints = None
        self.goal_joints = None
        self.start_joints = None
        self.traj = None

    def is_below_goal_sensitivity(self, start_joints, goal_joints):
        start_pose = self.openrave_manager.get_target_pose(start_joints)
        goal_pose = self.openrave_manager.get_target_pose(goal_joints)
        pose_distance = np.linalg.norm(np.array(start_pose) - np.array(goal_pose))
        return pose_distance < self.goal_sensitivity

    def start_specific(self, traj, verify_traj=True):
        self.traj = traj
        start_joints = traj[0]
        goal_joints = traj[-1]
        # assert path is legal
        if verify_traj:
            step_size = self.action_step_size + 0.00001
            for i in range(len(traj)-1):
                step_i_size = np.linalg.norm(np.array(traj[i]) - np.array(traj[i+1]))
                assert step_i_size < step_size, 'step_i_size {}'.format(step_i_size)
        steps_required_for_motion_plan = len(traj)
        self.current_joints = np.array(start_joints)
        self.start_joints = np.array(start_joints)
        self.goal_joints = np.array(goal_joints)
        return start_joints, goal_joints, steps_required_for_motion_plan

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
