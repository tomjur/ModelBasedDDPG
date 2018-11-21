import datetime
import yaml
import os

from openrave_manager import OpenraveManager
from potential_point import PotentialPoint
from workspace_generation_utils import *

number_of_workspaces = 10
test_trajectories = 1000
attempts_to_find_single_trajectory = 1000
trajectories_required_to_pass = 500
planner_iterations = 1500

# init the generator
generator = WorkspaceGenerator()
# init the openrave manager
config_path = os.path.join(os.getcwd(), 'config/config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)
    print('------------ Config ------------')
    print(yaml.dump(config))


def is_below_goal_sensitivity(start_pose, goal_pose):
    pose_distance = np.linalg.norm(np.array(start_pose) - np.array(goal_pose))
    return pose_distance < config['openrave_rl']['goal_sensitivity']


def is_challenging(start_pose, goal_pose, workspace_params):
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


def _try_plan(workspace_params, openrave_manager):
    for i in range(attempts_to_find_single_trajectory):
        # select at random
        start_joints = openrave_manager.get_random_joints({0: 0.0})
        goal_joints = openrave_manager.get_random_joints({0: 0.0})
        start_pose = openrave_manager.get_target_pose(start_joints)
        goal_pose = openrave_manager.get_target_pose(goal_joints)
        # if the start and goal are too close, re-sample
        if is_below_goal_sensitivity(start_joints, goal_joints):
            continue
        # valid region:
        if not start_pose[1] < 0.0 or goal_pose[1] < 0.0:
            continue
        # trajectories that must cross an obstacle
        if is_challenging(start_pose, goal_pose, workspace_params):
            continue
        traj = openrave_manager.plan(start_joints, goal_joints, planner_iterations)
        return traj is not None
    return None


workspace_count = 0
a = datetime.datetime.now()
while workspace_count < number_of_workspaces:
    workspace_params = generator.generate_workspace()
    openrave_manager = OpenraveManager(config['openrave_rl']['segment_validity_step'],
                                       PotentialPoint.from_config(config))
    openrave_manager.load_params(workspace_params)
    successful_trajectories_count = 0
    i = 0
    for i in range(test_trajectories):
        # see if there is hope
        trajectories_left = test_trajectories - i
        if trajectories_left + successful_trajectories_count < trajectories_required_to_pass:
            print 'no hope to get the required ratio'
            break
        # try a trajectory
        successful_trajectories_count += _try_plan(workspace_params, openrave_manager)
        # if successful update the status
        if successful_trajectories_count >= trajectories_required_to_pass:
            workspace_count += 1
            print 'workspace found'
            break
    b = datetime.datetime.now()
    print 'workspace count {}'.format(workspace_count)
    print 'trajectories tried {}'.format(i)
    print 'success count {}'.format(successful_trajectories_count)
    print 'time since start {}'.format(b-a)
    print ''


