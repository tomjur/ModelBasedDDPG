import cPickle as pickle
import os
import yaml
import time
import random
import bz2
import numpy as np
import tensorflow as tf
from openravepy import *

from openrave_manager import OpenraveManager
from potential_point import PotentialPoint
from pre_trained_reward import PreTrainedReward
from workspace_generation_utils import WorkspaceParams

# the scenario
scenario = 'hard'
# the reward model to use
reward_model_name = 'hard'
# the joints configuration of the goal
goal_joints = [-0.3, -0.7, 1.2, 1.0]
# the direction of the action
pose_action_direction = np.array([1.0, 0.0])
# the step sizes for each joint
step_sizes = [0.1] * 4

# visualization params:
show_free = True
# show_free = False
free_samples = 10000
# show_collision = True
show_collision = False
collision_samples = 10000
# show_close_to_goal = True
show_close_to_goal = False
close_to_goal_samples = 10000
show_pose_action_direction_arrow = True
show_goal_end_effector_pose = True


# load configuration
config_path = os.path.join(os.getcwd(), 'config/config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)

# load the workspace
openrave_manager = OpenraveManager(config['openrave_rl']['segment_validity_step'], PotentialPoint.from_config(config))
params_file = os.path.abspath(os.path.expanduser(
    os.path.join('~/ModelBasedDDPG/scenario_params', scenario, 'params.pkl')))
openrave_manager.load_params(WorkspaceParams.load_from_file(params_file))
openrave_manager.robot.SetDOFValues([0.0] + goal_joints, [0, 1, 2, 3, 4])

openrave_manager.get_initialized_viewer()
red_color = np.array([1.0, 0.0, 0.0])
yellow_color = np.array([1.0, 1.0, 0.0])
green_color = np.array([0.0, 1.0, 0.0])


def create_sphere(id, radius, openrave_manager):
    body = RaveCreateKinBody(openrave_manager.env, '')
    body.SetName('sphere{}'.format(id))
    body.InitFromSpheres(np.array([[0.0]*3 + [radius]]), True)
    openrave_manager.env.Add(body, True)
    return body


def move_body(body, offset, theta):
    transformation_matrix = np.eye(4)

    translation = np.array(offset)
    rotation_matrix = np.array([
        [np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]
    ])
    transformation_matrix[:3, -1] = translation
    transformation_matrix[:3, :3] = rotation_matrix
    body.SetTransform(transformation_matrix)


def get_scaled_color(scalar, color_at_1, color_at_0):
    m = color_at_1 - color_at_0
    n = color_at_0
    result = m*scalar + n
    return result


def get_reward_color(r):
    if r < 0.0:
        return get_scaled_color(r + 1.0, yellow_color, red_color)
    else:
        return get_scaled_color(r, green_color, yellow_color)


# remove robot from view
for link in openrave_manager.robot.GetLinks():
    link.SetVisible(False)

robot_goal_pose = openrave_manager.get_target_pose([0.0] + goal_joints)
if show_goal_end_effector_pose:
    goal_sphere = create_sphere('goal', 0.01, openrave_manager)
    move_body(goal_sphere, [robot_goal_pose[0], 0.0, robot_goal_pose[1]], 0.0)
    goal_sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array([0, 0, 1.0,])) # blue
    # for i in range(21):
    #     r = 1.0 - 0.1*i
    #     r_color = get_reward_color(r)
    #     print r, r_color
    #     goal_sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(r_color)
    #     time.sleep(2)

# set action arrow
if show_pose_action_direction_arrow:
    arrow_offset = np.array([0.0]*3)
    arrow_direction = np.array([pose_action_direction[0], 0.0, pose_action_direction[1]]) / 10.0
    arrow_width = 0.005
    action_arrow = openrave_manager.env.drawarrow(arrow_offset, arrow_offset + arrow_direction, arrow_width)


# get all the steps:
def recursive_get_all_steps(i, previous_joint_partitions):
    # lower bound
    bounds = openrave_manager.get_joint_bounds()
    steps = [bounds[0][i + 1] + 0.00001]
    # get all steps
    while steps[-1] + step_sizes[i] < bounds[1][i + 1]:
        steps.append(steps[-1] + step_sizes[i])

    # generate all combination with previous joints
    new_steps = []
    for p in previous_joint_partitions:
        for s in steps:
            new_step = p + [s]
            new_steps.append(new_step)

    if i < 3:
        new_steps = recursive_get_all_steps(i+1, new_steps)

    return new_steps


def get_steps():
    cache_filepath = os.path.abspath(
        os.path.expanduser(os.path.join('~/ModelBasedDDPG/', scenario + '_validated_visualization_points.pkl')))

    if os.path.isfile(cache_filepath):
        # loading from cache
        print 'loading valid positions from cache'
        with bz2.BZ2File(cache_filepath, 'r') as compressed_file:
            return pickle.load(compressed_file)

    # compute the steps
    print 'partitioning the space'
    all_steps = recursive_get_all_steps(0, [[0.0]])
    print 'validating steps'
    all_steps = [s for s in all_steps if openrave_manager.is_valid(s)]
    # save for later
    print 'saving valid positions for later'
    with bz2.BZ2File(cache_filepath, 'w') as compressed_file:
        pickle.dump(all_steps, compressed_file)

    return all_steps


def get_poses(all_validated_steps):
    cache_filepath = os.path.abspath(
        os.path.expanduser(os.path.join('~/ModelBasedDDPG/', scenario + '_validated_poses.pkl')))

    if os.path.isfile(cache_filepath):
        # loading from cache
        print 'loading valid poses from cache'
        with bz2.BZ2File(cache_filepath, 'r') as compressed_file:
            return pickle.load(compressed_file)

    # compute the poses
    print 'calculating poses'
    all_poses = []
    for j in all_validated_steps:
        poses = openrave_manager.get_potential_points_poses(j)
        poses = poses[openrave_manager.potential_points[-1].tuple]
        all_poses.append(poses)
    # save for later
    print 'saving valid poses for later'
    with bz2.BZ2File(cache_filepath, 'w') as compressed_file:
        pickle.dump(all_poses, compressed_file)

    return all_poses


def get_jacobians(all_validated_steps):
    cache_filepath = os.path.abspath(
        os.path.expanduser(os.path.join('~/ModelBasedDDPG/', scenario + '_validated_jacobians.pkl')))

    if os.path.isfile(cache_filepath):
        # loading from cache
        print 'loading valid jacobians from cache'
        with bz2.BZ2File(cache_filepath, 'r') as compressed_file:
            return pickle.load(compressed_file)

    # compute the jacobians
    print 'calculating jacobians'
    all_jacobians = []
    for j in all_validated_steps:
        jacobians = openrave_manager.get_potential_points_jacobians(j)
        jacobians = jacobians[openrave_manager.potential_points[-1].tuple]
        all_jacobians.append(jacobians)
    # save for later
    print 'saving valid jacobians for later'
    with bz2.BZ2File(cache_filepath, 'w') as compressed_file:
        pickle.dump(all_jacobians, compressed_file)

    return all_jacobians


# compute actions with jacobian
def get_actions_to_score(all_valid_jacobians):
    print 'getting actions'
    actions = np.array([np.matmul(j, pose_action_direction) for j in all_valid_jacobians])
    actions_norms = np.linalg.norm(actions, ord=2, axis=-1)
    return actions / np.expand_dims(actions_norms, axis=1)





robot_steps = get_steps()
robot_poses = get_poses(robot_steps)
assert len(robot_steps) == len(robot_poses)
robot_jacobians = get_jacobians(robot_steps)
assert len(robot_steps) == len(robot_jacobians)
robot_actions = get_actions_to_score(robot_jacobians)

# load reward model
pre_trained_reward = PreTrainedReward(reward_model_name, config)

# query reward model
print 'computing reward predictions'
with tf.Session(
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
        )) as sess:
    sess.run(tf.global_variables_initializer())
    if pre_trained_reward is not None:
        pre_trained_reward.load_weights(sess)
    all_reward_predictions = None
    all_status_predictions = None
    batch_size = 10000
    while len(robot_steps) > 0:
        # get the current information
        current_joints = robot_steps[:batch_size]
        current_joints = [j[1:] for j in current_joints]
        current_actions = robot_actions[:batch_size]
        # advance the buffer
        robot_steps = robot_steps[batch_size:]
        robot_actions = robot_actions[batch_size:]

        current_goal_joints = [goal_joints] * len(current_joints)
        current_goal_pose = [robot_goal_pose] * len(current_joints)
        reward_predictions, status_predictions = pre_trained_reward.make_prediction(
            sess, current_joints, current_goal_joints, current_actions, current_goal_pose)
        if all_reward_predictions is None:
            all_reward_predictions = reward_predictions
            all_status_predictions = status_predictions
        else:
            all_reward_predictions = np.concatenate((all_reward_predictions, reward_predictions), axis=0)
            all_status_predictions = np.concatenate((all_status_predictions, status_predictions), axis=0)

    assert len(all_reward_predictions) == len(robot_poses)

    all_status_predictions_argmax = np.argmax(all_status_predictions, axis=1)
    all_status_predictions_argmax = list(all_status_predictions_argmax)
    all_reward_predictions = list(all_reward_predictions)

    # numeric eval:
    free_bad_value_count = 0
    collision_bad_value_count = 0
    close_to_goal_bad_value_count = 0
    for i, s in enumerate(all_status_predictions_argmax):
        r = all_reward_predictions[i]
        if s == 0 and not -0.8 <= r <= 0.8:
            free_bad_value_count += 1
        elif s == 1 and r > -0.8:
            collision_bad_value_count += 1
        elif s == 2 and r < 0.8:
            close_to_goal_bad_value_count += 1

    print 'bad values: free {} of {}, collision {} of {}, close to goal {} of {}'.format(
        free_bad_value_count, len([s for s in all_status_predictions_argmax if s == 0]),
        collision_bad_value_count, len([s for s in all_status_predictions_argmax if s == 1]),
        close_to_goal_bad_value_count, len([s for s in all_status_predictions_argmax if s == 2]),
    )

    reward_spheres = []

    reddish_indices = [i for i, r in enumerate(all_reward_predictions) if r < -0.8]
    yellowish_indices = [i for i, r in enumerate(all_reward_predictions) if -0.8 <= r < 0.8]
    greenish_indices = [i for i, r in enumerate(all_reward_predictions) if r > 0.8]

    indices_to_display = []
    if show_collision:
        if collision_samples > len(reddish_indices):
            indices_to_display.extend(reddish_indices)
        else:
            selected = list(np.random.choice(reddish_indices, collision_samples, replace=False))
            indices_to_display.extend(selected)

    if show_close_to_goal:
        if close_to_goal_samples > len(greenish_indices):
            indices_to_display.extend(greenish_indices)
        else:
            selected = list(np.random.choice(greenish_indices, close_to_goal_samples, replace=False))
            indices_to_display.extend(selected)

    if show_free:
        if free_samples > len(yellowish_indices):
            indices_to_display.extend(yellowish_indices)
        else:
            selected = list(np.random.choice(yellowish_indices, free_samples, replace=False))
            indices_to_display.extend(selected)

    for i in indices_to_display:
        reward = all_reward_predictions[i]

        pose = robot_poses[i]
        current_sphere = create_sphere('{}'.format(i), 0.005, openrave_manager)
        move_body(current_sphere, [pose[0], 0.0, pose[1]], 0.0)
        color = get_reward_color(reward)
        current_sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)  # between red and green


print 'done'


