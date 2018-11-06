import glob
import os
import pickle
import yaml
import time
import numpy as np

from data_filepaths import get_trajectory_path
from openrave_manager import OpenraveManager
from openravepy import *

# global variables:
from potential_point import PotentialPoint
from workspace_generation_utils import WorkspaceParams

trajectories_dir = os.path.abspath(os.path.expanduser('/home/tom/ModelBasedDDPG/trajectories'))
model_name = '2018_11_05_11_41_15'
global_step = '36040'
path_id = '0'
message = 'max_len'
# message = 'collision'
# message = 'success'
# workspace_params_path = None
workspace_params_path = 'params.pkl'
speed = 10.0

display_start_goal_end_spheres = True
trajectory_spheres_radius = 0.01


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


def main():
    config_path = os.path.join(os.getcwd(), 'config/config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        print('------------ Config ------------')
        print(yaml.dump(config))
    model_dir = os.path.join(trajectories_dir, model_name)
    potential_points_location = os.path.join(model_dir, 'potential_points.p')
    potential_points = pickle.load(open(potential_points_location))

    openrave_manager = OpenraveManager(0.001, potential_points)
    if workspace_params_path is not None:
        openrave_manager.load_params(WorkspaceParams.load_from_file(workspace_params_path))
    openrave_manager.get_initialized_viewer()
    # visualize the trajectory
    # search_key = os.path.join(model_dir, '{}_step_{}_{}.p'.format(global_step, message, path_id))
    search_key = os.path.join(model_dir, global_step, '{}_{}.p'.format(message, path_id))
    trajectory_files = glob.glob(search_key)
    trajectory_file = trajectory_files[0]
    pose_goal, trajectory = pickle.load(open(trajectory_file))
    trajectory = [[0.0] + list(j) for j in trajectory]

    if display_start_goal_end_spheres:
        start = trajectory[0]
        end = trajectory[-1]
        pose_start = openrave_manager.get_target_pose(start)
        pose_start = (pose_start[0], 0.0, pose_start[1])
        pose_goal = (pose_goal[0], 0.0, pose_goal[1])
        pose_end = openrave_manager.get_target_pose(end)
        pose_end = (pose_end[0], 0.0, pose_end[1])
        start_sphere = create_sphere('start', trajectory_spheres_radius, openrave_manager)
        move_body(start_sphere, pose_start, 0.0)
        goal_sphere = create_sphere('goal', trajectory_spheres_radius, openrave_manager)
        move_body(goal_sphere, pose_goal, 0.0)
        end_sphere = create_sphere('end', trajectory_spheres_radius, openrave_manager)

        start_sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array([0, 0, 204]))
        goal_sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array([240, 100, 10]))
        end_sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array([100, 204, 204]))
        move_body(end_sphere, pose_end, 0.0)

    print 'len(trajectory) ', len(trajectory)
    for i in range(len(trajectory)):
        print 'i ', i
        openrave_manager.robot.SetDOFValues(trajectory[i])
        time.sleep(1/speed)


# def get_coordinate(link_transforms, attached_link_index, initial_offset):
#     link_transform = link_transforms[attached_link_index]
#     coordinate = np.matmul(link_transform, np.append(initial_offset, [1.0]))
#     return coordinate[:3]

if __name__ == '__main__':
    main()
