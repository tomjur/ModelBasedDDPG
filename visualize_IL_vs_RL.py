import cPickle as pickle
import os
import yaml
import random
import bz2
import numpy as np

from openrave_manager import OpenraveManager
from potential_point import PotentialPoint

# the scenario
scenario = 'hard'
model_name = '2019_01_25_10_09_04'
number_of_imitation_files = 3
sphere_limitation = 1000

imitation_data_path = os.path.abspath(os.path.expanduser(os.path.join('~/ModelBasedDDPG/imitation_data', scenario)))
rl_trajectories_data_path = os.path.abspath(os.path.expanduser(
    os.path.join('~/ModelBasedDDPG/', scenario, 'trajectories', model_name)))

# load configuration
config_path = os.path.join(os.getcwd(), 'config/config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)

# load the workspace
openrave_manager = OpenraveManager(config['openrave_rl']['segment_validity_step'], PotentialPoint.from_config(config))


def process_poses(target_poses, x_coordinate_range=(0.0, 0.13), z_coordinate_range=(0.3, 0.45)):
    return [p for p in target_poses if x_coordinate_range[0] <= p[0] <= x_coordinate_range[1] and z_coordinate_range[0] <= p[1] <= z_coordinate_range[1]]


def process_rl_files(data_dir, trajectory_limitation):
    steps_offset = 40
    steps_increase = 2000
    trajectories_seen = 0
    result = []
    while trajectories_seen < trajectory_limitation:
        global_step_dir = os.path.join(data_dir, '{}'.format(steps_offset))
        steps_offset += steps_increase
        for dirpath, dirnames, filenames in os.walk(global_step_dir):
            for filename in filenames:
                if not filename.endswith('.p'):
                    continue
                source_file = os.path.join(global_step_dir, filename)
                print 'working on {}'.format(source_file)
                with open(source_file, 'r') as f:
                    trajectory = pickle.load(f)
                joints = trajectory[1]
                poses = [openrave_manager.get_target_pose(tuple([0.0] + list(j))) for j in joints]
                poses_after_filter = process_poses(poses)
                result.extend(poses_after_filter)
                trajectories_seen += 1
                if trajectories_seen == trajectory_limitation:
                    break
    return result


rl_transitions = process_rl_files(rl_trajectories_data_path, number_of_imitation_files * 1000)


def process_dir(data_dir, limit_files, target_point_tuple=(5, -0.02, 0.035)):
    for dirpath, dirnames, filenames in os.walk(data_dir):
        result = []
        for filename in filenames[:limit_files]:
            if not filename.endswith('.path_pkl'):
                continue
            source_file = os.path.join(data_dir, filename)
            print 'working on {}'.format(source_file)
            with bz2.BZ2File(source_file, 'r') as compressed_file:
                all_trajectories = pickle.load(compressed_file)
                for trajectory in all_trajectories:
                    poses = trajectory[1]
                    poses_after_filter = process_poses([p[target_point_tuple] for p in poses])
                    result.extend(poses_after_filter)
        return result


il_transitions = process_dir(os.path.join(imitation_data_path, 'train'), number_of_imitation_files)

# load the openrave view
params_file = os.path.abspath(os.path.expanduser(
    os.path.join('~/ModelBasedDDPG/scenario_params', scenario, 'params.pkl')))
openrave_manager.set_params(params_file)
openrave_manager.get_initialized_viewer()

# remove robot from view
for link in openrave_manager.robot.GetLinks():
    link.SetVisible(False)


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

def paint_transition(transitions, limitation, color):
    random.shuffle(transitions)
    for i, t in enumerate(transitions[:limitation]):
        goal_sphere = create_sphere('goal_{}'.format(i), 0.01, openrave_manager)
        move_body(goal_sphere, [t[0], 0.0, t[1]], 0.0)
        goal_sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)

paint_transition(il_transitions, sphere_limitation, red_color)
# paint_transition(rl_transitions, sphere_limitation, green_color)

print 'here'