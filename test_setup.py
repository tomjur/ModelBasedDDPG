import tensorflow as tf
import os
import yaml

from openrave_manager import OpenraveManager
from potential_point import PotentialPoint

is_gpu = tf.test.is_gpu_available()
config_path = os.path.join(os.getcwd(), 'config/config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)
potential_points = PotentialPoint.from_config(config)
openrave_manager = OpenraveManager(0.01, potential_points)
random_joints = openrave_manager.get_random_joints()

print 'has gpu result {}'.format(is_gpu)
print 'random joints result {}'.format(random_joints)
