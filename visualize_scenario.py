import os
import yaml

from openrave_manager import OpenraveManager
from potential_point import PotentialPoint

# the scenario
scenario = 'hard'

# load configuration
config_path = os.path.join(os.getcwd(), 'config/config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)

# load the workspace
openrave_manager = OpenraveManager(config['openrave_rl']['segment_validity_step'], PotentialPoint.from_config(config))

# load the openrave view
params_file = os.path.abspath(os.path.expanduser(
    os.path.join('~/ModelBasedDDPG/scenario_params', scenario, 'params.pkl')))
openrave_manager.set_params(params_file)
openrave_manager.get_initialized_viewer()

print 'here'