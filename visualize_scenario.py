import os
import yaml

from openrave_manager import OpenraveManager
from potential_point import PotentialPoint

# the scenario
# scenario = 'simple'
# scenario = 'hard'
scenario = 'vision'
workspace_id = 35

# load configuration
config_path = os.path.join(os.getcwd(), 'config/config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)

# load the workspace
openrave_manager = OpenraveManager(config['openrave_rl']['segment_validity_step'], PotentialPoint.from_config(config))

# load the openrave view
params_file_path = os.path.abspath(os.path.expanduser('~/ModelBasedDDPG/scenario_params'))
if scenario == 'vision':
    params_file_path = os.path.join(params_file_path, 'vision', '{}_workspace.pkl'.format(workspace_id))
else:
    params_file_path = os.path.join(params_file_path, scenario, 'params.pkl')
openrave_manager.set_params(params_file_path)
openrave_manager.get_initialized_viewer()

print 'here'