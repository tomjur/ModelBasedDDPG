import glob
import os


def get_all_workspaces_directories(directory):
    workspaces = [s[0] for s in os.walk(directory)]
    return [w for w in workspaces if len(w.replace(directory, '').split('/')) == 2]


def get_workspace_params_path(workspace_dir):
    return os.path.join(workspace_dir, 'params.pkl')


def get_image_path(workspace_dir):
    return os.path.join(workspace_dir, 'img.png')


def get_trajectory_path(directory, path_id):
    return os.path.join(directory, '{}.p'.format(path_id))


def get_paths_in_dir(workspace_dir):
    search_key = os.path.join(workspace_dir, '[0-9]*.p')
    return glob.glob(search_key)
