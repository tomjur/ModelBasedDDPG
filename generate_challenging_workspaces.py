import time
import datetime
import yaml
import os
import multiprocessing
import Queue

from openrave_manager import OpenraveManager
from potential_point import PotentialPoint
from workspace_generation_utils import *

# number_of_workspaces = 10
# number_of_workers = 2
# test_trajectories = 100
# attempts_to_find_single_trajectory = 100
# trajectories_required_to_pass = 5
# planner_iterations = 150

number_of_workspaces = 1000
number_of_workers = 72
test_trajectories = 1000
attempts_to_find_single_trajectory = 1000
trajectories_required_to_pass = 500
planner_iterations = 1500

output_dir = 'vision_workspaces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class WorkerQueue(multiprocessing.Process):
    def __init__(self, test_trajectories, attempts_to_find_single_trajectory, trajectories_required_to_pass,
                 planner_iterations, workspace_generation_queue, worker_specific_queue, results_queue):
        multiprocessing.Process.__init__(self)
        # parameters
        self.test_trajectories = test_trajectories
        self.attempts_to_find_single_trajectory = attempts_to_find_single_trajectory
        self.trajectories_required_to_pass = trajectories_required_to_pass
        self.planner_iterations = planner_iterations
        # queues
        self.workspace_generation_queue = workspace_generation_queue
        self.worker_specific_queue = worker_specific_queue
        self.results_queue = results_queue
        # members
        self.generator = WorkspaceGenerator()
        config_path = os.path.join(os.getcwd(), 'config/config.yml')
        with open(config_path, 'r') as yml_file:
            self.config = yaml.load(yml_file)
        self.openrave_manager = None

    def _is_below_goal_sensitivity(self, start_pose, goal_pose):
        pose_distance = np.linalg.norm(np.array(start_pose) - np.array(goal_pose))
        return pose_distance < self.config['openrave_rl']['goal_sensitivity']

    @staticmethod
    def _is_challenging(start_pose, goal_pose, workspace_params):
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

    def _try_plan(self, workspace_params, openrave_manager):
        for i in range(self.attempts_to_find_single_trajectory):
            # select at random
            start_joints = openrave_manager.get_random_joints({0: 0.0})
            goal_joints = openrave_manager.get_random_joints({0: 0.0})
            start_pose = openrave_manager.get_target_pose(start_joints)
            goal_pose = openrave_manager.get_target_pose(goal_joints)
            # if the start and goal are too close, re-sample
            if self._is_below_goal_sensitivity(start_joints, goal_joints):
                continue
            # valid region:
            if not start_pose[1] < 0.0 or goal_pose[1] < 0.0:
                continue
            # trajectories that must cross an obstacle
            if self._is_challenging(start_pose, goal_pose, workspace_params):
                continue
            traj = openrave_manager.plan(start_joints, goal_joints, self.planner_iterations)
            return traj is not None
        return None

    def _generate_single_workspace(self, workspace_id):
        while True:
            a = datetime.datetime.now()
            workspace_params = self.generator.generate_workspace()
            self.openrave_manager = OpenraveManager(self.config['openrave_rl']['segment_validity_step'],
                                                    PotentialPoint.from_config(self.config))
            self.openrave_manager.load_params(workspace_params)
            successful_trajectories_count = 0
            i = 0
            for i in range(self.test_trajectories):
                # see if there is hope
                trajectories_left = self.test_trajectories - i
                if trajectories_left + successful_trajectories_count < self.trajectories_required_to_pass:
                    print 'no hope to get the required ratio'
                    break
                # try a trajectory
                successful_trajectories_count += self._try_plan(workspace_params, self.openrave_manager)
                # if successful update the status
                if successful_trajectories_count >= self.trajectories_required_to_pass:
                    print 'workspace found'
                    save_path = os.path.join(output_dir, '{}_workspace.pkl'.format(workspace_id))
                    workspace_params.save(save_path)
                    return
            b = datetime.datetime.now()
            print 'trajectories tried {}'.format(i)
            print 'success count {}'.format(successful_trajectories_count)
            print 'time since start {}'.format(b - a)
            print ''

    def run(self):
        while True:
            try:
                # wait 1 second for a workspace generation request
                workspace_id = self.workspace_generation_queue.get(block=True, timeout=1)
                self._generate_single_workspace(workspace_id)
                self.results_queue.put(workspace_id)
                self.workspace_generation_queue.task_done()
            except Queue.Empty:
                pass
            try:
                # need to terminate
                worker_specific_task = self.worker_specific_queue.get(block=True, timeout=0.001)
                self.worker_specific_queue.task_done()
                break
            except Queue.Empty:
                pass

# set the queues
workers_specific_queues = [multiprocessing.JoinableQueue() for _ in range(number_of_workers)]
requests_queue = multiprocessing.JoinableQueue()
results_queue = multiprocessing.Queue()
# init the workers
workers = [WorkerQueue(test_trajectories, attempts_to_find_single_trajectory, trajectories_required_to_pass,
                       planner_iterations, requests_queue, workers_specific_queues[i], results_queue)
           for i in range(number_of_workers)]
# run the workers
for w in workers:
    w.start()

global_start_time = datetime.datetime.now()
# set generate workspaces request
for workspace_id in range(number_of_workspaces):
    requests_queue.put(workspace_id)
# wait for them
for i in range(number_of_workspaces):
    workspace_id = results_queue.get()
    current_time = datetime.datetime.now()
    print 'workspace {} found'.format(workspace_id)
    print 'time from start {}'.format(current_time - global_start_time)
    print ''
# close all workers
for queue in workers_specific_queues:
    queue.put(None)
time.sleep(10)
