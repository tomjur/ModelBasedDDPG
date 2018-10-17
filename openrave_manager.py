import os
import random
import numpy as np
import time
from openravepy import *
import data_filepaths
from potential_point import PotentialPoint
from workspace_generation_utils import WorkspaceParams


class OpenraveManager(object):
    def __init__(self, segment_validity_step, potential_points):
        env_path = os.path.abspath(
            os.path.expanduser('~/ModelBasedDDPG/config/widowx_env.xml'))
        self.env = Environment()
        self.env.StopSimulation()
        self.env.Load(env_path)  # load a simple scene
        self.robot = self.env.GetRobots()[0] # load the robot
        self.links_names = [l.GetName() for l in self.robot.GetLinks()]
        self.robot.SetActiveDOFs(range(1, 5)) # make the first joint invalid
        # set the color
        color = np.array([33, 213, 237])
        for link in self.robot.GetLinks():
            for geom in link.GetGeometries():
                geom.SetDiffuseColor(color)
        self.objects = []
        self.segment_validity_step = segment_validity_step
        # translate the potential to list of (unprocessed_point, link, coordinate)
        self.potential_points = potential_points
        self.joint_safety = 0.0001

    def load_params(self, workspace_params):
        with self.env:
            for i in range(workspace_params.number_of_obstacles):
                body = RaveCreateKinBody(self.env, '')
                body.SetName('box{}'.format(i))
                body.InitFromBoxes(np.array([[0, 0, 0, workspace_params.sides_x[i], 0.01, workspace_params.sides_z[i]]]),
                                   True)
                self.env.Add(body, True)

                transformation_matrix = np.eye(4)
                translation = np.array([
                    workspace_params.centers_position_x[i], 0.0, workspace_params.centers_position_z[i]])

                theta = workspace_params.y_axis_rotation[i]
                rotation_matrix = np.array([
                    [np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]
                ])
                transformation_matrix[:3, -1] = translation
                transformation_matrix[:3, :3] = rotation_matrix
                body.SetTransform(transformation_matrix)
                self.objects.append(body)

    def get_number_of_joints(self):
        return self.robot.GetDOF()

    def get_joint_bounds(self):
        return self.robot.GetDOFLimits()

    def get_random_joints(self, fixed_positions_dictionary=None):
        joint_bounds = self.get_joint_bounds()
        result = []
        for i in range(self.get_number_of_joints()):
            if i in fixed_positions_dictionary:
                result.append(fixed_positions_dictionary[i])
            else:
                result.append(random.uniform(joint_bounds[0][i], joint_bounds[1][i]))
        result = self.truncate_joints(result)
        return tuple(result)

    def truncate_joints(self, joints):

        bounds = self.get_joint_bounds()
        res = list(joints)
        for i, j in enumerate(joints):
            lower = bounds[0][i] + self.joint_safety
            res[i] = max(res[i], lower)
            upper = bounds[1][i] - self.joint_safety
            res[i] = min(res[i], upper)
        return tuple(res)

    def is_valid(self, joints):
        self.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
        res = not self.robot.CheckSelfCollision()
        if self.objects is not None:
            for item in self.objects:
                res = res and not self.env.CheckCollision(self.robot, item)
        return res

    def plan(self, start_joints, goal_joints, max_planner_iterations):
        with self.env:
            if not self.is_valid(start_joints) or not self.is_valid(goal_joints):
                return None
            self.robot.SetDOFValues(start_joints, [0, 1, 2, 3, 4])
            manipprob = interfaces.BaseManipulation(self.robot)  # create the interface for basic manipulation programs
            try:
                items_per_trajectory_step = 10
                active_joints = self.robot.GetActiveDOF()
                # call motion planner with goal joint angles
                traj = manipprob.MoveActiveJoints(goal=goal_joints[1:], execute=False, outputtrajobj=True, maxtries=1,
                                                  maxiter=max_planner_iterations)
                # found plan, if not an exception is thrown and caught below
                traj = list(traj.GetWaypoints(0, traj.GetNumWaypoints()))
                assert len(traj) % items_per_trajectory_step == 0
                # take only the joints values and add the 0 joint.
                traj = [[0.0] + traj[x:x + items_per_trajectory_step][:active_joints] for x in
                        xrange(0, len(traj), items_per_trajectory_step)]
                # assert validity
                if self.get_last_valid_in_trajectory(traj) != traj[-1]:
                    return None
                # plan found and validated!
                return traj
            except Exception, e:
                print str(e)
                return None

    def check_segment_validity(self, start_joints, end_joints):
        steps = self.partition_segment(start_joints, end_joints)
        random.shuffle(steps)
        for step in steps:
            if not self.is_valid(step):
                return False
        return True

    def partition_segment(self, start_joints, end_joints):
        # partition the segment between start joints to end joints
        current = np.array(start_joints)
        next = np.array(end_joints)
        difference = next - current
        difference_norm = np.linalg.norm(difference)
        step_size = self.segment_validity_step
        if difference_norm < step_size:
            # if smaller than allowed step just append the next step
            return [tuple(end_joints)]
        else:
            scaled_step = (step_size / difference_norm) * difference
            steps = []
            for alpha in range(int(np.floor(difference_norm / step_size))):
                processed_step = current + (1 + alpha) * scaled_step
                steps.append(processed_step)
            # we probably have a leftover section, append it to res
            last_step_difference = np.linalg.norm(steps[-1] - next)
            if last_step_difference > 0.0:
                steps.append(next)
            # append to list of configuration points to test validity
            return [tuple(s) for s in steps]

    def get_last_valid_in_trajectory(self, trajectory):
        for i in range(len(trajectory)-1):
            if not self.check_segment_validity(trajectory[i], trajectory[i+1]):
                return trajectory[i]
        return trajectory[-1]

    def get_initialized_viewer(self):
        if self.env.GetViewer() is None:
            self.env.SetViewer('qtcoin')
        # set camera
        camera_transform = np.eye(4)
        theta = -np.pi / 2
        rotation_matrix = np.array([
            [1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]
        ])
        camera_transform[:3, :3] = rotation_matrix
        camera_transform[:3, 3] = np.array([0.0, -1.0, 0.25])
        time.sleep(1)
        viewer = self.env.GetViewer()
        viewer.SetCamera(camera_transform)
        return viewer

    @staticmethod
    def get_manager_for_workspace(workspace_id, config):
        directory = os.path.abspath(os.path.expanduser(config['data']['directory']))
        workspace_dir = os.path.join(directory, workspace_id)
        potential_points = PotentialPoint.from_config(config)
        openrave_manager = OpenraveManager(config['data']['joint_segment_validity_step'], potential_points)
        workspace_params = WorkspaceParams.load_from_file(data_filepaths.get_workspace_params_path(workspace_dir))
        openrave_manager.load_params(workspace_params)
        return openrave_manager, workspace_dir

    def get_links_poses(self, joints):
        self.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
        poses = self.robot.GetLinkTransformations()
        result = {
            link_name: tuple(poses[i][[0, 2], -1])
            for i, link_name in enumerate(self.links_names) if link_name in self.links_names
        }
        return result

    def get_links_poses_array(self, joints):
        poses = self.get_links_poses(joints)
        return [poses[link_name] for link_name in self.links_names]

    def get_potential_points_poses(self, joints, post_process=True):
        self.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
        link_transform = self.robot.GetLinkTransformations()
        result = {p.tuple: np.matmul(link_transform[p.link], p.coordinate) for p in self.potential_points}
        if post_process:
            result = {k: (result[k][0], result[k][2]) for k in result}
        return result

    def get_target_pose(self, joints):
        # target is the last potential
        return self.get_potential_points_poses(joints)[self.potential_points[-1].tuple]

    @staticmethod
    def _post_process_jacobian(j, is_numeric=False):
        return j[[0, 2], 1 if is_numeric else 0:].transpose()

    def get_links_jacobians(self, joints, modeling_links=None):
        if modeling_links is None:
            modeling_links = self.links_names
        self.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
        poses = self.robot.GetLinkTransformations()
        return {
            link_name: self._post_process_jacobian(self.robot.CalculateActiveJacobian(i, poses[i][:3, 3]))
            for i, link_name in enumerate(self.links_names) if link_name in modeling_links
        }

    def get_potential_points_jacobians(self, joints):
        potential_points_poses = self.get_potential_points_poses(joints, post_process=False)
        self.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
        return {
            p.tuple: self._post_process_jacobian(
                self.robot.CalculateActiveJacobian(p.link, potential_points_poses[p.tuple])
            )
            for p in self.potential_points
        }

    # def get_links_numeric_jacobians(self, joints, modeling_links=None, epsilon=0.0001):
    #     if modeling_links is None:
    #         modeling_links = self.links_names
    #     res = {link_name: np.zeros((3, len(joints))) for link_name in modeling_links}
    #     bounds = self.get_joint_bounds()
    #     for i in range(len(joints)):
    #         local_j = [j for j in joints]
    #         new_value = local_j[i] + epsilon
    #         use_upper = new_value < bounds[1][i]
    #         if use_upper:
    #             local_j[i] += epsilon
    #         self.robot.SetDOFValues(local_j, [0, 1, 2, 3, 4])
    #         transforms = self.robot.GetLinkTransformations()
    #         p1 = {link_name: transforms[i][:3, 3]
    #               for i, link_name in enumerate(self.links_names) if link_name in modeling_links}
    #         local_j = [j for j in joints]
    #         new_value = local_j[i] - epsilon
    #         use_lower = new_value > bounds[0][i]
    #         if use_lower:
    #             local_j[i] -= epsilon
    #         self.robot.SetDOFValues(local_j, [0, 1, 2, 3, 4])
    #         transforms = self.robot.GetLinkTransformations()
    #         p2 = {link_name: transforms[i][:3, 3]
    #               for i, link_name in enumerate(self.links_names) if link_name in modeling_links}
    #         for link_name in modeling_links:
    #             local_res = (p1[link_name]-p2[link_name]) / (use_lower*epsilon + use_upper*epsilon)
    #             res[link_name][:, i] = local_res
    #     return {link_name: self._post_process_jacobian(res[link_name], is_numeric=True) for link_name in res}

    # def get_target_jacobian(self, joints):
    #     return self.get_links_jacobians(joints, self.links_names[-1])


# if __name__ == "__main__":
#     m = OpenraveManager(0.01)
#
#
#     # joints0 = [0.0]*5
#     # poses0 = m.get_links_poses(joints0)
#     # print m.is_valid(joints0)
#     # joints1 = [0.0]*4 + [1.5]
#     # poses1 = m.get_links_poses(joints1)
#     # print m.is_valid(joints1)
#     # m.get_links_poses_array(joints0)
#     #
#     # joints = [0.1 + 0.2*i for i in range(5)]
#     # poses = m.get_links_poses(joints)
#     # numeric_jacobians = m.get_links_numeric_jacobians(joints)
#     # for i in range(7):
#     #     m.robot.SetDOFValues(joints, [0, 1, 2, 3, 4])
#     #     print i
#     #     print m.links_names[i]
#     #     print 'pose:'
#     #     print poses[m.links_names[i]]
#     #     print 'local jacobian:'
#     #     print m.robot.CalculateJacobian(i, joints)
#     #     print 'active  jacobian at 0'
#     #     print m.robot.CalculateActiveJacobian(i, [0.0, 0.0, 0.0])
#     #     print 'active  jacobian at pose'
#     #     p = m.robot.GetLinkTransformations()[i][:3,3]
#     #     print m.robot.CalculateActiveJacobian(i, p)
#     #     print 'numerical jacobian'
#     #     print numeric_jacobians[m.links_names[i]]
#     #     print ''
#
#
#
#
#
#     # transformed = []
#     # for x_corner in [-x_length / 2.0, x_length / 2.0]:
#     #     for y_corner in [-y_length / 2.0, y_length / 2.0]:
#     #         for z_corner in [-z_length / 2.0, z_length / 2.0]:
#     #             corner = np.array([x_corner,y_corner,z_corner,1.0])
#     #             print 'corner {}'.format(corner)
#     #             print 'transform:'
#     #             transformed_corner = np.matmul(body.GetTransform(), corner)
#     #             transformed.append(transformed_corner)
#     #             print transformed_corner
#
#     from mpl_toolkits.mplot3d import Axes3D
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter([t[0] for t in transformed], [t[1] for t in transformed], [t[2] for t in transformed])
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#
#     plt.show()
#
#     print 'here'


if __name__ == "__main__":
    potential_points = [PotentialPoint(t) for t in [(4, 0.0, 0.0), (5, 0.0, 0.0)]]
    m = OpenraveManager(0.01, potential_points)
    joints0 = [0.0] * 5
    res1 = m.get_potential_points_poses(joints0)
    res2 = m.get_links_poses(joints0)
    print res1[potential_points[0].tuple] == res2[m.links_names[potential_points[0].link]]
    print res1[potential_points[1].tuple] == res2[m.links_names[potential_points[1].link]]

    res3 = m.get_potential_points_jacobians(joints0)
    res4 = m.get_links_jacobians(joints0)
    print res3[potential_points[0].tuple] == res4[m.links_names[potential_points[0].link]]
    print res3[potential_points[1].tuple] == res4[m.links_names[potential_points[1].link]]

    joints0 = [0.0] * 5
    joints0[2] = np.pi/4
    res1 = m.get_potential_points_poses(joints0)
    res2 = m.get_links_poses(joints0)
    print res1[potential_points[0].tuple] == res2[m.links_names[potential_points[0].link]]
    print res1[potential_points[1].tuple] == res2[m.links_names[potential_points[1].link]]

    res3 = m.get_potential_points_jacobians(joints0)
    res4 = m.get_links_jacobians(joints0)
    print res3[potential_points[0].tuple] == res4[m.links_names[potential_points[0].link]]
    print res3[potential_points[1].tuple] == res4[m.links_names[potential_points[1].link]]