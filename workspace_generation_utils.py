from PIL import Image
import pickle
from random import randint, uniform
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate
from matplotlib import pyplot
from descartes.patch import PolygonPatch


class WorkspaceParams:
    def __init__(self):
        self.number_of_obstacles = 0
        self.centers_position_x = []
        self.centers_position_z = []
        self.sides_x = []
        self.sides_z = []
        self.y_axis_rotation = []
        self.rays = []

    def save(self, file_path):
        pickle.dump(self, open(file_path, 'w'))

    @staticmethod
    def load_from_file(file_path):
        instance = pickle.load(open(file_path))
        shrink = 0.7
        # shrink = 1.0
        instance.sides_x = [s * shrink for s in instance.sides_x]
        instance.sides_z = [s * shrink for s in instance.sides_z]
        return instance
        # return pickle.load(open(file_path))

    @staticmethod
    def _get_box_polygon(center_x, center_z, side_x, side_z, y_rotation):
        points = [
            (center_x - side_x / 2.0, center_z + side_z / 2.0 - 0/125),
            (center_x + side_x / 2.0, center_z + side_z / 2.0 - 0/125),
            (center_x + side_x / 2.0, center_z - side_z / 2.0 - 0/125),
            (center_x - side_x / 2.0, center_z - side_z / 2.0 - 0/125),
        ]
        box = Polygon(points)
        box = rotate(geom=box, angle=-y_rotation, origin='center', use_radians=True)
        return box

    def print_image(self, trajectory=None, reference_trajectory=None, starting_pose=None, trajectory_end_pose=None,
                    reference_end_pose=None):
        fig = pyplot.figure(1, dpi=90)
        ax = fig.add_subplot(111)

        # plot obstacles
        for i in range(self.number_of_obstacles):
            rotated_box = WorkspaceParams._get_box_polygon(self.centers_position_x[i], self.centers_position_z[i],
                                                           self.sides_x[i], self.sides_z[i], self.y_axis_rotation[i])
            patch = PolygonPatch(rotated_box, facecolor='#6699cc', edgecolor='#6699cc', alpha=1.0, zorder=2)
            ax.add_patch(patch)

        def plot_path(path, path_color):
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, '.-', color=path_color)

        if trajectory is not None:
            plot_path(trajectory, 'red')

        if reference_trajectory is not None:
            plot_path(reference_trajectory, 'green')

        if starting_pose is not None:
            plot_path(starting_pose, 'cyan')

        if trajectory_end_pose is not None:
            plot_path(trajectory_end_pose, 'magenta')

        if reference_end_pose is not None:
            plot_path(reference_end_pose, 'black')

        # print according to bounding box
        # x_range = [-int(0.5), int(self.outerbox_length)]
        # y_range = [-int(self.outerbox_length), int(self.outerbox_length)]
        x_range = [-0.5, 0.5]
        y_range = [0, 0.5]
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)
        return fig

    def print_image_many_trajectories(self, ax, other_trajectories, reference_trajectory=None):
        # fig = pyplot.figure(1, dpi=90)
        # ax = fig.add_subplot(111)

        # plot obstacles
        for i in range(self.number_of_obstacles):
            rotated_box = WorkspaceParams._get_box_polygon(self.centers_position_x[i], self.centers_position_z[i],
                                                           self.sides_x[i], self.sides_z[i], self.y_axis_rotation[i])
            patch = PolygonPatch(rotated_box, facecolor='#6699cc', edgecolor='#6699cc', alpha=1.0, zorder=2)
            ax.add_patch(patch)

        def plot_path(path, path_color):
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, '.-', color=path_color)

        for trajectory in other_trajectories:
            plot_path(trajectory, 'red')

        if reference_trajectory is not None:
            plot_path(reference_trajectory, 'green')

        # print according to bounding box
        # x_range = [-int(0.5), int(self.outerbox_length)]
        # y_range = [-int(self.outerbox_length), int(self.outerbox_length)]
        x_range = [-0.5, 0.5]
        y_range = [0, 0.5]
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)
        # return fig
        return ax

    @staticmethod
    def _figure_to_nparray(fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        return buf

    @staticmethod
    def _figure_to_image(fig):
        buf = WorkspaceParams._figure_to_nparray(fig)
        w, h, d = buf.shape
        return Image.frombytes("RGBA", (w, h), buf.tobytes())

    @staticmethod
    def _remove_transparency(im, bg_colour=(255, 255, 255)):
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

            # Need to convert to RGBA if LA format due to a bug in PIL
            alpha = im.convert('RGBA').split()[-1]

            # Create a new background image of our matt color.
            # Must be RGBA because paste requires both images have the same format

            bg = Image.new("RGBA", im.size, bg_colour + (255,))
            bg.paste(im, mask=alpha)
            return bg

        else:
            return im

    def get_image_as_numpy(self):
        f = self.print_image()
        im = WorkspaceParams._figure_to_image(f)
        im = WorkspaceParams._remove_transparency(im).convert('L')
        width = im.width / 16
        height = im.height / 16
        im.thumbnail((width, height), Image.ANTIALIAS)
        res = np.asarray(im)
        # res = np.array(im.getdata()).reshape((im.size[0], im.size[1], 1))
        # res = res.reshape((im.size[0], im.size[1], 1))
        pyplot.clf()
        return res


class WorkspaceGenerator:
    # problem settings
    center_offset = np.array([0.0, 0.125])
    rightmost_position = np.array([0.355, 0.078])
    rightmost_position_centered = rightmost_position - center_offset
    stretched_length = np.linalg.norm(rightmost_position_centered)
    rightmost_position_centered_direction = rightmost_position_centered / stretched_length
    min_angle = 0.0
    max_angle = np.pi

    def __init__(self, print_info=True, min_obstacles=1, max_obstacles=3, min_center=0.2, max_center=0.3,
                 min_side=0.01, max_side=0.1, obstacle_count_probabilities=None):
        # should print parameters?
        self.print_info = print_info
        # parameters that control the generation random process
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.obstacle_count_probabilities = obstacle_count_probabilities
        self.min_center = min_center
        self.max_center = max_center
        self.min_side = min_side
        self.max_side = max_side

    def _print_variable(self, name, variable):
        if self.print_info:
            print name, variable

    def _randomize_obstacle_parameters(self):
        ray_angle = uniform(WorkspaceGenerator.min_angle, WorkspaceGenerator.max_angle)
        center_distance = uniform(self.min_center, self.max_center)
        x_side = uniform(self.min_side, self.max_side)
        z_side = uniform(self.min_side, self.max_side)
        y_axis_rotation = uniform(0.0, np.pi/2.0)
        return {
            'ray_angle': ray_angle,
            'center_distance': center_distance,
            'x_side': x_side,
            'z_side': z_side,
            'y_axis_rotation': y_axis_rotation
        }

    @staticmethod
    def center_to_ray_angle(center):
        center = np.array(center) - WorkspaceGenerator.center_offset
        center = center / np.linalg.norm(center)
        ref = WorkspaceGenerator.rightmost_position_centered_direction
        ref = ref / np.linalg.norm(center)
        return np.arccos(np.clip(np.dot(center, ref), -1.0, 1.0))

    def _generate_obstacle(self, description):
        for k in description:
            self._print_variable(k, description[k])
        # first scale the rightmost position direction to be center_distance from the origin
        scaled_rightmost = WorkspaceGenerator.rightmost_position_centered_direction * description['center_distance']
        scaled_rightmost.resize((2, 1))
        # next, rotate it about the y axis
        ray_angle = description['ray_angle']
        rotation_matrix = np.matrix([[np.cos(ray_angle), -np.sin(ray_angle)], [np.sin(ray_angle), np.cos(ray_angle)]])
        rotated_center = rotation_matrix * scaled_rightmost
        rotated_center = rotated_center.squeeze()
        final_center = rotated_center + WorkspaceGenerator.center_offset
        final_center.resize((2,))
        self._print_variable('final_center', final_center)
        return final_center, [description['x_side'], description['z_side']], description['y_axis_rotation'], ray_angle, description['center_distance']

    def generate_workspace(self):
        result = WorkspaceParams()
        if self.obstacle_count_probabilities is None:
            number_of_obstacles = randint(self.min_obstacles, self.max_obstacles)
        else:
            count, probabilities = [], []
            for c in self.obstacle_count_probabilities:
                count.append(c)
                probabilities.append(self.obstacle_count_probabilities[c])
            number_of_obstacles = np.random.choice(count, p=probabilities)
        result.number_of_obstacles = number_of_obstacles
        self._print_variable('number_of_obstacles', number_of_obstacles)
        for i in range(number_of_obstacles):
            self._print_variable('obstacle index', i)
            # description = self._fixed_obstacle_parameters(i, number_of_obstacles)
            description = self._randomize_obstacle_parameters()
            center, sides, y_axis_rotation, ray_angle, center_distance = self._generate_obstacle(description)
            result.centers_position_x.append(center[0])
            result.centers_position_z.append(center[1])
            result.sides_x.append(sides[0])
            result.sides_z.append(sides[1])
            result.y_axis_rotation.append(y_axis_rotation)
            result.rays.append(ray_angle)
        return result

    def rays_to_slices(self, rays):
        slices_bounds = [WorkspaceGenerator.min_angle, WorkspaceGenerator.max_angle] + rays
        slices_bounds.sort()
        return slices_bounds


class TrajectoryGenerator:
    def __init__(self, environment, print_info=True, joint0_position=0.0):
        # properties that relate to the problem
        self.environment = environment
        self.joint0_position = joint0_position
        # configuration values
        self.print_info = print_info

    # def plan_start_goal(self, slices, max_planner_iterations):
    #     # get the joint position for the start state, and the related slice
    #     start_ray, start_slice_index = self._select_random_ray(slices)
    #     self._print_variable('start_ray', start_ray)
    #     self._print_variable('start_slice_index', start_slice_index)
    #     start_joints = self._get_random_joints(start_ray)
    #     self._print_variable('start_joints', start_joints)
    #     # get the joint position for the goal state while ignoring the start slice
    #     goal_ray, goal_slice_index = TrajectoryGenerator._select_random_ray(slices, start_slice_index)
    #     self._print_variable('goal_ray', goal_ray)
    #     self._print_variable('goal_slice_index', goal_slice_index)
    #     goal_joints = self._get_random_joints(goal_ray)
    #     self._print_variable('goal_joints', goal_joints)
    #     return self.environment.plan(start_joints, goal_joints, max_planner_iterations)
    def plan_start_goal(self, slices, max_planner_iterations):
        start_joints, start_slice = self._get_valid_joints(slices, None)
        self._print_variable('start_slice_index', start_slice)
        # get the joint position for the goal state while ignoring the start slice
        goal_joints, goal_slice = self._get_valid_joints(slices, start_slice)
        self._print_variable('goal_slice_index', goal_slice)
        return self.environment.plan(start_joints, goal_joints, max_planner_iterations)

    def _print_variable(self, name, variable):
        if self.print_info:
            print name, variable

    def _get_valid_joints(self, slices, forbidden_slice=None):
        while True:
            joints = self.environment.get_random_joints({0: 0.0})
            while not self.environment.is_valid(joints):
                joints = self.environment.get_random_joints({0: 0.0})
            target_pose = self.environment.get_target_pose(joints)
            target_angle = WorkspaceGenerator.center_to_ray_angle(target_pose)
            target_slice = [i for i in range(len(slices)-1) if slices[i] <= target_angle <= slices[i+1]][0]
            if forbidden_slice is None or target_slice != forbidden_slice:
                return joints, target_slice

    #
    # def _get_random_joints(self, joint1_position):
    #     joint_bounds = self.environment.get_joint_bounds()
    #     joints = [self.joint0_position, -(joint1_position - np.pi / 2.0)] + [
    #         uniform(joint_bounds[0][i], joint_bounds[1][i]) for i in
    #         range(2, self.environment.get_number_of_joints())]
    #     joints = self.environment.truncate_joints(joints)
    #     return tuple(joints)

    @staticmethod
    def _select_random_ray(slices_bounds, ignore_slice=None):
        # compute the upper bounds per slice
        upper_bounds = []
        reduce_bounds = 0.0
        for i in range(len(slices_bounds)-1):
            upper_bound = slices_bounds[i+1]
            if ignore_slice is not None:
                if i == ignore_slice:
                    reduce_bounds = upper_bound - slices_bounds[i]
                    continue
            upper_bounds.append(upper_bound - reduce_bounds)
        # get a random ray
        random_ray_selection = uniform(slices_bounds[0], upper_bounds[-1])
        # find the slice index of the ray
        slice_index = -1
        for i in range(len(upper_bounds)):
            if random_ray_selection <= upper_bounds[i]:
                slice_index = i
                break
        # if the slice index is greater than ignore_slice,
        # we also need to add the weight of the ignored slice in order to skip the range
        if ignore_slice is not None and slice_index >= ignore_slice:
            random_ray_selection += reduce_bounds
            slice_index += 1
        return random_ray_selection, slice_index
