import tensorflow as tf
from dqn_model import DqnModel
from supervised_model import SupervisedModel


class BaselineDirectionMagnitudeForcePredictor(SupervisedModel):
    def __init__(self, config, workspace_image_size, joints_configuration_size, pose_size):
        # init super
        super(BaselineDirectionMagnitudeForcePredictor, self).__init__(
            config, workspace_image_size, joints_configuration_size, pose_size)

        self.magnitude_loss_coefficient = config['force_predictor']['magnitude_loss_coefficient']

        consider_image = self.workspace_image_inputs is not None
        consider_current_pose = self.current_pose_inputs is not None
        consider_current_joints = self.current_joints_inputs is not None
        consider_current_jacobians = self.current_jacobian_inputs is not None

        assert consider_current_joints or consider_current_pose, \
            'force predictor: either current joints or current pose must be included in all versions of this model'

        assert consider_current_jacobians, 'force predictor: must have jacobian inputs'

        hidden_layers_after_combine = self.config['force_predictor']['layers']

        activation = self._get_activation(self.config['force_predictor']['activation'])

        self.workspace_dqn = DqnModel(self.image_shape) if consider_image else None

        required_output = (1+self.pose_dimensions) * len(self.potential_points)
        initial_size = self.pose_dimensions  # goal pose
        if consider_current_joints:
            initial_size += self.number_of_joints
        if consider_image:
            initial_size += self.workspace_dqn.dqn_output_size
        if consider_current_pose:
            initial_size += self.pose_dimensions * len(self.potential_points)
        sizes = [initial_size] + hidden_layers_after_combine + [required_output]
        with tf.name_scope('baseline_force_predictor/var_creation'):
            self.dense_vars = [
                self._create_dense_layer_vars('global_distance_feature_combiner_dense{}'.format(i + 1), sizes[i],
                                              sizes[i + 1])
                for i in range(len(sizes) - 1)
            ]

        with tf.name_scope('baseline_force_predictor/prediction'):
            combined_features = self.goal_pose_inputs
            if consider_image:
                workspace_features = self.workspace_dqn.predict(self.workspace_image_inputs)
                combined_features = tf.concat((workspace_features, combined_features), axis=1)
            if consider_current_pose:
                flatten_poses = tf.concat(self.current_pose_inputs.values(), axis=1)
                combined_features = tf.concat((combined_features, flatten_poses), axis=1)
            if consider_current_joints:
                combined_features = tf.concat((combined_features, self.current_joints_inputs), axis=1)

            current = combined_features
            for i in range(len(self.dense_vars)):
                activation = None if i == len(self.dense_vars) - 1 else activation
                current = self._apply_dense_with_bias(current, self.dense_vars[i], activation=activation)

            split_result = tf.split(current, len(self.potential_points), axis=1)

        self.pose_force = {}
        magnitudes = []
        for i, p in enumerate(self.potential_points):
            direction, magnitude = tf.split(split_result[i], [self.pose_dimensions, 1], axis=1)
            direction = tf.nn.l2_normalize(direction, 1)

            magnitude = tf.square(magnitude)
            current_pose_force = direction * magnitude

            self.pose_force[p.tuple] = current_pose_force
            magnitudes.append(magnitude)

        if self.magnitude_loss_coefficient > 0.0:
            magnitude_loss = tf.reduce_sum(tf.add_n(magnitudes))
            magnitude_loss *= self.magnitude_loss_coefficient
            tf.losses.add_loss(magnitude_loss)
            tf.summary.scalar('magnitude_loss', magnitude_loss)

    def get_params(self):
        res = []
        for t in self.dense_vars:
            res += t
        if self.workspace_dqn is not None:
            res += self.workspace_dqn.get_params()
        return res
