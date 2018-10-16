import tensorflow as tf

from network import Network
from dqn_model import DqnModel
from modeling_utils import get_activation


class BaselineActionPredictor(Network):
    def __init__(self, config, is_rollout_agent, image_shape=(445, 222, 3), number_of_joints=4, pose_dimensions=2):
        super(BaselineActionPredictor, self).__init__(
            config, is_rollout_agent, image_shape, number_of_joints, pose_dimensions
        )

    def _predict_action(self, create_losses, create_summaries):

        hidden_layers_after_combine = self.config['action_predictor']['layers']
        activation = get_activation(self.config['action_predictor']['activation'])

        # features = [goal_joints_inputs]
        features = [self.goal_pose_inputs, self.goal_joints_inputs]
        if self.workspace_image_inputs is not None:
            perception = DqnModel()
            features.append(perception.predict(self.workspace_image_inputs))
        if self.joints_inputs is not None:
            features.append(self.joints_inputs)
            # features.append(goal_joints_inputs - joint_inputs)
        if self.pose_inputs is not None:
            features += self.pose_inputs.values()
        if self.jacobian_inputs is not None:
            features += [tf.layers.flatten(j) for j in self.jacobian_inputs.values()]

        layers = hidden_layers_after_combine + [self.number_of_joints]
        current = tf.concat(features, axis=1)
        extra_loss = None
        actor_summaries = None
        for i, layer_size in enumerate(layers):
            if i == len(layers) - 1:
                current = tf.layers.dense(current, layer_size, activation=None)
                if self.config['action_predictor']['tanh_preactivation_loss_coefficient'] > 0.0:
                    tanh_preactivation_loss = tf.losses.mean_squared_error(tf.zeros_like(current), current)
                    tanh_preactivation_loss *= self.config['action_predictor']['tanh_preactivation_loss_coefficient']
                    if create_losses:
                        extra_loss = tanh_preactivation_loss
                    if create_summaries:
                        actor_summaries = tf.summary.scalar('tanh_preactivation_loss', tanh_preactivation_loss)
                current = tf.nn.tanh(current)
            else:
                current = tf.layers.dense(current, layer_size, activation=activation)  # this was the last active
            # _activation = None if i == len(layers) - 1 else activation
            # current = tf.layers.dense(current, layer_size, activation=_activation)  # this was the last active
            # current = tf.layers.dense(current, layer_size, activation=activation)
            # out = tf.layers.dense(current, layer_size, activation=_activation)
            # current = out if i == len(layers)-1 else tf.concat((current, out), axis=1)
        return current, None, extra_loss, actor_summaries
