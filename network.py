import tensorflow as tf
import tensorflow.contrib.layers as layers

from dqn_model import DqnModel
from modeling_utils import get_activation
from potential_point import PotentialPoint


class Network(object):
    def __init__(self, config, is_rollout_agent, image_shape=(445, 222, 3), number_of_joints=4, pose_dimensions=2):
        self.config = config
        tau = self.config['model']['tau']

        self.potential_points = PotentialPoint.from_config(config)

        # input related data
        self.image_shape = image_shape
        self.number_of_joints = number_of_joints
        self.pose_dimensions = pose_dimensions

        # generate inputs
        all_inputs = self._create_inputs()
        self.joints_inputs = all_inputs[0]
        self.workspace_image_inputs = all_inputs[1]
        self.goal_joints_inputs = all_inputs[2]
        self.goal_pose_inputs = all_inputs[3]

        self.network_features = self._generate_features()

        # since we take partial derivatives w.r.t subsets of the parameters, we always need to remember which parameters
        # are currently being added. note that this also causes the model to be non thread safe, therefore the creation
        # must happen sequentially
        variable_count = len(tf.trainable_variables())

        # online actor network
        actor_results = self._create_actor_network(True, True)
        self.online_action = actor_results[0]
        actor_extra_losses = actor_results[1]
        actor_summaries = actor_results[2]
        self.online_actor_params = tf.trainable_variables()[variable_count:]

        # create placeholders and assign ops to set these weights manually (used by rollout agents)
        self.online_actor_parameter_weights_placeholders = {
            var.name: tf.placeholder(tf.float32, var.get_shape()) for var in self.online_actor_params
        }
        self.online_actor_parameters_assign_ops = [
            tf.assign(var, self.online_actor_parameter_weights_placeholders[var.name])
            for var in self.online_actor_params
        ]

        # this is as much as a rollout agent needs
        if is_rollout_agent:
            return

        # target actor network
        variable_count = len(tf.trainable_variables())
        actor_results = self._create_actor_network(False, False)
        self.target_action = actor_results[0]
        assert actor_results[1] is None
        assert actor_results[2] is None
        target_actor_params = tf.trainable_variables()[variable_count:]

        # periodically update target actor with online actor weights
        self.update_actor_target_params = \
            [target_actor_params[i].assign(
                tf.multiply(self.online_actor_params[i], tau) + tf.multiply(target_actor_params[i], 1. - tau)
            ) for i in range(len(target_actor_params))]

        # create inputs for the critic
        # when using a constant action
        self.action_inputs = tf.placeholder_with_default(input=[[0.0]*4], shape=[None, self.number_of_joints])
        # a flag that indicates if we need to use the policy, or a constant action
        self.use_policy = tf.placeholder(dtype=tf.bool)

        # online critic network
        variable_count = len(tf.trainable_variables())
        self.online_q_value = self._create_critic_network(True)
        online_critic_params = tf.trainable_variables()[variable_count:]

        # target critic network
        variable_count = len(tf.trainable_variables())
        self.target_q_value = self._create_critic_network(False)
        target_critic_params = tf.trainable_variables()[variable_count:]

        # periodically update target critic with online critic weights
        self.update_critic_target_params = \
            [target_critic_params[i].assign(
                tf.multiply(online_critic_params[i], tau) + tf.multiply(target_critic_params[i], 1. - tau)
            ) for i in range(len(target_critic_params))]

        # the label to use to train the online critic network
        self.q_label = tf.placeholder(tf.float32, [None, 1])

        # critic optimization
        critic_prediction_loss = tf.losses.mean_squared_error(self.q_label, self.online_q_value)
        critic_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        critic_regularization_loss = tf.add_n(critic_regularization) if len(critic_regularization) > 0 else 0.0
        critic_total_loss = critic_prediction_loss + critic_regularization_loss

        critic_initial_gradients_norm, critic_clipped_gradients_norm, self.optimize_critic = self._optimize_by_loss(
            critic_total_loss, online_critic_params, self.config['critic']['learning_rate'],
            self.config['critic']['gradient_limit']
        )

        # summaries for the critic optimization
        self.critic_optimization_summaries = tf.summary.merge([
            tf.summary.scalar('critic_prediction_loss', critic_prediction_loss),
            tf.summary.scalar('critic_regularization_loss', critic_regularization_loss),
            tf.summary.scalar('critic_total_loss', critic_total_loss),
            tf.summary.scalar('critic_gradients_norm_initial', critic_initial_gradients_norm),
            tf.summary.scalar('critic_gradients_norm_clipped', critic_clipped_gradients_norm),
        ])

        # when training the actor we derive Q(s, mu(s)) w.r.t mu's network params (mu is the online policy)
        actor_loss = -tf.squeeze(self.online_q_value)
        # if we have extra losses for the actor:
        if actor_extra_losses is not None:
            actor_loss += actor_extra_losses
        # divide by the batch size
        batch_size = tf.shape(self.goal_joints_inputs)[0]
        actor_loss = tf.div(actor_loss, tf.cast(batch_size, tf.float32))

        actor_initial_gradients_norm, actor_clipped_gradients_norm, self.optimize_actor = self._optimize_by_loss(
            actor_loss, self.online_actor_params, self.config['actor']['learning_rate'],
            self.config['actor']['gradient_limit']
        )

        # summaries for the optimization
        merge_list = [
            tf.summary.scalar('actor_gradients_norm_initial', actor_initial_gradients_norm),
            tf.summary.scalar('actor_gradients_norm_clipped', actor_clipped_gradients_norm),
        ]
        if actor_summaries is not None:
            merge_list.append(actor_summaries)
        self.actor_optimization_summaries = tf.summary.merge(merge_list)

    @staticmethod
    def _optimize_by_loss(loss, parameters_to_optimize, learning_rate, gradient_limit):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss, parameters_to_optimize))
        initial_gradients_norm = tf.global_norm(gradients)
        if gradient_limit > 0.0:
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_limit, use_norm=initial_gradients_norm)
        clipped_gradients_norm = tf.global_norm(gradients)
        optimize_op = optimizer.apply_gradients(zip(gradients, variables))
        return initial_gradients_norm, clipped_gradients_norm, optimize_op

    def _create_inputs(self):
        joints_inputs = tf.placeholder(tf.float32, (None, self.number_of_joints), name='joints_inputs')
        goal_joints_inputs = tf.placeholder(tf.float32, (None, self.number_of_joints), name='goal_joints_inputs')

        # sometimes we don't want to get an image (single workspace)
        workspace_image_inputs = None
        if self.config['model']['consider_image']:
            workspace_image_inputs = tf.placeholder(tf.float32, (None,) + self.image_shape,
                                                    name='workspace_image_inputs')

        goal_pose_inputs = None
        if self.config['model']['consider_goal_pose']:
            goal_pose_inputs = tf.placeholder(tf.float32, (None, self.pose_dimensions), name='goal_pose_inputs')
        return joints_inputs, workspace_image_inputs, goal_joints_inputs, goal_pose_inputs

    def _generate_features(self):
        # features = [self.joints_inputs, self.goal_joints_inputs]
        features = [self.joints_inputs, self.goal_joints_inputs, self.goal_joints_inputs-self.joints_inputs]
        if self.workspace_image_inputs is not None:
            perception = DqnModel()
            features.append(perception.predict(self.workspace_image_inputs))
        if self.goal_pose_inputs is not None:
            features.append(self.goal_pose_inputs)
        return tf.concat(features, axis=1)

    def _create_actor_network(self, create_losses, create_summaries):
        hidden_layers_after_combine = self.config['action_predictor']['layers']
        activation = get_activation(self.config['action_predictor']['activation'])
        layers = hidden_layers_after_combine + [self.number_of_joints]
        current = self.network_features
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

        action = tf.nn.l2_normalize(current, 1)

        return action, extra_loss, actor_summaries

    def _create_critic_network(self, is_online):
        layers_before_action = self.config['critic']['layers_before_action']
        layers_after_action = self.config['critic']['layers_after_action'] + [1]
        activation = get_activation(self.config['critic']['activation'])

        current = self.network_features
        scale = self.config['critic']['l2_regularization_coefficient'] if is_online else 0.0
        for i, layer_size in enumerate(layers_before_action):
            # current = tf.layers.dense(current, layer_size, activation=activation)
            current = tf.layers.dense(current, layer_size, activation=activation,
                                      kernel_regularizer=layers.l2_regularizer(scale))
        # if the computation needs an action from the policy, use online if the critic is online and offline if critic
        # is offline
        policy_action = self.online_action if is_online else self.target_action
        # but the action may be fixed
        action_inputs = tf.cond(self.use_policy, lambda: policy_action, lambda: self.action_inputs)
        current = tf.concat((current, action_inputs), axis=1)
        for i, layer_size in enumerate(layers_after_action):
            _activation = None if i == len(layers_after_action) - 1 else activation
            # current = tf.layers.dense(current, layer_size, activation=_activation)
            # current = tf.layers.dense(current, layer_size, activation=activation)
            current = tf.layers.dense(current, layer_size, activation=_activation,
                                      kernel_regularizer=layers.l2_regularizer(scale))
        q_value = current

        return q_value

    def train_critic(self, joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, action_inputs, q_label, sess):
        feed_dictionary = self._generate_feed_dictionary(
            joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, action_inputs
        )
        feed_dictionary[self.use_policy] = False
        feed_dictionary[self.q_label] = q_label
        return sess.run([self.critic_optimization_summaries, self.optimize_critic], feed_dictionary)

    def train_actor(self, joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, sess):
        feed_dictionary = self._generate_feed_dictionary(
            joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs
        )
        feed_dictionary[self.use_policy] = True
        return sess.run([self.actor_optimization_summaries, self.optimize_actor], feed_dictionary)

    def predict_q(
            self, joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, sess, use_online_network,
            action_inputs=None
    ):
        feed_dictionary = self._generate_feed_dictionary(
            joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, action_inputs
        )
        feed_dictionary[self.use_policy] = action_inputs is None
        return sess.run(self.online_q_value if use_online_network else self.target_q_value, feed_dictionary)

    def predict_action(
            self, joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, sess, use_online_network
    ):
        feed_dictionary = self._generate_feed_dictionary(
            joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs
        )
        return sess.run(self.online_action if use_online_network else self.target_action, feed_dictionary)

    def get_actor_online_weights(self, sess):
        return sess.run(self.online_actor_params)

    def set_actor_online_weights(self, sess, weights):
        feed = {
            self.online_actor_parameter_weights_placeholders[var.name]: weights[i]
            for i, var in enumerate(self.online_actor_params)
        }
        sess.run(self.online_actor_parameters_assign_ops, feed)

    def update_target_networks(self, sess):
        sess.run([self.update_critic_target_params, self.update_actor_target_params])

    def _generate_feed_dictionary(
            self, joints_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, action_inputs=None
    ):
        feed_dictionary = {
            self.joints_inputs: joints_inputs,
            self.goal_joints_inputs: goal_joints_inputs,
        }
        if action_inputs is not None:
            feed_dictionary[self.action_inputs] = action_inputs
        if self.workspace_image_inputs is not None:
            feed_dictionary[self.workspace_image_inputs] = workspace_image_inputs
        if self.goal_pose_inputs is not None:
            feed_dictionary[self.goal_pose_inputs] = goal_pose_inputs
        return feed_dictionary
