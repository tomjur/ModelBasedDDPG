import os
import tensorflow as tf
import tensorflow.contrib.layers as layers

from dqn_model import DqnModel
from modeling_utils import get_activation
from openrave_manager import OpenraveManager
from potential_point import PotentialPoint


class Network(object):
    def __init__(self, config, is_rollout_agent, image_shape=(445, 222, 3), number_of_joints=4, pose_dimensions=2):
        self.config = config
        tau = self.config['model']['tau']
        gamma = self.config['model']['gamma']

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

        # since we take partial derivatives w.r.t subsets of the parameters, we always need to remember which parameters
        # are currently being added. note that this also causes the model to be non thread safe, therefore the creation
        # must happen sequentially
        variable_count = len(tf.trainable_variables())

        # online actor network
        actor_results = self._create_actor_network(self.joints_inputs, is_online=True, reuse_flag=False)
        self.online_action = actor_results[0]
        online_actor_tanh = actor_results[1]
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

        use_reward_model = self.config['model']['use_reward_model']
        forward_model_next_state, forward_model_action, forward_model_tanh = None, None, None
        if use_reward_model:
            # deterministic value of the next state (from current state, executing the online action)
            forward_model_next_state = self._next_state_model() if use_reward_model else None

            # online actor network for the result of the forward model
            variable_count = len(tf.trainable_variables())
            actor_results = self._create_actor_network(forward_model_next_state, is_online=True, reuse_flag=True)
            forward_model_action = actor_results[0]
            forward_model_tanh = actor_results[1]
            assert variable_count == len(tf.trainable_variables())  # make sure no new parameters were added

        # target actor network
        variable_count = len(tf.trainable_variables())
        actor_results = self._create_actor_network(self.joints_inputs, is_online=False, reuse_flag=False)
        self.target_action = actor_results[0]
        target_actor_params = tf.trainable_variables()[variable_count:]

        # periodically update target actor with online actor weights
        self.update_actor_target_params = \
            [target_actor_params[i].assign(
                tf.multiply(self.online_actor_params[i], tau) + tf.multiply(target_actor_params[i], 1. - tau)
            ) for i in range(len(target_actor_params))]

        # create inputs for the critic and reward network when using a constant action
        self.action_inputs = tf.placeholder_with_default(input=[[0.0]*4], shape=[None, self.number_of_joints])

        # online critic for predicting the q value for a specific joints+action pair
        variable_count = len(tf.trainable_variables())
        self.online_q_value_fixed_action = self._create_critic_network(
            self.joints_inputs, self.action_inputs, is_online=True, reuse_flag=False, add_regularization_loss=True
        )
        online_critic_params = tf.trainable_variables()[variable_count:]

        # online critic for predicting the q value for actor update.
        # if using a reward model, the joint inputs are given by the forward model and so are the actions.
        # if in regular ddpg, the joints inputs are given by the current state inputs, the actions are the policy on
        # these jonts.
        variable_count = len(tf.trainable_variables())
        self.online_q_value_under_policy = self._create_critic_network(
            joints_input=forward_model_next_state if use_reward_model else self.joints_inputs,
            action_input=forward_model_action if use_reward_model else self.online_action,
            is_online=True, reuse_flag=True, add_regularization_loss=False
        )
        assert variable_count == len(tf.trainable_variables())  # make sure no new parameters were added

        # target critic network, predicting the q value current state under the target policy
        variable_count = len(tf.trainable_variables())
        self.target_q_value_under_policy = self._create_critic_network(
            self.joints_inputs, self.target_action, is_online=False, reuse_flag=False, add_regularization_loss=False
        )
        target_critic_params = tf.trainable_variables()[variable_count:]

        # periodically update target critic with online critic weights
        self.update_critic_target_params = \
            [target_critic_params[i].assign(
                tf.multiply(online_critic_params[i], tau) + tf.multiply(target_critic_params[i], 1. - tau)
            ) for i in range(len(target_critic_params))]

        self.fixed_action_reward, self.online_action_reward = None, None
        if use_reward_model:
            # reward network to predict the immediate reward of a given action
            self.fixed_action_reward = self._create_reward_network(
                self.joints_inputs, self.action_inputs, reuse_flag=False)
            # reward network to predict the immediate reward of the online policy action
            self.online_action_reward = self._create_reward_network(
                self.joints_inputs, self.online_action, reuse_flag=True)

        # the label to use to train the online critic network or reward network
        self.scalar_label = tf.placeholder(tf.float32, [None, 1])

        batch_size = tf.cast(tf.shape(self.goal_joints_inputs)[0], tf.float32)

        self.optimize_reward, self.reward_optimization_summaries = None, None
        if use_reward_model:
            # reward network optimization
            reward_loss = tf.div(tf.losses.mean_squared_error(self.scalar_label, self.fixed_action_reward), batch_size)
            reward_initial_gradients_norm, reward_clipped_gradients_norm, self.optimize_reward = self._optimize_by_loss(
                reward_loss, online_critic_params, self.config['reward']['learning_rate'],
                self.config['reward']['gradient_limit']
            )
            # summaries for the reward optimization
            self.reward_optimization_summaries = tf.summary.merge([
                tf.summary.scalar('reward_loss', reward_loss),
                tf.summary.scalar('reward_gradients_norm_initial', reward_initial_gradients_norm),
                tf.summary.scalar('reward_gradients_norm_clipped', reward_clipped_gradients_norm),
            ])

        # critic optimization
        critic_prediction_loss = tf.div(
            tf.losses.mean_squared_error(self.scalar_label, self.online_q_value_fixed_action), batch_size)
        critic_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        critic_regularization_loss = tf.div(tf.add_n(critic_regularization), batch_size) \
            if len(critic_regularization) > 0 else 0.0
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

        # when training the actor we derive the advantage w.r.t mu's network params (mu is the online policy)
        if use_reward_model:
            # advantage is r(s, mu(s)) + \gamma * q(f(s, mu(s)), mu(f(s, mu(s))))
            actor_loss = -tf.squeeze(
                self.online_action_reward + gamma * self.online_q_value_under_policy
                # this is actually the policy on the forward model output
            )
        else:
            # advantage is q(s, mu(s))
            actor_loss = -tf.squeeze(self.online_q_value_under_policy)
        # if we have extra losses for the actor:
        tanh_loss_summary = None
        if self.config['action_predictor']['tanh_preactivation_loss_coefficient'] > 0.0:
            tanh_preactivation_loss = tf.losses.mean_squared_error(
                tf.zeros_like(online_actor_tanh), online_actor_tanh
            )
            if use_reward_model:
                forward_model_tanh_preactivation_loss = tf.losses.mean_squared_error(
                    tf.zeros_like(forward_model_tanh), forward_model_tanh
                )
                tanh_preactivation_loss += forward_model_tanh_preactivation_loss
            tanh_preactivation_loss *= self.config['action_predictor']['tanh_preactivation_loss_coefficient']
            actor_loss += tanh_preactivation_loss
            tanh_loss_summary = tf.summary.scalar('tanh_preactivation_loss', tanh_preactivation_loss)

        # divide by the batch size
        actor_loss = tf.div(actor_loss, batch_size)

        actor_initial_gradients_norm, actor_clipped_gradients_norm, self.optimize_actor = self._optimize_by_loss(
            actor_loss, self.online_actor_params, self.config['actor']['learning_rate'],
            self.config['actor']['gradient_limit']
        )

        # summaries for the optimization
        merge_list = [
            tf.summary.scalar('actor_gradients_norm_initial', actor_initial_gradients_norm),
            tf.summary.scalar('actor_gradients_norm_clipped', actor_clipped_gradients_norm),
        ]
        if tanh_loss_summary is not None:
            merge_list.append(tanh_loss_summary)
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

    def _generate_policy_features(self, current_joints, include_goal_information=True):
        features = [current_joints]
        if include_goal_information:
            features.append(self.goal_joints_inputs)
            # features.append(self.goal_joints_inputs - current_joints)
        if self.workspace_image_inputs is not None:
            perception = DqnModel()
            features.append(perception.predict(self.workspace_image_inputs))
        if self.goal_pose_inputs is not None and include_goal_information:
            features.append(self.goal_pose_inputs)
        return tf.concat(features, axis=1)

    def _next_state_model(self):
        # next step is a deterministic computation
        action_step_size = self.config['openrave_rl']['action_step_size']
        step = self.online_action * action_step_size
        result = self.joints_inputs + step
        # we initiate an openrave manager to get the robot, to get the joint bounds and the safety
        openrave_manager = OpenraveManager(0.0, self.potential_points)
        joint_bounds = openrave_manager.get_joint_bounds()
        joint_safety = openrave_manager.joint_safety
        lower_bounds = joint_bounds[0] + joint_safety
        upper_bounds = joint_bounds[1] - joint_safety
        # clip the result
        result = tf.maximum(result, lower_bounds)
        result = tf.minimum(result, upper_bounds)
        return result

    def _create_actor_network(self, joints_input, is_online, reuse_flag):
        name_prefix = '{}_actor_{}'.format(os.getpid(), 'online' if is_online else 'target')
        hidden_layers_after_combine = self.config['action_predictor']['layers']
        activation = get_activation(self.config['action_predictor']['activation'])
        layers = hidden_layers_after_combine + [self.number_of_joints]
        current = self._generate_policy_features(joints_input)
        for i, layer_size in enumerate(layers[:-1]):
            current = tf.layers.dense(
                current, layer_size, activation=activation, name='{}_{}'.format(name_prefix, i), reuse=reuse_flag
            )
        tanh_preactivation = tf.layers.dense(
            current, layers[-1], activation=None, name='{}_tanh'.format(name_prefix), reuse=reuse_flag
        )
        action = tf.nn.l2_normalize(tf.nn.tanh(tanh_preactivation), 1)

        return action, tanh_preactivation

    def _create_critic_network(self, joints_input, action_input, is_online, reuse_flag, add_regularization_loss):
        name_prefix = '{}_critic_{}'.format(os.getpid(), 'online' if is_online else 'target')
        layers_before_action = self.config['critic']['layers_before_action']
        layers_after_action = self.config['critic']['layers_after_action'] + [1]
        activation = get_activation(self.config['critic']['activation'])

        current = self._generate_policy_features(joints_input)
        scale = self.config['critic']['l2_regularization_coefficient'] if add_regularization_loss else 0.0
        for i, layer_size in enumerate(layers_before_action):
            current = tf.layers.dense(
                current, layer_size, activation=activation, name='{}_before_action_{}'.format(name_prefix, i),
                reuse=reuse_flag, kernel_regularizer=layers.l2_regularizer(scale)
            )
        current = tf.concat((current, action_input), axis=1)
        for i, layer_size in enumerate(layers_after_action):
            _activation = None if i == len(layers_after_action) - 1 else activation
            current = tf.layers.dense(
                current, layer_size, activation=_activation, name='{}_after_action_{}'.format(name_prefix, i),
                reuse=reuse_flag, kernel_regularizer=layers.l2_regularizer(scale)
            )
        return current

    def _create_reward_network(self, joints_input, action_input, reuse_flag):
        name_prefix = '{}_reward'.format(os.getpid())
        layers_before_action = self.config['reward']['layers_before_action']
        layers_after_action = self.config['reward']['layers_after_action'] + [1]
        activation = get_activation(self.config['reward']['activation'])

        current = self._generate_policy_features(joints_input, include_goal_information=False)
        for i, layer_size in enumerate(layers_before_action):
            current = tf.layers.dense(
                current, layer_size, activation=activation, name='{}_before_action_{}'.format(name_prefix, i),
                reuse=reuse_flag
            )
        current = tf.concat((current, action_input), axis=1)
        for i, layer_size in enumerate(layers_after_action):
            _activation = None if i == len(layers_after_action) - 1 else activation
            current = tf.layers.dense(
                current, layer_size, activation=_activation, name='{}_after_action_{}'.format(name_prefix, i),
                reuse=reuse_flag
            )
        return current

    def train_critic(
            self, joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, action_inputs, q_label,
            sess
    ):
        feed_dictionary = self._generate_feed_dictionary(
            joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, action_inputs
        )
        feed_dictionary[self.scalar_label] = q_label
        return sess.run([self.critic_optimization_summaries, self.optimize_critic], feed_dictionary)

    def train_reward(self, joint_inputs, workspace_image_inputs, action_inputs, observed_reward, sess):
        if self.optimize_reward is None:
            return None, None
        feed_dictionary = self._generate_feed_dictionary(
            joint_inputs, workspace_image_inputs, action_inputs=action_inputs
        )
        feed_dictionary[self.scalar_label] = observed_reward
        return sess.run([self.reward_optimization_summaries, self.optimize_reward], feed_dictionary)

    def train_actor(self, joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, sess):
        feed_dictionary = self._generate_feed_dictionary(
            joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs
        )
        return sess.run([self.actor_optimization_summaries, self.optimize_actor], feed_dictionary)

    def predict_policy_q(
            self, joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, sess, use_online_network,
    ):
        feed_dictionary = self._generate_feed_dictionary(
            joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs
        )
        return sess.run(
            self.online_q_value_under_policy if use_online_network else self.target_q_value_under_policy,
            feed_dictionary
        )

    def predict_fixed_action_q(
            self, joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, action_inputs, sess
    ):
        feed_dictionary = self._generate_feed_dictionary(
            joint_inputs, workspace_image_inputs, goal_pose_inputs, goal_joints_inputs, action_inputs
        )
        return sess.run(self.online_q_value_fixed_action, feed_dictionary)

    def predict_policy_reward(self, joint_inputs, workspace_image_inputs, sess):
        if self.online_action_reward is None:
            return None
        feed_dictionary = self._generate_feed_dictionary(joint_inputs, workspace_image_inputs)
        return sess.run(self.online_action_reward, feed_dictionary)

    def predict_fixed_action_reward(self, joint_inputs, workspace_image_inputs, action_inputs, sess):
        if self.fixed_action_reward is None:
            return None
        feed_dictionary = self._generate_feed_dictionary(
            joint_inputs, workspace_image_inputs, action_inputs=action_inputs
        )
        return sess.run(self.fixed_action_reward, feed_dictionary)

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
            self, joints_inputs, workspace_image_inputs, goal_pose_inputs=None, goal_joints_inputs=None, action_inputs=None
    ):
        feed_dictionary = {
            self.joints_inputs: joints_inputs,
        }
        if goal_joints_inputs is not None:
            feed_dictionary[self.goal_joints_inputs] = goal_joints_inputs
        if action_inputs is not None:
            feed_dictionary[self.action_inputs] = action_inputs
        if self.workspace_image_inputs is not None:
            feed_dictionary[self.workspace_image_inputs] = workspace_image_inputs
        if self.goal_pose_inputs is not None and goal_pose_inputs is not None:
            feed_dictionary[self.goal_pose_inputs] = goal_pose_inputs
        return feed_dictionary
