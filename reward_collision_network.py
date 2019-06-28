import os
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from dqn_model import DqnModel
from modeling_utils import get_activation


class CollisionNetwork:

    def __init__(self, config, model_dir):
        self._reuse_flag = False

        self.config = config
        self.is_vision_enabled = 'vision' in config['general']['scenario']

        self.joints_inputs = tf.placeholder(tf.float32, (None, 4), name='joints_inputs')
        self.action_inputs = tf.placeholder(tf.float32, (None, 4), name='action_inputs')

        self.workspace_image_inputs, self.images_3d = None, None
        if self.is_vision_enabled:
            self.workspace_image_inputs = tf.placeholder(tf.float32, (None, 55, 111), name='workspace_image_inputs')
            self.images_3d = tf.expand_dims(self.workspace_image_inputs, axis=-1)

        current_variables_count = len(tf.trainable_variables())
        self.status_softmax_logits = self._create_network(self.joints_inputs, self.action_inputs, self.images_3d)
        self.collision_variables = tf.trainable_variables()[current_variables_count:]

        # model path to load
        self.model_dir = model_dir
        assert os.path.exists(self.model_dir)
        self.saver_path = os.path.join(self.model_dir, "model_saver")
        self.saver = tf.train.Saver(self.collision_variables, max_to_keep=4, save_relative_paths=True)

    def _next_state_model(self, joints_inputs, action_inputs):
        # next step is a deterministic computation
        action_step_size = self.config['openrave_rl']['action_step_size']
        step = action_inputs * action_step_size
        unclipped_result = joints_inputs + step
        # we initiate an openrave manager to get the robot, to get the joint bounds and the safety
        joint_safety = 0.0001
        lower_bounds = [-2.617, -1.571, -1.571, -1.745, -2.617]
        lower_bounds = [b + joint_safety for b in lower_bounds[1:]]
        upper_bounds = [-b for b in lower_bounds]

        # clip the result
        clipped_result = tf.maximum(unclipped_result, lower_bounds)
        clipped_result = tf.minimum(clipped_result, upper_bounds)
        return clipped_result, unclipped_result

    def _create_network(self, joints_inputs, action_inputs, images_3d):
        name_prefix = 'reward'

        # get L2 regularization scale
        l2_scale = 0.0
        if 'l2_regularization_coefficient' in self.config['reward']:
            l2_scale = self.config['reward']['l2_regularization_coefficient']

        # get the next joints
        clipped_next_joints, unclipped_next_joints = self._next_state_model(joints_inputs, action_inputs)
        current = tf.concat((joints_inputs, clipped_next_joints), axis=1)

        # add vision if needed
        if self.is_vision_enabled:
            visual_inputs = DqnModel(name_prefix).predict(images_3d, self._reuse_flag)
            current = tf.concat((current, visual_inputs), axis=1)

        layers = self.config['reward']['layers'] + [2]
        for i, layer_size in enumerate(layers):
            _activation = get_activation(self.config['reward']['activation']) if i < len(layers) - 1 else None
            current = tf.layers.dense(
                current,
                layer_size,
                activation=_activation,
                name='{}_layers_{}'.format(name_prefix, i),
                kernel_regularizer=tf_layers.l2_regularizer(l2_scale),
                reuse=self._reuse_flag
            )
        softmax_logits = current

        self._reuse_flag = True
        return softmax_logits

    def save_weights(self, sess, global_step=None):
        self.saver.save(sess, self.saver_path, global_step=global_step)

    def load_weights(self, sess):
        self.saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))

    def make_feed(self, all_start_joints, all_actions, images=None):
        feed = {
            self.joints_inputs: all_start_joints,
            self.action_inputs: all_actions,
        }
        if self.is_vision_enabled:
            assert images is not None
            assert images[0] is not None
            feed[self.workspace_image_inputs] = images
        return feed
