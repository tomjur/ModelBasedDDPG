import tensorflow as tf


def get_activation(activation):
    if activation == 'relu':
        return tf.nn.relu
    if activation == 'tanh':
        return tf.nn.tanh
    if activation == 'elu':
        return tf.nn.elu
    return None


def create_inputs(prefix, model_inputs, potential_points, number_of_joints, pose_dimensions, image_shape):
    joints_inputs = None
    if model_inputs.consider_current_joints:
        joints_inputs = tf.placeholder(tf.float32, (None, number_of_joints), name='{}_joints_inputs'.format(prefix))

    # sometimes we don't want to get an image (single workspace)
    workspace_image_inputs = None
    if model_inputs.consider_image:
        workspace_image_inputs = tf.placeholder(tf.float32, (None,) + image_shape,
                                                name='{}_workspace_image_inputs'.format(prefix))

    # not all models should get current pose
    pose_inputs = None
    if model_inputs.consider_current_pose:
        pose_inputs = {
            p.tuple: tf.placeholder(tf.float32, (None, pose_dimensions), name='{}_pose_inputs_{}'.format(prefix, p.str))
            for p in potential_points
        }

    # not all models should get current jacobian
    jacobian_inputs = None
    if model_inputs.consider_current_jacobian:
        jacobian_inputs = {
            p.tuple: tf.placeholder(tf.float32, (None, number_of_joints, pose_dimensions),
                                    name='{}_jacobian_inputs_{}'.format(prefix, p.str))
            for p in potential_points
        }

    # goal pose is always needed
    goal_pose_inputs = tf.placeholder(tf.float32, (None, pose_dimensions),
                                      name='{}_goal_pose_inputs'.format(prefix))
    goal_joints_inputs = tf.placeholder(tf.float32, (None, number_of_joints),
                                        name='{}_goal_joints_inputs'.format(prefix))
    return joints_inputs, workspace_image_inputs, pose_inputs, jacobian_inputs, goal_pose_inputs, goal_joints_inputs
