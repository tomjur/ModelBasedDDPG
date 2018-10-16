from baseline_action_predictor import BaselineActionPredictor


def get_network(config, is_rollout_agent):
    if config['model']['actor'] == 'action_predictor':
        return BaselineActionPredictor(config, is_rollout_agent)
    # elif model_type == 'force_predictor':
    #     from baseline_direction_magnitude_force_predictor import BaselineDirectionMagnitudeForcePredictor
    #     return BaselineDirectionMagnitudeForcePredictor(config, workspace_image_shape, joints_configuration_size, pose_size)
    assert False