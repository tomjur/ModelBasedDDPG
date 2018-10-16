class ModelInputs:
    _instance = None

    def __init__(self, config):
        self.consider_current_joints = config['model']['consider_current_joints']
        self.consider_image = config['model']['consider_image']
        self.consider_current_pose = config['model']['consider_current_pose']
        self.consider_current_jacobian = config['model']['consider_current_jacobian']

    @staticmethod
    def from_config(config):
        if ModelInputs._instance is None:
            ModelInputs._instance = ModelInputs(config)
        return ModelInputs._instance
