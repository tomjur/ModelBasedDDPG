import numpy as np


class PotentialPoint:
    _instance = None

    def __init__(self, tuple_from_config):
        self.tuple = tuple_from_config
        self.link = tuple_from_config[0]
        self.x = tuple_from_config[1]
        self.z = tuple_from_config[2]
        self.coordinate = np.array([self.tuple[1], 0.0, self.tuple[2], 1.0])
        self.str = 'l{}_x{}_z{}'.format(self.link, self.x, self.z)

    @staticmethod
    def from_config(config):
        if PotentialPoint._instance is None:
            potential_points = config['model']['potential_points']
            potential_points = [tuple(potential_points[i:i + 3]) for i in range(0, len(potential_points), 3)]
            PotentialPoint._instance = [PotentialPoint(t) for t in potential_points]
        return PotentialPoint._instance
