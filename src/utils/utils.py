import yaml



class Utils:


    @staticmethod
    def load_yaml_config(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


    @staticmethod
    def manhattan_distance(x1, y1, x2, y2):
        """
        Calculate the Manhattan distance between two points.

        :param x1: X-coordinate of the first point.
        :param y1: Y-coordinate of the first point.
        :param x2: X-coordinate of the second point.
        :param y2: Y-coordinate of the second point.
        :return: Manhattan distance between the points.
        """
        return abs(x1 - x2) + abs(y1 - y2)