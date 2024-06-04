import yaml



class Utils:
    """
    Utility class providing common utility functions.
    """


    @staticmethod
    def load_yaml_config(config_path):
        """
        Load YAML configuration from a file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: Loaded configuration.
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


    @staticmethod
    def manhattan_distance(x1, y1, x2, y2):
        """
        Calculate the Manhattan distance between two points.

        Args:
            x1 (float): X-coordinate of the first point.
            y1 (float): Y-coordinate of the first point.
            x2 (float): X-coordinate of the second point.
            y2 (float): Y-coordinate of the second point.

        Returns:
            float: Manhattan distance between the points.
        """
        return abs(x1 - x2) + abs(y1 - y2)