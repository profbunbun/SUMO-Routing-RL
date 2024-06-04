""" Observation Class"""
from .outmask import OutMask


class Observation:
    """
    Class to handle observations in the SUMO environment.
    """
  
    def __init__(self):
        """
        Initialize the Observation class with an OutMask instance.
        """

        self.out_mask = OutMask()

    
    def get_state(self, sumo, step, vehicle, destination_loc, life, distcheck, final_destination, distcheck_final_dest, picked_up, done):
        """
        Get the current state of the environment.

        Args:
            sumo: SUMO simulation instance.
            step: Current simulation step.
            vehicle: Vehicle object.
            destination_loc: Destination location coordinates.
            life: Agent's current life value.
            distcheck: Distance check flag.
            final_destination: Final destination coordinates.
            distcheck_final_dest: Final destination distance check flag.
            picked_up: Pickup status flag.
            done: Done flag.

        Returns:
            list: State representation of the environment.
        """
      
        bounding_box = sumo.simulation.getNetBoundary()
        self.min_x, self.min_y = bounding_box[0]
        self.max_x, self.max_y = bounding_box[1]

        # pad max and min with 1 to allow for vehicles to be on the edge
        self.max_manhat_dist = abs((self.max_x + 1) - (self.min_x - 1)) + abs((self.max_y + 1) - (self.min_y - 1))
        
        vloc = vehicle.location()
        choices = vehicle.get_out_dict()
        state = []
        
        state.append(self.normalize(vloc[0], self.min_x, self.max_x))
        state.append(self.normalize(vloc[1], self.min_y, self.max_y))
        state.append(self.normalize(destination_loc[0], self.min_x, self.max_x))
        state.append(self.normalize(destination_loc[1], self.min_y, self.max_y))
        state.append(self.normalize(final_destination[0], self.min_x, self.max_x))
        state.append(self.normalize(final_destination[1], self.min_y, self.max_y))
        state.append(step * 0.0001)
        state.append(self.manhat_dist(vloc[0], vloc[1], destination_loc[0], destination_loc[1]))  
        state.extend(self.out_mask.get_outmask_valid(choices))
        state.append(life)
        state.append(distcheck)
        state.append(distcheck_final_dest)
        state.append(picked_up)
        state.append(done)

        return state

    def manhat_dist(self, x1, y1, x2, y2):
        """
        Calculate and normalize the Manhattan distance between two points.

        Args:
            x1, y1: Coordinates of the first point.
            x2, y2: Coordinates of the second point.

        Returns:
            float: Normalized Manhattan distance.
        """
        distance = abs(x1 - x2) + abs(y1 - y2)
        return distance / self.max_manhat_dist
    
    def normalize(self, value, min_value, max_value):
        """
        Normalize a value within a given range.

        Args:
            value: Value to be normalized.
            min_value: Minimum value of the range.
            max_value: Maximum value of the range.

        Returns:
            float: Normalized value.
        """
        range = max_value - min_value if max_value - min_value != 0 else 1
        return (value - min_value) / range

