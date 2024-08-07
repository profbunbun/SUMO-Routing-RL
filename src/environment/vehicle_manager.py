from .vehicle import Vehicle

class VehicleManager:
    """
    Manages the creation and management of vehicles in the simulation.

    Attributes:
        num_of_vehicles (int): Number of vehicles to create.
        edge_position (dict): Dictionary of edge positions.
        sumo: SUMO simulation instance.
        out_dict (dict): Dictionary of outgoing edges.
        index_dict (dict): Dictionary of edge indices.
        config (dict): Configuration dictionary.
        disability_categories (list): List of disability categories.
    """

    def __init__(self, num_of_vehicles, edge_position, sumo, out_dict, index_dict, config):
        """
        Initialize the VehicleManager with the given parameters.

        Args:
            num_of_vehicles (int): Number of vehicles to create.
            edge_position (dict): Dictionary of edge positions.
            sumo: SUMO simulation instance.
            out_dict (dict): Dictionary of outgoing edges.
            index_dict (dict): Dictionary of edge indices.
            config (dict): Configuration dictionary.
        """
      
        self.num_of_vehicles = num_of_vehicles
        self.edge_position = edge_position
        self.sumo = sumo
        self.out_dict = out_dict
        self.index_dict = index_dict
        self.config = config
        self.types = self.config['env']['types_of_passengers']
        self.life = self.config['training_settings']['initial_life']


    def create_vehicles(self):
        """
        Creates vehicles for the simulation.

        Returns:
            list: List of created Vehicle instances.
        """
     
        vehicles = []
        for v_id in range(self.num_of_vehicles):
            modulo_type=v_id + 1
            vehicle_type = (modulo_type % self.types) + 1
            vehicles.append(
                Vehicle(
                    str(v_id),
                    vehicle_type,
                    self.out_dict,
                    self.index_dict,
                    self.edge_position,
                    self.sumo,
                    self.life
                    
                )
            )
        # vedges = [vehicle.get_road() for vehicle in vehicles]
        return vehicles
