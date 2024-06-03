from .vehicle import Vehicle

class VehicleManager:

    def __init__(self, num_of_vehicles, edge_position, sumo, out_dict, index_dict):
      
        self.num_of_vehicles = num_of_vehicles
        self.edge_position = edge_position
        self.sumo = sumo
        self.out_dict = out_dict
        self.index_dict = index_dict

    def create_vehicles(self):
     
        vehicles = []
        for v_id in range(self.num_of_vehicles):
            vehicles.append(
                Vehicle(
                    str(v_id),
                    self.out_dict,
                    self.index_dict,
                    self.edge_position,
                    self.sumo,
                    v_id + 1,
                )
            )
        return vehicles
