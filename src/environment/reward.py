

class RewardManager:
    """
    Manages reward calculations for the simulation.

    .. todo:: figure out a better way for stages
    """
    def __init__(self, finder, edge_position, sumo):
        """
        Initializes the RewardManager with the given parameters.

        Args:
            finder: StopFinder instance.
            edge_dic (dict): Dictionary of edge locations.
            sumo: SUMO simulation instance.
        """
        

        self.finder = finder
        self.edge_position = edge_position
        self.sumo = sumo

    def update_stage(self, vedge,  vehicle, final_edge):
        """
        Updates the stage of the simulation.

        Args:
            stage (str): Current stage.
            destination_edge (str): Destination edge ID.
            vedge (str): Current vehicle edge ID.
            person: Person instance.
            vehicle: Vehicle instance.
            final_destination (str): Final destination edge ID.

        Returns:
            tuple: Updated stage, destination edge, pickup status, and done flag.
        """
     

        if vedge == vehicle.current_destination:

            match vehicle.current_stage:

                case 0:
                    
                    vehicle.update_stage(1)

                case 1:
                    vehicle.update_stage(2)
                    vehicle.teleport(vehicle.current_destination)

                    self.sumo.simulationStep()
                    vehicle.update_stage(3)


                case 3:
                    vehicle.update_stage(4)
        
        elif vedge == final_edge and vehicle.picked_up == 1:
             print('Skipped Bus')
             vehicle.update_stage(4)

        return vehicle.current_stage, vehicle.current_destination, vehicle.picked_up
    
    # def get_initial_stage(self):
    #     """
    #     Gets the initial stage of the simulation.

    #     Returns:
    #         str: Initial stage of the simulation.
    #     """

    #     return vehicle.current_stage
    
    def distance_checks(self,vehicle):
        

        if vehicle.destination_old_distance > vehicle.destination_distance:
            
            vehicle.distcheck, = 1
        elif vehicle.destination_old_distance < vehicle.destination_distance:
            
            vehicle.distcheck, = 0
            
        if  vehicle.final_destination_old_distance> vehicle.final_destination_distance:
            
            vehicle.distcheck_final = 1  * vehicle.picked_up
        elif vehicle.final_destination_old_distance < vehicle.final_destination_distance:
            
           vehicle.distcheck_final = 0
     
        return
