

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
    
    def calculate_reward(self, old_dist, edge_distance, final_old_dist, final_edge_distance,vehicle):
        """
        Calculates the reward for the current step.

        Args:
            old_dist (float): Previous distance to the destination.
            edge_distance (float): Current distance to the destination.
            destination_edge (str): Destination edge ID.
            vedge (str): Current vehicle edge ID.
            life (float): Agent's current life value.
            final_destination (str): Final destination edge ID.
            final_old_dist (float): Previous distance to the final destination.
            final_edge_distance (float): Current distance to the final destination.

        Returns:
            tuple: Calculated reward, updated distance check, life value, and final distance check.
        """
        
        reward = 0
        distcheck = 0
        distcheck_final = 0

        if old_dist > edge_distance:
            # reward = 0.02
            distcheck = 1
        elif old_dist < edge_distance:
            # reward = -0.025
            distcheck = 0
            
        if  final_old_dist > final_edge_distance:
            # reward = 0.02 * self.pickedup
            distcheck_final = 1  * vehicle.picked_up
        elif final_old_dist < final_edge_distance:
            # reward = -0.025 * self.pickedup
            distcheck_final = 0
     

        # if vedge == destination_edge:
        #     life += 0.1
        #     # reward = 0.8
        #     # make_choice_flag = True

        # if vedge == final_destination and self.pickedup == 1:
        #     life += 0.1
        #     # reward = 0.8
        #     # make_choice_flag = False

        return reward,  distcheck,  distcheck_final
