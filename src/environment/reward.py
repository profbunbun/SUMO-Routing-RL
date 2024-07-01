

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
        self.stage = "pickup"
        self.pickedup = 0

    def update_stage(self, current_stage, destination_edge, vedge, person, vehicle, final_edge, done):
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
        # done = 0

        new_stage = current_stage
        
        new_destination_edge = destination_edge

        fin = False

        if vedge == destination_edge:

            if current_stage == "pickup":
                new_stage = "picked up"
                self.pickedup = 1
                
                new_destination_edge = self.finder.find_begin_stop(
                    person.get_road(), self.edge_position, self.sumo
                ).partition("_")[0]
                print(new_stage)

            elif current_stage == "picked up":

                end_stop = self.finder.find_end_stop(
                    person.destination, self.edge_position, self.sumo
                ).partition("_")[0]
                new_destination_edge = end_stop
                vehicle.teleport(new_destination_edge)

                self.sumo.simulationStep()
                new_destination_edge = person.destination
                new_stage = "final"
                print(new_stage)

            elif current_stage == "final":
                new_stage = "done"
                done = 1
                fin =True
                print(new_stage)
        
        elif vedge == final_edge and self.pickedup == 1:
             new_stage = "done"
             print('Skipped Bus')
             done = 1
             fin = True
             new_destination_edge =final_edge

        return new_stage, new_destination_edge, self.pickedup, fin ,done

    def get_initial_stage(self):
        """
        Gets the initial stage of the simulation.

        Returns:
            str: Initial stage of the simulation.
        """

        return self.stage
    
    def calculate_reward(self, old_dist, edge_distance, destination_edge, vedge,  life, final_destination, final_old_dist, final_edge_distance):
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
            distcheck_final = 1  * self.pickedup
        elif final_old_dist < final_edge_distance:
            # reward = -0.025 * self.pickedup
            distcheck_final = 0
     

        if vedge == destination_edge:
            life += 0.1
            # reward = 0.8
            # make_choice_flag = True

        if vedge == final_destination and self.pickedup == 1:
            life += 0.1
            # reward = 0.8
            # make_choice_flag = False

        return reward,  distcheck, life, distcheck_final
