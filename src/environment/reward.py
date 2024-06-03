

class RewardManager:
    def __init__(self, finder, edge_position, sumo):

        self.finder = finder
        self.edge_position = edge_position
        self.sumo = sumo
        self.stage = "pickup"
        self.pickedup = 0

    def update_stage(self, current_stage, destination_edge, vedge, person, vehicle, final_edge):
        done = 0

        new_stage = current_stage
        
        new_destination_edge = destination_edge

        if vedge == destination_edge:

            if current_stage == "pickup":
                new_stage = "picked up"
                self.pickedup = 1
                
                new_destination_edge = self.finder.find_begin_stop(
                    person.get_road(), self.edge_position, self.sumo
                ).partition("_")[0]
                # print(new_stage)

            elif current_stage == "picked up":

                end_stop = self.finder.find_end_stop(
                    person.destination, self.edge_position, self.sumo
                ).partition("_")[0]
                new_destination_edge = end_stop
                vehicle.teleport(new_destination_edge)

                self.sumo.simulationStep()
                new_destination_edge = person.destination
                new_stage = "final"
                # print(new_stage)

            elif current_stage == "final":
                new_stage = "done"
                done = 1
                # print(new_stage)
        
        elif vedge == final_edge and self.pickedup == 1:
             new_stage = "done"
             print('Skipped Bus')
             done = 1
             new_destination_edge =final_edge

        return new_stage, new_destination_edge, self.pickedup, done

    def get_initial_stage(self):

        return self.stage
    
    def calculate_reward(self, old_dist, edge_distance, destination_edge, vedge,  life, final_destination, final_old_dist, final_edge_distance):
        
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
