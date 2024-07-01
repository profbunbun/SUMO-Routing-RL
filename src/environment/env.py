from .reward import RewardManager
import numpy as np
from utils.connect import SUMOConnection
from .ride_select import RideSelect
from .bus_stop import StopFinder
from .observation import Observation
from .vehicle_manager import VehicleManager 
from .person_manager import PersonManager 
from utils.utils import Utils

class Env():
    """
    Environment class for the SUMO-RL project.
    """

    def __init__(self, config, edge_locations, out_dict, index_dict):
        """
        Initialize the environment with the given configuration and parameters.

        Args:
            config (dict): Configuration dictionary.
            edge_locations (dict): Edge locations dictionary.
            out_dict (dict): Output dictionary.
            index_dict (dict): Index dictionary.
        """
        self.config = config  
        self.path = config['training_settings']['experiment_path']
        self.sumo_config_path = self.path + config['training_settings']['sumoconfig']
        self.num_of_vehicles = config['env']['num_of_vehicles']
        self.types_of_passengers = config['env']['types_of_passengers']
        self.penalty = config['agent_hyperparameters']['penalty']
        self.start_life = self.config['training_settings']['initial_life']
        self.num_people = self.config['env']['num_of_people']

        self.direction_choices = ['R', 'r', 's', 'L', 'l', 't']

        self.obs = Observation()
        self.finder = StopFinder()
        self.sumo_con = SUMOConnection(self.sumo_config_path)
        self.ride_selector = RideSelect()  
        self.edge_locations = edge_locations
        self.sumo = None   
        self.agent_step = 0
        self.accumulated_reward = 0
        
        self.old_dist = None
        self.rewards = []
        self.epsilon_hist = []
        self.vehicles = []
        self.people = []
        self.life = [self.start_life] * self.num_of_vehicles
        self.distcheck = [0] * self.num_of_vehicles
        self.distcheck_final = [0] * self.num_of_vehicles
        self.edge_distance = [0] * self.num_of_vehicles
        self.route = [[] for _ in range(self.num_of_vehicles)]
        self.destination_edges = []
        self.final_destinations = []
        self.final_locs = []
        self.stage = "reset"
        self.done = [0] * self.num_of_vehicles
        self.dispatched = [False] * self.num_of_vehicles  
        self.vedges = [None] * self.num_of_vehicles  
        self.old_vedges = [None] * self.num_of_vehicles

        self.infos = [None]* self.num_of_vehicles 

        self.reservations = []
        self.observations = []

        self.out_dict = out_dict
        self.index_dict = index_dict

    def reset(self, seed=42):
        """
        Reset the environment to its initial state.

        Args:
            seed (int): Random seed for reproducibility.

        Returns:
            observations (list): Initial observations for each agent after reset.
        """
        self.route = [[] for _ in range(self.num_of_vehicles)]
        self.distance_traveled = 0



        self.agent_step = [0] * self.num_of_vehicles
        self.accumulated_reward = [[]] * self.num_of_vehicles
        self.life = [self.start_life] * self.num_of_vehicles
        self.done = [0] * self.num_of_vehicles
        self.dispatched = [False] * self.num_of_vehicles  
        self.rewards = [0] * self.num_of_vehicles
        self.observations = [0] * self.num_of_vehicles  

        self.vehicle_manager = VehicleManager(self.num_of_vehicles, self.edge_locations, self.sumo, self.out_dict, self.index_dict, self.config)
        self.person_manager = PersonManager(self.num_people, self.edge_locations, self.sumo, self.index_dict, self.config)
        self.reward_manager = RewardManager(self.finder, self.edge_locations, self.sumo)

        self.stage = self.reward_manager.get_initial_stage()
        self.vehicles = self.vehicle_manager.create_vehicles()

        self.sumo.simulationStep()

        self.people = self.person_manager.create_people()
        self.sumo.simulationStep()

        self.done = [False] * self.num_of_vehicles
        self.fin= [False] * self.num_of_vehicles
        
        self.assign_rides()


        self.old_dist = [0] * self.num_of_vehicles
        self.final_old_dist = [0] * self.num_of_vehicles

    
        self.picked_up = [0] * self.num_of_vehicles
        self.destination_edges = [person.get_road() for person in self.people]
        self.final_destinations = [person.get_destination() for person in self.people]
        self.final_locs = [self.edge_locations[dest] for dest in self.final_destinations]

        self.sumo.simulationStep()
        self.sumo.simulationStep()

        
        self.observations = [self.get_observation(i) for i in range(self.num_of_vehicles)]
        return self.observations

    def assign_rides(self):
        reservations = self.sumo.person.getTaxiReservations()
        fleet = self.sumo.vehicle.getTaxiFleet(0)
        empty_fleet=False
        i = 0
        j = 0
        while i < len(fleet):
            v_type = self.sumo.vehicle.getParameter(fleet[i], "type")
            dispatched = False
            # self.done[i] = True
            while j < len(reservations) and (empty_fleet == False):
                p_type = self.sumo.person.getParameter(reservations[j].persons[0], "type")
                if p_type == v_type:
                    slot_index= int(fleet[0])
                    self.dispatched[slot_index] = True  
                    self.vedges[slot_index] = self.sumo.vehicle.getRoadID(fleet[0])  
                    self.old_vedges[slot_index] = self.vedges[slot_index] 
                    self.sumo.vehicle.dispatchTaxi(fleet[i], reservations[j].id)
                    self.vehicles[int(fleet[i])].dispatched = True
                    self.vehicles[int(fleet[i])].current_reservation_id = reservations[j].id
                    self.vehicles[int(fleet[i])].pickup_location = reservations[j].fromEdge
                    self.sumo.simulationStep()
                    fleet = self.sumo.vehicle.getTaxiFleet(0)
                    if not fleet:
                        empty_fleet=True 
                    dispatched = True
                    # print("dispatched")
                    j+=1
                    break
                j+=1
            if not dispatched:
                i += 1
                

    
             

    def get_observation(self, index):
        dest_loc = self.edge_locations[self.destination_edges[index]]
        return self.obs.get_state(self.sumo,
                                  self.agent_step[index],
                                  self.vehicles[index],
                                  dest_loc, 
                                  self.life[index],
                                  self.distcheck[index], 
                                  self.final_locs[index],
                                  self.distcheck_final[index],
                                  self.picked_up[index],
                                  self.done[index])

    def step(self, actions):
        """
        Perform actions in the environment for each agent.

        Args:
            actions (list): Actions to be performed by each agent.

        Returns:
            observations (list): Next states for each agent.
            rewards (list): Rewards obtained by each agent.
            done (list): Whether each agent's episode is done.
            infos (list): Additional info for each agent.
        """


        for i in range(self.num_of_vehicles):
        # for i in range(self.num_of_vehicles):
            dist = self.vehicles[i].dispatched
            if not self.dispatched[i]:
                self.done[i] = True
                continue  

            self.agent_step[i] += 1
            self.life[i]-= .01
            vehicle = self.vehicles[i]
            vedge = vehicle.get_lane()
            while vedge not in self.index_dict:
                self.sumo.simulationStep()
                vedge = vehicle.get_lane()
            vedge_loc = self.edge_locations[vedge]
            choices = self.vehicles[i].get_out_dict()
            choices_keys = choices.keys()
            choice = self.direction_choices[actions[i]]


            if (self.life[i] <= 0) or (choice not in choices_keys):
                self.done[i] = True
                self.rewards[i] += self.penalty
                self.observations[i]=self.get_observation(i)
                self.infos[i]=self.vehicles[i].get_road()
                self.dispatched[i] = False
                self.vehicles[i].dispatched = False
                self.vehicles[i].park()
                # self.sumo.simulationStep()
            else:
                target = self.vehicles[i].set_destination(self.direction_choices[actions[i]])
                # pickup_loc = self.people[i].get_road()
                pickup_loc = self.vehicles[i].pickup_location 

                if target == pickup_loc:
                    self.vehicles[i].pickup(self.people[i].get_reservation())
                else:
                    self.vehicles[i].retarget(target)

                dest_edge_loc = self.edge_locations[self.destination_edges[i]]

                edge_distance = Utils.manhattan_distance(
                    vedge_loc[0], vedge_loc[1], dest_edge_loc[0], dest_edge_loc[1]
                )

                final_edge_distance = Utils.manhattan_distance(
                    vedge_loc[0], vedge_loc[1], self.final_locs[i][0], self.final_locs[i][1]
                )

                self.old_dist[i] = edge_distance
                self.final_old_dist[i] = final_edge_distance
        self.sumo.simulationStep()

        for vehicle in self.vehicles:

            vedge = vehicle.get_lane()
            if len(vedge) == 0:
                continue
            else:
                stop_state = vehicle.get_stop_state()
                while vedge not in self.index_dict:
                    self.sumo.simulationStep()
                    vedge = vehicle.get_lane()

        

        # Post-step logic
        for i in range(self.num_of_vehicles):
            if not self.dispatched[i] or self.life[i] <= 0 or actions.count(None)==len(actions):
            # if not self.vehicles[i].dispatched or self.life[i] <= 0 or actions.count(None)==len(actions):
                continue  # Skip vehicles that are not dispatched or are done

            vedge = self.vehicles[i].get_road()
            choices = self.vehicles[i].get_out_dict()
            self.route[i].append(self.vehicles[i].get_road())

            edge_distance = Utils.manhattan_distance(
                vedge_loc[0], vedge_loc[1], dest_edge_loc[0], dest_edge_loc[1]
            )

            final_edge_distance = Utils.manhattan_distance(
                vedge_loc[0], vedge_loc[1], self.final_locs[i][0], self.final_locs[i][1]
            )

            reward, self.distcheck[i], self.life[i], self.distcheck_final[i] = self.reward_manager.calculate_reward(
                self.old_dist[i], edge_distance, self.destination_edges[i], vedge, self.life[i], self.final_destinations[i], self.final_old_dist[i], final_edge_distance
            )

            self.stage, self.destination_edges[i], self.picked_up[i], self.fin[i] ,self.done[i]= self.reward_manager.update_stage(
                self.stage, self.destination_edges[i], vedge, self.people[i], self.vehicles[i], self.final_destinations[i], self.done[i]
            )

            if self.fin[i]:
                reward += 0.99 + self.life[i]
                print("Successful dropoff")
            self.observations[i] = self.get_observation(i)
            self.rewards[i]+=reward
            self.infos[i]=vedge
            # self.accumulated_reward[i] += reward

        return self.observations, self.rewards, self.done, self.infos

    def render(self, mode='text'):
        """
        Render the environment.

        Args:
            mode (str): Mode of rendering. Options are 'human', 'text', 'no_gui'.

        .. todo:: Figure out how to determine os for rendering
        """
        if mode == "human":
            self.sumo = self.sumo_con.connect_gui()
        elif mode == "text":
            self.sumo = self.sumo_con.connect_libsumo_no_gui()
        elif mode == "no_gui":
            self.sumo = self.sumo_con.connect_no_gui()

    def pre_close(self, episode, agent, accu, current_epsilon):
        """
        Close the environment and log rewards.

        Args:
            episode (int): Current episode number.
            accu (float): Accumulated reward.
            current_epsilon (float): Current epsilon value.
        """
        # self.sumo.close()
        acc_r = float(accu)
        self.accumulated_reward[agent].append(acc_r)
        self.epsilon_hist.append(current_epsilon)
        avg_reward = np.mean(self.accumulated_reward[agent][-100:])

        print_info = {
            "EP": episode,
            "Agent": agent,
            "Reward": f"{acc_r:.5}",
            "Avg Reward": f"{avg_reward:.3}",
            "Epsilon": f"{current_epsilon:.3}",
            "Steps": f"{self.agent_step[agent]}",
            "Distance": f"{self.vehicles[agent].get_dist():.2f}",
        }
        print(", ".join(f"{k}: {v}" for k, v in print_info.items()))
        return
    
    def quiet_close(self):
        """
        Quietly close the environment without logging.
        """
        self.sumo.close()
        return
    
    def get_route_length(self, route):
        """
        Get the total length of the route.

        Args:
            route (list): List of edges in the route.

        Returns:
            int: Total length of the route.
        """
        distances = []
        for edge in route:
            distances.append(self.sumo.lane.getLength(''.join([edge, '_0'])))
        return round(sum(distances))
