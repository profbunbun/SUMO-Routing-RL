
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

    .. todo:: Clean up config stuff
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
        self.life = config['agent_hyperparameters']['initial_life']
        self.penalty = config['agent_hyperparameters']['penalty']
        self.start_life = self.config['training_settings']['initial_life']
        self.num_of_vehicles = self.config['env']['num_of_vehicles']
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
        
        # self.old_edge = None  
        self.old_dist = None
        self.rewards = []
        self.epsilon_hist = []
        self.vehicle = None
        self.person = None
        self.p_index = 0
        self.distcheck = 0
        self.distcheck_final = 0
        self.edge_distance = None
        self.destination_edge = None
        self.stage = "reset"
        self.done = 0

        self.out_dict = out_dict
        self.index_dict = index_dict
        

    def reset(self, seed=42):
        """
        Reset the environment to its initial state.

        Args:
            seed (int): Random seed for reproducibility.

        Returns:
            observation: Initial observation after reset.
        """
        self.route = []
        self.distance_traveled = 0

  
        self.agent_step = 0
        self.accumulated_reward = 0
        self.life = self.start_life
        
        self.done = 0

        self.vehicle_manager = VehicleManager(self.num_of_vehicles, self.edge_locations, self.sumo, self.out_dict, self.index_dict, self.config)
        self.person_manager = PersonManager(self.num_people, self.edge_locations, self.sumo, self.index_dict, self.config)
        self.reward_manager = RewardManager(self.finder, self.edge_locations, self.sumo)


        self.stage = self.reward_manager.get_initial_stage()
        vehicles = self.vehicle_manager.create_vehicles()
        people = self.person_manager.create_people()
        self.person = people[0]
        vid_selected = self.ride_selector.select(vehicles, self.person)
        self.vehicle = vehicles[int(vid_selected)]
        self.sumo.simulationStep()
        self.old_dist = 0
        self.final_old_dist = 0
        
        self.destination_edge = self.person.get_road()
        self.final_destination = self.person.get_destination()
        dest_loc = self.edge_locations[self.destination_edge]
        self.final_loc = self.edge_locations[self.final_destination]
        self.picked_up = 0
        vedge = self.vehicle.get_road()

        self.route.append(vedge)

        observation = self.obs.get_state(self.sumo, self.agent_step, self.vehicle, dest_loc, self.life, self.distcheck, self.final_loc, self.distcheck_final, self.picked_up, self.done)
        return observation


    def step(self, action):
        """
        Perform an action in the environment.

        Args:
            action: Action to be performed.

        Returns:
            observation: Next state.
            reward: Reward obtained.
            done: Whether the episode is done.
            info: Additional info.


        .. todo:: Change the way the vehicle travles from the current teleportation method.
                  Figure out if we can reduce to one return
        """
        done = 0
        self.agent_step += 1
        choices = self.vehicle.get_out_dict()
        if self.life <= 0:
            self.stage = "done"
            done = 1

        vedge = self.vehicle.get_road()
        vedge_loc = self.edge_locations[vedge]
        dest_edge_loc = self.edge_locations[self.destination_edge]

        edge_distance = Utils.manhattan_distance(
            vedge_loc[0], vedge_loc[1], dest_edge_loc[0], dest_edge_loc[1]
        )


        final_edge_distance = Utils.manhattan_distance(
            vedge_loc[0], vedge_loc[1], self.final_loc[0], self.final_loc[1]
        )


        if self.direction_choices[action] in choices and done != 1 :
            
            vedge = self.perform_step(self.vehicle, self.direction_choices[action], self.destination_edge)
            
            self.life -= 0.01

            vedge = self.vehicle.get_road()
            choices = self.vehicle.get_out_dict()
            dest_loc = self.edge_locations[self.destination_edge]

            self.route.append(self.vehicle.get_road())

            self.distance_traveled = self.get_route_length(self.route)

            reward,  self.distcheck, self.life, self.distcheck_final = self.reward_manager.calculate_reward(
                self.old_dist, edge_distance, self.destination_edge, vedge,  self.life, self.final_destination, self.final_old_dist, final_edge_distance)

            self.stage, self.destination_edge, self.picked_up , done= self.reward_manager.update_stage(
                self.stage, self.destination_edge, vedge, self.person, self.vehicle, self.final_destination
            )
            if done == 1: 
                reward += 0.99 + self.life - (self.distance_traveled * 0.001)
                print("successfull dropoff")



            observation = self.obs.get_state(self.sumo, self.agent_step, self.vehicle, dest_loc, self.life, self.distcheck, self.final_loc, self.distcheck_final, self.picked_up, self.done)
            choices = self.vehicle.get_out_dict()

            self.old_dist = edge_distance
            self.final_old_dist = final_edge_distance
            info = vedge
            return observation, reward, done, info
        
        choices = self.vehicle.get_out_dict()
        done = 1
        reward = self.penalty
        dest_loc = self.edge_locations[self.destination_edge]
        observation = self.obs.get_state(self.sumo, self.agent_step, self.vehicle, dest_loc, self.life, self.distcheck, self.final_loc, self.distcheck_final, self.picked_up, self.done)
        self.accumulated_reward += reward
        info = vedge
        return observation, reward, done, info

    def render(self, mode='text'):
        """
        Render the environment.

        Args:
            mode (str): Mode of rendering. Options are 'human', 'text', 'no_gui'.

        .. todo:: Figure iout how to determone os for 
        """

        if mode == "human":
            self.sumo = self.sumo_con.connect_gui()

        elif mode == "text":
            self.sumo = self.sumo_con.connect_libsumo_no_gui()

        elif mode == "no_gui":
            self.sumo = self.sumo_con.connect_no_gui()
    
    # @timeit
    def close(self, episode, accu, current_epsilon):
        """
        Close the environment and log rewards.

        Args:
            episode (int): Current episode number.
            accu (float): Accumulated reward.
            current_epsilon (float): Current epsilon value.
        """

        

        self.sumo.close()
        acc_r = float(accu)
        self.rewards.append(acc_r)
        self.epsilon_hist.append(current_epsilon)
        avg_reward = np.mean(self.rewards[-100:])

        print_info = {
            "EP": episode,
            "Reward": f"{acc_r:.5}",
            "Avg Reward": f"{avg_reward:.3}",
            "Epsilon": f"{current_epsilon:.3}",
            "Steps": f"{self.agent_step}",
            "Distance": f"{self.distance_traveled}",
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

           distances.append(self.sumo.lane.getLength(''.join([edge,'_0'])))
        return round(sum(distances))
    




    def perform_step(self, vehicle, action, destination_edge):
        """
        Perform a step in the environment by moving the vehicle.

        Args:
            vehicle: Vehicle object.
            action: Action to be performed.
            destination_edge: Destination edge for the vehicle.

        Returns:
            vedge: Current edge of the vehicle after performing the step.
        """
   
        target = vehicle.set_destination(action, destination_edge)
        vehicle.teleport(target)
        
        self.sumo.simulationStep()
        vedge = vehicle.get_road()

        return  vedge



   