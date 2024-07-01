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
        # self.destination_edges = []
        self.final_destinations = []

        self.stage = "reset"
        self.dones = [0] * self.num_of_vehicles
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

        self.accumulated_reward = [[]] * self.num_of_vehicles

        self.dones = [0] * self.num_of_vehicles
        self.dispatched = [False] * self.num_of_vehicles  
        self.rewards = [0] * self.num_of_vehicles
        self.observations = [0] * self.num_of_vehicles  

        self.vehicle_manager = VehicleManager(self.num_of_vehicles, self.edge_locations, self.sumo, self.out_dict, self.index_dict, self.config)
        self.person_manager = PersonManager(self.num_people, self.edge_locations, self.sumo, self.index_dict, self.config)
        self.reward_manager = RewardManager(self.finder, self.edge_locations, self.sumo)

        # self.stage = self.reward_manager.get_initial_stage()
        self.vehicles = self.vehicle_manager.create_vehicles()

        self.sumo.simulationStep()

        self.people = self.person_manager.create_people()
        self.sumo.simulationStep()


        
        self.assign_rides()




    
        self.picked_up = [0] * self.num_of_vehicles
        self.pickup_edges = [person.get_road() for person in self.people]
        self.final_destinations = [person.get_destination() for person in self.people]

        self.sumo.simulationStep()
        self.sumo.simulationStep()

        
        self.observations = [self.get_observation(v) for v in self.vehicles if v.dispatched==True]
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

            while j < len(reservations) and (empty_fleet == False):
                p_type = self.sumo.person.getParameter(reservations[j].persons[0], "type")
                if p_type == v_type:
                    slot_index= int(fleet[0])
                    self.dispatched[slot_index] = True  
                    self.vedges[slot_index] = self.sumo.vehicle.getRoadID(fleet[0])  
                    self.old_vedges[slot_index] = self.vedges[slot_index] 
                    self.sumo.vehicle.dispatchTaxi(fleet[i], reservations[j].id)
                    self.vehicles[int(fleet[i])].passenger_id = reservations[j].persons
                    self.vehicles[int(fleet[i])].dispatched = True
                    self.vehicles[int(fleet[i])].current_reservation_id = reservations[j].id
                    self.vehicles[int(fleet[i])].passenger_pick_up_edge = reservations[j].fromEdge
                    self.vehicles[int(fleet[i])].bus_stop_drop_edge = self.finder.find_begin_stop(
                        reservations[j].fromEdge,
                        self.edge_locations,
                        self.sumo).partition("_")[0]
                    self.vehicles[int(fleet[i])].bus_stop_pick_up_edge = self.finder.find_end_stop(
                        reservations[j].toEdge,
                        self.edge_locations,
                        self.sumo).partition("_")[0]
                    self.vehicles[int(fleet[i])].final_edge = reservations[j].toEdge
                    self.vehicles[int(fleet[i])].final_destination_edge_location = self.edge_locations[reservations[j].fromEdge]
                    self.vehicles[int(fleet[i])].update_stage(0)
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
                

    
             

    def get_observation(self, vehicle):
        dest_loc = self.edge_locations[vehicle.current_destination ]
        return self.obs.get_state(self.sumo,
                                  vehicle.agent_step,
                                  vehicle,
                                  dest_loc, 
                                  vehicle.life,
                                  vehicle.distcheck, 
                                  vehicle.final_destination_edge_location,
                                  vehicle.distcheck_final,
                                  vehicle.picked_up,
                                  vehicle.done)

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


        for i in range(len(actions)):
            vehicle = self.vehicles[i]

            if not vehicle.dispatched:
                vehicle.done = True
                self.dones[i] = True
                continue  


            vehicle.agent_step += 1
            vehicle.life -= .01

            vedge = vehicle.get_lane()
            while vedge not in self.index_dict:
                self.sumo.simulationStep()
                vedge = vehicle.get_lane()
            vedge_loc = self.edge_locations[vedge]
            choices = vehicle.get_out_dict()
            choices_keys = choices.keys()
            choice = self.direction_choices[actions[i]]


            if (vehicle.life <= 0) or (choice not in choices_keys):
                self.dones[i] = True
                self.rewards[i] += self.penalty
                self.observations[i]=self.get_observation(vehicle)
                self.infos[i]=vehicle.get_road()
                self.dispatched[i] = False
                vehicle.dispatched = False
                vehicle.park()
                # self.sumo.simulationStep()
            else:
                target = vehicle.set_destination(self.direction_choices[actions[i]])


                match vehicle.current_stage:

                    case 0:
                        pickup_loc = vehicle.passenger_pick_up_edge

                        if target == pickup_loc:
                            vehicle.pickup(self.people[i].get_reservation())
                        else:
                            vehicle.retarget(target)
                    case 1:
                        vehicle.retarget(target)
                    case 2:
                        vehicle.retarget(target)
                    case 3:
                        vehicle.retarget(target)
                    case 4:
                        vehicle.retarget(target)
                    case default:
                        vehicle.retarget(target)


                vehicle.destination_edge_location = self.edge_locations[vehicle.current_destination]

                vehicle.destination_distance = Utils.manhattan_distance(
                    vedge_loc[0], vedge_loc[1],
                    vehicle.destination_edge_location[0], vehicle.destination_edge_location[1]
                )

                vehicle.final_destination_distance = Utils.manhattan_distance(
                    vedge_loc[0], vedge_loc[1],
                    vehicle.final_destination_edge_location[0], vehicle.final_destination_edge_location[1]
                )

                vehicle.destination_old_distance = vehicle.destination_distance
                vehicle.final_destination_old_distance = vehicle.final_destination_distance
        self.sumo.simulationStep()

        for vehicle in self.vehicles:

            vedge = vehicle.get_lane()
            if len(vedge) == 0:
                continue
            else:
                # stop_state = vehicle.get_stop_state()
                while vedge not in self.index_dict:
                    self.sumo.simulationStep()
                    vedge = vehicle.get_lane()

        

        # Post-step logic
        for i in range(len(actions)):
            vehicle = self.vehicles[i]
            if not vehicle.dispatched or vehicle.life <= 0 or actions.count(None)==len(actions):
            # if not vehicle.dispatched or vehicle.life <= 0 or actions.count(None)==len(actions):
                continue  # Skip vehicles that are not dispatched or are done

            vedge = vehicle.get_road()
            choices = vehicle.get_out_dict()
            vehicle.route.append(vehicle.get_road())

            vehicle.destination_distance = Utils.manhattan_distance(
                    vedge_loc[0], vedge_loc[1],
                    vehicle.destination_edge_location[0], vehicle.destination_edge_location[1]
                )

            vehicle.final_destination_distance = Utils.manhattan_distance(
                vedge_loc[0], vedge_loc[1],
                vehicle.final_destination_edge_location[0], vehicle.final_destination_edge_location[1]
            )

            reward, vehicle.distcheck, vehicle.distcheck_final = self.reward_manager.calculate_reward(
                vehicle.destination_old_distance,
                vehicle.destination_distance,
                vehicle.final_destination_old_distance,
                vehicle.final_destination_distance,
                vehicle)

            vehicle.current_stage, vehicle.current_destination, vehicle.picked_up = self.reward_manager.update_stage(
                vedge,
                vehicle,
                vehicle.final_edge,
            )

            if vehicle.fin:
                reward += 0.99 + vehicle.life
                print("Successful dropoff")
            self.observations[i] = self.get_observation(vehicle)
            self.rewards[i]+=reward
            self.infos[i]=vedge
            self.dones[i] = vehicle.done
            # self.accumulated_reward[i] += reward

        return self.observations, self.rewards, self.dones, self.infos

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
            "Steps": f"{self.vehicles[agent].agent_step}",
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
