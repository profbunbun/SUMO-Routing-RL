
# import numpy as np
import random

# directions from https://github.com/guangli-dai/Selfless-Traffic-Routing-Testbed/blob/master/core/STR_SUMO.py
STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"
SLIGHT_LEFT = "L"
SLIGHT_RIGHT = "R"

class Vehicle:
    """
    Vehicle class representing a vehicle in the simulation.

    :param str vehicle_id: Unique identifier for the vehicle.
    :param dict out_dict: Dictionary of outgoing edges.
    :param dict index_dict: Dictionary of edge indices.
    :param dict edge_position: Dictionary of edge positions.
    :param sumo: SUMO simulation instance.
    :param int i: Arbitrary parameter for vehicle type.
    """

    def __init__(self, vehicle_id, out_dict, index_dict, edge_position, sumo, i) -> None:
        """
        Initialize a Vehicle instance with the given parameters.
        """

        # self.direction_choices = [RIGHT, STRAIGHT, LEFT]
        # self.direction_choices = [RIGHT, STRAIGHT, LEFT, TURN_AROUND]
        self.direction_choices = [SLIGHT_RIGHT, RIGHT, STRAIGHT, SLIGHT_LEFT, LEFT, TURN_AROUND]
        self.vehicle_id = vehicle_id
        self.out_dict = out_dict
        self.index_dict = index_dict
        self.sumo = sumo
        self.edge_position = edge_position
        self.sumo.vehicle.add(self.vehicle_id, "r_0", typeID="taxi")
        self.sumo.vehicle.setParameter(vehicle_id,
                                       "type", str(2)
                                    #    str(random.randint(1, i))
                                       )


        
        # self.random_relocate()
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]

    def get_lane(self):
        """
        Get the current lane of the vehicle.

        :return: Current lane ID.
        :rtype: str
        """
 
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]
        return self.cur_loc

    def get_lane_id(self):
        """
        Get the full lane ID of the vehicle.

        :return: Full lane ID.
        :rtype: str
        """
 
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane
        return self.cur_loc

    def location(self):
        """
        Get the current location of the vehicle.

        :return: Current (x, y) position of the vehicle.
        :rtype: list
        """

        vpos = self.sumo.vehicle.getPosition(self.vehicle_id)
        return [vpos[0], vpos[1]]

    def get_out_dict(self):
        """
        Get the dictionary of possible outgoing edges from the current lane.
        
        :return: Dictionary of outgoing edges.
        :rtype: dict
        """
 
        lane = self.get_lane()
        if lane not in self.out_dict.keys():
            options = None
        else:
             options = self.out_dict[lane]

        return options

    def set_destination(self, action, destination_edge):


        # self.sumo.vehicle.changeTarget(self.vehicle_id, destination_edge.partition("_")[0])
        # route = self.sumo.vehicle.getRoute(self.vehicle_id)
        # best_choice = route[1]
        # self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        

        self.cur_loc = self.current_lane.partition("_")[0]
        outlist = list(self.out_dict[self.cur_loc].keys())
        if action in outlist:
            target_lane = self.out_dict[self.cur_loc][action]
            # self.sumo.vehicle.changeTarget(self.vehicle_id, target_lane)
        return target_lane

    def pickup(self):

        reservation = self.sumo.person.getTaxiReservations(0)
        reservation_id = reservation[0]
        self.sumo.vehicle.dispatchTaxi(self.vehicle_id,"0")
        # print(reservation_id)
        
    def get_road(self):  

        return self.sumo.vehicle.getRoadID(self.vehicle_id)

    def random_relocate(self):
 
        new_lane=random.choice(list(self.index_dict.keys()))      
        self.sumo.vehicle.changeTarget(self.vehicle_id,edgeID=new_lane)
        self.sumo.vehicle.moveTo(self.vehicle_id,new_lane+"_0",5)

    def get_type(self):

        return self.sumo.vehicle.getParameter(self.vehicle_id,
                                              "type")
        
    def teleport(self, dest):
   
        self.sumo.vehicle.changeTarget(self.vehicle_id, edgeID=dest)
        self.sumo.vehicle.moveTo(self.vehicle_id, dest+"_0", 1)
    
