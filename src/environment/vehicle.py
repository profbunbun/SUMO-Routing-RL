
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

    Attributes:
        vehicle_id (str): Unique identifier for the vehicle.
        direction_choices (list): List of possible direction choices.
        out_dict (dict): Dictionary of outgoing edges.
        index_dict (dict): Dictionary of edge indices.
        sumo: SUMO simulation instance.
        edge_position (dict): Dictionary of edge positions.
    """

    def __init__(self, vehicle_id, vtype, out_dict, index_dict, edge_position, sumo) -> None:
        """
        Initialize a Vehicle instance with the given parameters.

        Args:
            vehicle_id (str): Unique identifier for the vehicle.
            types (int): Number of vehicle types.
            out_dict (dict): Dictionary of outgoing edges.
            index_dict (dict): Dictionary of edge indices.
            edge_position (dict): Dictionary of edge positions.
            sumo: SUMO simulation instance.
        """

        self.direction_choices = [SLIGHT_RIGHT, RIGHT, STRAIGHT, SLIGHT_LEFT, LEFT, TURN_AROUND]
        self.vehicle_id = vehicle_id
        self.out_dict = out_dict
        self.index_dict = index_dict
        self.sumo = sumo
        self.edge_position = edge_position
        self.sumo.vehicle.add(self.vehicle_id, "r_0", typeID="taxi")
        self.sumo.simulationStep()
        # vtype = str(random.randint(1, types))
        self.sumo.vehicle.setParameter(vehicle_id,"type",str(vtype))
        self.dispatched = False

        # self.random_relocate()
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]

    def get_lane(self):
        """
        Get the current lane of the vehicle.

        Returns:
            str: Current lane ID.
        """
 
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]
        return self.cur_loc

    def get_lane_id(self):
        """
        Get the full lane ID of the vehicle.

        Returns:
            str: Full lane ID.
        """
 
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane
        return self.cur_loc

    def location(self):
        """
        Get the current location of the vehicle.

        Returns:
            list: Current (x, y) position of the vehicle.
        """

        vpos = self.sumo.vehicle.getPosition(self.vehicle_id)
        return [vpos[0], vpos[1]]

    def get_out_dict(self):
        """
        Get the dictionary of possible outgoing edges from the current lane.

        Returns:
            dict: Dictionary of outgoing edges.
        """
 
        lane = self.get_lane()
        if lane not in self.out_dict.keys():
            options = None
        else:
             options = self.out_dict[lane]

        return options

    def set_destination(self, action):

        """
        Set the destination edge for the vehicle.

        Args:
            action (str): Chosen action.
            destination_edge (str): Destination edge ID.

        Returns:
            str: Target lane ID.
        """



        

        self.cur_loc = self.current_lane.partition("_")[0]
        outlist = list(self.out_dict[self.cur_loc].keys())
        if action in outlist:
            target_lane = self.out_dict[self.cur_loc][action]
            # self.sumo.vehicle.changeTarget(self.vehicle_id, target_lane)
        return target_lane

    def pickup(self, reservation_id):
        """
        Dispatch the vehicle to pick up a passenger.
        """

        reservation = self.sumo.person.getTaxiReservations(0)[0].id
        # reservation = reservation_id(0)[0].id
        self.sumo.vehicle.dispatchTaxi(self.vehicle_id,reservation)
        self.dispatched = True
        # print(reservation_id)
        
    def get_road(self): 
        """
        Get the current road ID of the vehicle.

        Returns:
            str: Current road ID.
        """ 

        return self.sumo.vehicle.getRoadID(self.vehicle_id)

    def random_relocate(self):
        """
        Relocate the vehicle to a random lane.
        """
 
        new_lane=random.choice(list(self.index_dict.keys()))      
        self.sumo.vehicle.changeTarget(self.vehicle_id,edgeID=new_lane)
        self.sumo.vehicle.moveTo(self.vehicle_id,new_lane+"_0",5)

    def get_type(self):
        """
        Get the type of the vehicle.

        Returns:
            str: Vehicle type.
        """

        return self.sumo.vehicle.getParameter(self.vehicle_id,
                                              "type")
        
    def teleport(self, dest):
        """
        Teleport the vehicle to the destination edge.

        Args:
            dest (str): Destination edge ID.
        """
   
        self.sumo.vehicle.changeTarget(self.vehicle_id, edgeID=dest)
        self.sumo.vehicle.moveTo(self.vehicle_id, dest+"_0", 1)

    def retarget(self, dest):
        """
        Retarget the vehicle to the destination edge.

        Args:
            dest (str): Destination edge ID.
        """
        # route = self.get_route()
        # print(route)
   
        
        self.sumo.vehicle.setRoute(self.vehicle_id,[self.get_road(),dest])
        self.sumo.vehicle.changeTarget(self.vehicle_id, edgeID=dest)
        # route = self.get_route()
        # # print(route)

    

    def get_route(self):
        """
        Get vehicle route

        Returns:
            str: Vehicle route.
        
        """

        return self.sumo.vehicle.getRouteIndex(self.vehicle_id),self.sumo.vehicle.getRoute(self.vehicle_id)
    
    def park(self):


        vehicle_edge = self.get_lane()

        parking_edge = "49664167#5"


        if vehicle_edge==parking_edge:
            self.teleport("-49664167#5")

        new_route = self.sumo.simulation.findRoute(vehicle_edge, parking_edge).edges
        self.sumo.vehicle.setRoute(self.vehicle_id, new_route)
        
        self.sumo.vehicle.setParkingAreaStop(self.vehicle_id, "pa_1")
