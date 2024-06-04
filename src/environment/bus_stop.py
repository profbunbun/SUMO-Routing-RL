

class StopFinder:
    """
    Class to find bus stops and calculate distances.
    """
 

    def __init__(self) -> None:
        """
        Initialize the StopFinder class.
        """
     
        self.con = None

    def manhat_dist(self, x1, y1, x2, y2):
        """
        Calculate the Manhattan distance between two points.

        Args:
            x1, y1: Coordinates of the first point.
            x2, y2: Coordinates of the second point.

        Returns:
            The Manhattan distance between the two points.
        """

        return abs(x1 - x2) + abs(y1 - y2)

    def find_bus_locs(self):
        """
        Find bus stop locations.

        Returns:
            A list of bus stop locations.
        """
   
        bus_stops = self.con.busstop.getIDList()
        bus_locs = []
        for stop in bus_stops:
            # print(stop)
            stop_loc = [stop, self.con.busstop.getLaneID(stop)]
            bus_locs.append(stop_loc)
        return bus_locs

    def get_stop_dists(self, loc, loc_dic):
        """
        Get distances from a location to all bus stops.

        Args:
            loc: The location to calculate distances from.
            loc_dic: A dictionary of location coordinates.

        Returns:
            A dictionary of bus stop distances.
        """
   
        stops = self.find_bus_locs()
        dist_dic = {}

        for stop in stops:
            stop_lane = stop[1].partition("_")[0]
            stop_loc = loc_dic[stop_lane]
            dest_loc = loc_dic[loc]
            dist_dic[stop[0]] = self.manhat_dist(
                dest_loc[0], dest_loc[1], stop_loc[0], stop_loc[1]
            )

        return dist_dic

    def find_end_stop(self, end_loc, loc_dic, con):
        """
        Find the nearest bus stop to the end location.

        Args:
            end_loc: The end location.
            loc_dic: A dictionary of location coordinates.
            con: SUMO connection.

        Returns:
            The nearest bus stop lane.
        """
       

        self.con = con

        dic = self.get_stop_dists(end_loc, loc_dic)
        stop = min(dic, key=dic.get)
        lane = con.busstop.getLaneID(stop)
        return lane

    def find_begin_stop(self, begin_loc, loc_dic, con):
        """
        Find the nearest bus stop to the beginning location.

        Args:
            begin_loc: The beginning location.
            loc_dic: A dictionary of location coordinates.
            con: SUMO connection.

        Returns:
            The nearest bus stop lane.
        """
        
        self.con = con

        dic = self.get_stop_dists(begin_loc, loc_dic)
        stop = min(dic, key=dic.get)

        lines = self.get_line(stop)
        
        lane = con.busstop.getLaneID(stop)
        return lane

    def get_line(self, stop_id):
        """
        Get the bus line for a given stop.

        Args:
            stop_id: The bus stop ID.

        Returns:
            The bus line associated with the stop.
        """
       
        # traci.busstop.getParameterWithKey()
        return self.con.busstop.getParameter(
            stop_id, "line"
        )  # getParameterWithKey(stop_id,"busStop")

    def get_line_route(self, con):
        """
        Get the route edges for a bus line.

        Args:
            con: SUMO connection.

        Returns:
            A list of edges for the bus route.
        """
       
        self.con = con
        # traci.route.getEdges()?
        return self.con.route.getEdges("bus_1")

