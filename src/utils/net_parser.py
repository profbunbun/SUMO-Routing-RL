
import xml.etree.ElementTree as ET
import sumolib
from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


class NetParser:
    '''
    NetParser class for parsing network files and retrieving information from SUMO simulations.

    :param str sumocfg: Path to the SUMO configuration file.
    '''

    def __init__(self, sumocfg) -> None:
        '''
        Initialize the NetParser object with a specific SUMO configuration file.

        :param str sumocfg: Path to the SUMO configuration file.
        '''
        self.sumocfg = sumocfg
    # @timeit
    def parse_net_files(self):
        '''
        Get the network file from the SUMO configuration file.

        :return: Path to the network file extracted from the SUMO configuration.
        :rtype: str
        '''

        tree = ET.parse(self.sumocfg)
        root = tree.getroot()
        for infile in root.findall("input"):
            for network in infile.findall("net-file"):
                network_file = str(network.get("value"))
            return network_file
    # @timeit
    def _clean_path(self):
        '''
        Clean the file path for the network file.

        :return: The network object after reading the network file.
        :rtype: sumolib.net.Net
        '''

        net_file = self.parse_net_files()
        path_ = self.sumocfg.rsplit("/")
        path_.pop()
        path_b = "/".join(path_)
        return sumolib.net.readNet(path_b + "/" + net_file)
    # @timeit
    def get_edges_info(self):
        '''
        Get a list of edges that allow passenger vehicles.

        :return: List of edges allowing passenger vehicles.
        :rtype: list
        '''

        net = self._clean_path()
        edge_list = []
        all_edges = net.getEdges()
        for current_edge in all_edges:
            if current_edge.allows("passenger"):
                edge_list.append(current_edge)
        return edge_list
    # @timeit
    def get_edge_pos_dic(self):
        '''
        Get a dictionary of edge IDs and their XY coordinates at the center.

        :return: Dictionary of edge IDs and their center XY coordinates.
        :rtype: dict
        '''

        net = self._clean_path()
        edge_position_dict = {}
        all_edges = net.getEdges()
        dims = []
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()

            if current_edge_id in edge_position_dict:
                print(current_edge_id + " already exists!")
            else:
                dims = list(current_edge.getShape())
                edge_start = dims[0]
                edge_end = dims[1]
                x = (edge_start[0] + edge_end[0]) / 2

                y = (edge_start[1] + edge_end[1]) / 2

                edge_position_dict[current_edge_id] = x, y
        return edge_position_dict
    # @timeit
    def get_out_dic(self):
        '''
        Get a dictionary of edges and their connecting edges.

        :return: Dictionary of edges and their respective connecting edges.
        :rtype: dict
        '''

        net = self._clean_path()
        out_dict = {}
        all_edges = net.getEdges()
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()
            if current_edge_id in out_dict:
                print(current_edge_id + " already exists!")
            else:
                out_dict[current_edge_id] = {}
            out_edges = current_edge.getOutgoing()
            for current_out_edge in out_edges:
                if not current_out_edge.allows("passenger"):
                    # print("Found some roads prohibited")
                    continue
                conns = current_edge.getConnections(current_out_edge)
                for conn in conns:
                    dir_now = conn.getDirection()
                    out_dict[
                        current_edge_id][dir_now] = current_out_edge.getID()
        return out_dict
    # @timeit
    def get_edge_index(self):
        '''
        Get an indexed dictionary of edge IDs.

        :return: Indexed dictionary of edge IDs.
        :rtype: dict
        '''


        net = self._clean_path()
        index_dict = {}
        counter = 0
        all_edges = net.getEdges()
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()

            if current_edge_id in index_dict:
                print(current_edge_id + " already exists!")
            else:
                index_dict[current_edge_id] = counter
                counter += 1
        return index_dict
    # @timeit
    def get_length_dic(self):
        '''
        Get a dictionary of edge IDs and their lengths.

        :return: Dictionary of edge IDs and their lengths.
        :rtype: dict
        '''

        net = self._clean_path()

        length_dict = {}
        all_edges = net.getEdges()
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()
            if current_edge_id in length_dict:
                print(current_edge_id + " already exists!")
            else:
                length_dict[current_edge_id] = current_edge.getLength()
        return length_dict
    # @timeit
    def get_route_edges(self):
        '''
        Get a list of edge IDs from a specific route file.

        :return: List of edge IDs from the specified route.
        :rtype: list
        '''
        edge_ids = []
        for route in sumolib.xml.parse_fast("Experiments/balt1/Nets/osm_pt.rou.xml", 'route', ['id','edges']):
        # for route in sumolib.xml.parse_fast("Experiments/3x3/Nets/3x3.rou.xml", 'route', ['id','edges']):
            if 'bus' in route.id:
                edge_ids = route.edges.split()
        # print (edge_ids)
        return edge_ids
    # @timeit
    def get_max_manhattan(self):
        '''
        Calculate the maximum Manhattan distance between any two edges in the network.

        :return: Maximum Manhattan distance.
        :rtype: float
        '''
        a=self.get_edge_pos_dic()
        a=list(a.values())
        n=len(a)
        
        V = [0 for i in range(n)]
        V1 = [0 for i in range(n)]
 
        for i in range(n):
            V[i] = a[i][0] + a[i][1]
            V1[i] = [i][0] - a[i][1]
 
        # Sorting both the vectors
        V.sort()
        V1.sort()
    
        maximum = max(V[-1] - V[0],
                    V1[-1] - V1[0])
 
        print(maximum)
        return maximum
    # @timeit
    def net_minmax(self):
        '''get net minmax xy coords for scaling input'''
        net = self._clean_path()
        return sumolib.net.Net.getBBoxXY(net)
    # @timeit
    def get_junctions(self):
        """Retrieve junctions and their internal edges from the network."""
        net = self._clean_path()
        junctions = set()
        for junction in net.getNodes():
            # Add the junction ID
            if junction.getType() != "internal":  # Exclude external junctions
                junctions.add(junction.getID())

        # Add internal edges from the network
        for edge in net.getEdges():
            if edge.getID().startswith(":"):  # Check if it's an internal edge
                junctions.add(edge.getID())
        return junctions
