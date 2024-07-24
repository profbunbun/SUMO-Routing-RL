from .person import Person
import random

class PersonManager:
    """
    Manages the creation and management of persons in the simulation.

    .. todo:: consolodate manager and person class
    """

    def __init__(self, num_people, edge_dict, sumo, index_dic, config):
        """
        Initializes the PersonManager with the given parameters.

        Args:
            num_people (int): Number of people to create.
            edge_dict (dict): Dictionary of edge locations.
            sumo: SUMO simulation instance.
            index_dic (dict): Dictionary of edge indices.
            config (dict): Configuration dictionary.
        """
        self.num_people = num_people
        self.edge_dict = edge_dict
        self.sumo = sumo
        self.index_dic = index_dic
        self.config = config
        self.people = []

        self.disability_catagories = self.config['env']['types_of_passengers']

    def create_people(self):
        """
        Creates persons for the simulation.

        Returns:
            list: List of created Person instances.
        """
        people = []
        for i in range(self.num_people):
            start_list = ['-521985670#5',
                          '6007194#9',
                        #   '-49664167#4',
                          '-138388359#16',
                          '-829470071#1',
                          '829470070#1']

            # start = '-521985670#5'
            end = '192469470#0'
            start = start_list[i]
            # start = random.choice(list(self.edge_dict.keys()))
            # end = random.choice(list(self.edge_dict.keys()))
            while start == end:  # Ensure start and end are different
                end = random.choice(list(self.edge_dict.keys()))
            person_id = f"person_{i}"
            person = Person(person_id, self.disability_catagories, start, end, self.sumo)
            people.append(person)
        self.people = people
        return people

    def get_people(self):
        """
        Gets the list of persons created by the manager.

        Returns:
            list: List of created Person instances.
        """
        return self.people
