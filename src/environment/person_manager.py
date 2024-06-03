from .person import Person

class PersonManager:


    def __init__(self, num_of_people, edge_position, sumo, index_dict, config):

        self.num_of_people = num_of_people
        self.edge_position = edge_position
        self.sumo = sumo
        self.index_dict = index_dict
        self.start_edge = config['person_settings']['start']
        self.destination = config['person_settings']['destination']

    def create_people(self):

        people = []
        for p_id in range(self.num_of_people):
            people.append(
                Person(
                    str(p_id),
                    self.sumo,
                    self.edge_position,
                    self.index_dict,
                    p_id + 1,
                    self.start_edge,
                    self.destination
                )
            )
        return people
