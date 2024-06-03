
class Person:


    def __init__(self, person_id, sumo, edge_position, index_dict, types, start_edge,destination) -> None:

        self.person_id = person_id
        self.sumo = sumo
        self.index_dict = index_dict
        self.edge_position = edge_position
        self.destination = destination

        self.sumo.person.add(person_id, start_edge, 20)

        self.sumo.person.appendDrivingStage(person_id,
                                            self.destination,
                                            lines="taxi")

        self.sumo.person.setParameter(person_id,
                                      "type", str(2))
        #   str(random.randint(1, types))

    def location(self):
 
        ppos = self.sumo.person.getPosition(self.person_id)

        return ppos

    def get_destination(self):
 
        return self.destination

    def remove_person(self):

        self.sumo.person.remove(self.person_id)
     
    def get_road(self):

        return self.sumo.person.getRoadID(self.person_id)

    def get_type(self):

        return self.sumo.person.getParameter(self.person_id,
                                             "type")
