import random
class Person:
    """
    Person class representing a person in the simulation.

    Attributes:
        person_id (str): Unique identifier for the person.
        start (str): Starting edge ID.
        end (str): Destination edge ID.
        sumo: SUMO simulation instance.
    """


    def __init__(self, person_id, types, start_edge,destination, sumo) -> None:
        """
        Initializes a Person instance with the given parameters.

        Args:
            person_id (str): Unique identifier for the person.
            start (str): Starting edge ID.
            end (str): Destination edge ID.
            sumo: SUMO simulation instance.
        """

        self.person_id = person_id
        self.sumo = sumo

        self.destination = destination

        self.sumo.person.add(person_id, start_edge, 20)

        self.sumo.person.appendDrivingStage(person_id,
                                            self.destination,
                                            lines="taxi")

        self.sumo.person.setParameter(person_id,
                                      "type",
                                       str(random.randint(1, types)))
                                        # str(2))
        #   str(random.randint(1, types))

    def location(self):
        """
        Gets the current location of the person.

        Returns:
            str: Current position.
        """
 
        ppos = self.sumo.person.getPosition(self.person_id)

        return ppos

    def get_destination(self):
        """
        Gets the destination edge ID of the person.

        Returns:
            str: Destination edge ID.
        """
 
        return self.destination

    def remove_person(self):
        """
        Removes person from simulation
        """

        self.sumo.person.remove(self.person_id)
     
    def get_road(self):
        """
        Gets the current road ID of the person.

        Returns:
            str: Current road ID.
        """

        return self.sumo.person.getRoadID(self.person_id)

    def get_type(self):
        """
        Gets the dissability catagory of the person.

        Returns:
            str: Dissability catagory
        """

        return self.sumo.person.getParameter(self.person_id,
                                             "type")
