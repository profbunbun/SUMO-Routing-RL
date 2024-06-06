class RideSelect:
    """
    Class to select a suitable vehicle for a person based on vehicle attributes.

    .. todo:: set this up for multiple taxi stands
    """
 
    def __init__(self) -> None:
        """
        Initialize the RideSelect class.
        """
       
        pass

    def select(self, vehicle_array, person):
        """
        Select a suitable vehicle for the given person.

        Args:
            vehicle_array (list): List of vehicle objects.
            person: Person object.

        Returns:
            str: Vehicle ID of the selected vehicle.

        Raises:
            ValueError: If no suitable vehicle is found.
        """
      
        
        v_dict = self.make_vehic_atribs_dic(vehicle_array)
        p_type = person.get_type()

        vehicle_id = {i for i in v_dict if v_dict[i]==p_type}

        # print(type(vehicle_id))
        return list(vehicle_id)[0]

    def make_vehic_atribs_dic(self, vehicle_array):
        """
        Create a dictionary of vehicle attributes.

        Args:
            vehicle_array (list): List of vehicle objects.

        Returns:
            dict: Dictionary of vehicle attributes.
        """
       
        
        v_dict = { }
        
        for v in vehicle_array:
            v_dict[v.vehicle_id] = v.get_type()

        # return v_a_dic
        return v_dict