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
        Select a suitable vehicle for the given person based on the vehicle type.

        Args:
            vehicle_array (list): List of vehicle objects.
            person: Person object with a method get_type() to retrieve the person's required vehicle type.

        Returns:
            str: Vehicle ID of the selected vehicle.

        Raises:
            ValueError: If no suitable vehicle is found.
        """
        
        # Create a dictionary of vehicle IDs mapped to their types
        v_dict = self.make_vehic_atribs_dic(vehicle_array)
        
        # Retrieve the type of vehicle required by the person
        p_type = person.get_type()
        
        # Find all vehicle IDs where the vehicle type matches the person's required type
        matching_vehicles = {vid for vid, vtype in v_dict.items() if vtype == p_type}
        
        # Check if any suitable vehicle was found
        if not matching_vehicles:
            raise ValueError("No suitable vehicle found.")
        
        # Return the first matching vehicle ID
        return next(iter(matching_vehicles))

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