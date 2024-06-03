class RideSelect:
 
    def __init__(self) -> None:
       
        pass

    def select(self, vehicle_array, person):
      
        
        v_dict = self. make_vehic_atribs_dic(vehicle_array)
        p_type = person.get_type()

        vehicle_id = {i for i in v_dict if v_dict[i]==p_type}

        # print(type(vehicle_id))
        return list(vehicle_id)[0]

    def make_vehic_atribs_dic(self, vehicle_array):
       
        
        v_dict = { }
        
        for v in vehicle_array:
            v_dict[v.vehicle_id] = v.get_type()

        # return v_a_dic
        return v_dict