
class OutMask:
    """
    OutMask class to generate output masks based on vehicle edge and position data.

    This class provides methods to generate masks for determining the valid movement directions and distance calculations.
    """

    

    def get_outmask_valid(self, choices):
        """
        Generate a validity mask for the available movement choices.

        :param dict choices: Dictionary of movement choices from the current edge.
        :return: List representing the output mask for valid movements.
        :rtype: list

        The output mask is a list with values 1 for valid movements and 0 for invalid, in the order of right, straight, left, and to.
        """

        outmask =[0,0,0,0,0,0]

        if choices is None:
            outmask = outmask
        else:

            for choice in choices.items():
                if choice[0] == "R":
                    outmask[0] = 1

                elif choice[0] == "r":
                    outmask[1] = 1

                elif choice[0] == "s":
                    outmask[2] = 1

                elif choice[0] == "L":
                    outmask[3] = 1
                
                elif choice[0] == "l":
                    outmask[4] = 1
                
                elif choice[0] == "t":
                    outmask[5] = 1

        return outmask
