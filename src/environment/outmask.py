
class OutMask:
    """
    Class to create a mask for valid outgoing directions from the current lane.
    """

    def __init__(self) -> None:
        """
        Initialize the OutMask class.
        """
        pass

    def get_outmask_valid(self, choices):
        """
        Get a mask of valid outgoing directions from the current lane.

        Args:
            out_dict (dict): Dictionary of outgoing directions.

        Returns:
            list: Mask of valid outgoing directions (1 if valid, 0 if not).
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
