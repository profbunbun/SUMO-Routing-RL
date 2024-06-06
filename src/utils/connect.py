
import sys
import os
import traci
import libsumo



if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")


class SUMOConnection:
    """
    SUMOConnection class for managing connections to the SUMO simulation.

    Attributes:
        label (str): Unique label for the connection.
        sumocfg (str): Path to the SUMO configuration file.
        sumo_ (object): The SUMO simulation object.
        sumo_cmd (list): Command list to start SUMO.
    """

    
    CONNECTION_LABEL = 0

    def __init__(self, sumocfg: str) -> None:
        """
        Initializes the SUMOConnection object with a specific SUMO configuration file.

        Args:
            sumocfg (str): Path to the SUMO configuration file.
        """
        global SUMO_CONNECTION_LABEL
        self.label = str(SUMOConnection.CONNECTION_LABEL)
        SUMOConnection.CONNECTION_LABEL += 1
        self.sumocfg = sumocfg
        self.sumo_ = None
        self.sumo_cmd = None

    def connect_gui(self):
        """
        Connects to the SUMO simulation with a Graphical User Interface.

        Returns:
            traci.Connection: The traci SUMO object after connection.
        """
        self.sumo_cmd = [
            "sumo-gui",
            "-c",
            self.sumocfg,
            "-d",
            "50",
            "--start",
            "--quit-on-end",
            "--human-readable-time",
        ]
        traci.start(self.sumo_cmd, label=self.label)
        self.sumo_ = traci
        # self.sumo_.addStepListener(self.listener)
        
        return self.sumo_

    def connect_libsumo_no_gui(self):
        """
        Connects to the SUMO simulation without a GUI using libsumo.

        Returns:
            libsumo.Connection: The libsumo SUMO object after connection.
        """

        self.sumo_cmd = [
            "sumo",
            "-c",
            self.sumocfg,
            "--no-warnings",
            "true"
        ]
        libsumo.start(self.sumo_cmd, label=self.label)
        self.sumo_ = libsumo
        # self.sumo_.addStepListener(self.listener)
        return self.sumo_

    def connect_no_gui(self):
        """
        Connects to the SUMO simulation without a Graphical User Interface.

        Returns:
            traci.Connection: The traci SUMO object after connection.
        """
  
        self.sumo_cmd = [
            "sumo",
            "-c",
            self.sumocfg,
        ]
        traci.start(self.sumo_cmd, label=self.label)
        self.sumo_ = traci

        return self.sumo_

    def close(self):
        """
        Closes the connection to the SUMO simulation.
        """
 
        self.sumo_.close()

    
