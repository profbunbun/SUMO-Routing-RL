
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
    '''
    SUMOConnection class for managing connections to the SUMO simulation.

    :param str label: Unique label for the connection.
    :param str sumocfg: Path to the SUMO configuration file.
    :param SUMO object sumo_: The SUMO simulation object.
    :param list sumo_cmd: Command list to start SUMO.

    :cvar CONNECTION_LABEL: Static variable to keep count of connections.
    '''

    
    CONNECTION_LABEL = 0

    def __init__(self, sumocfg: str) -> None:
        '''
        Initializes the SUMOConnection object with a specific SUMO configuration file.

        :param str sumocfg: Path to the SUMO configuration file.
        '''
        self.label = str(SUMOConnection.CONNECTION_LABEL)
        SUMOConnection.CONNECTION_LABEL += 1
        self.sumocfg = sumocfg
        self.sumo_ = None
        self.sumo_cmd = None

    def connect_gui(self):
        """
        Connects to the SUMO simulation with a Graphical User Interface.

        :return: The traci SUMO object after connection.
        :rtype: traci.Connection
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

        :return: The libsumo SUMO object after connection.
        :rtype: libsumo.Connection
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

        :return: The traci SUMO object after connection.
        :rtype: traci.Connection
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

    
