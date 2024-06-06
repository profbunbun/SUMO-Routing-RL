
from collections import namedtuple, deque

import random
from utils.utils import Utils

# config = Utils.load_yaml_config('/home/ahoope5/Desktop/SUMORL/SUMO-Routing-RL/src/configurations/config.yaml')
# randy = config['training_settings']['seed']
# random.seed(randy)
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state',  'done'))

class ReplayMemory(object):
    """
    Replay memory buffer for storing and sampling experiences.

    Attributes:
        memory (deque): Deque to store experiences with a fixed maximum length.
    """
    

    def __init__(self, capacaty, seed = 42):
        """
        Initialize the ReplayMemory with the given capacity.

        Args:
            capacaty (int): Maximum number of experiences to store in the buffer.
        """
        random.seed(seed)
        
        self.memory = deque([],maxlen=capacaty)

    def remember(self, *args):
        """
        Store an experience in the replay memory.

        Args:
            *args: Components of the experience to store.
        """
        self.memory.append(Transition(*args))

    def replay_batch(self, batch_size):
        """
        Sample a batch of experiences from the replay memory.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            list: List of sampled experiences.
        """
        return random.sample(self.memory, batch_size)
     
    
    def __len__(self):
        """
        Get the current size of the replay memory.

        Returns:
            int: Current number of experiences stored in the buffer.
        """
        return len(self.memory)