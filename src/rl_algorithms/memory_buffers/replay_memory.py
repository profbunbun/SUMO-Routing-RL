
from collections import namedtuple, deque

import random
from ..utilities.utils import Utils

config = Utils.load_yaml_config('config.yaml')
randy = config['training_settings']['seed']
random.seed(randy)
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state',  'done'))

class ReplayMemory(object):
    def __init__(self, capacaty):
        
        self.memory = deque([],maxlen=capacaty)

    def remember(self, *args):
        self.memory.append(Transition(*args))

    def replay_batch(self, batch_size):
        return random.sample(self.memory, batch_size)
     
    
    def __len__(self):
        return len(self.memory)