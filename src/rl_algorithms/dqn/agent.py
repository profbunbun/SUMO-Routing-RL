import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..models import DQN
from ..exploration import exploration
from ..memory_buffers import replay_memory
from utils.utils import Utils
config = Utils.load_yaml_config('src/config/config.yaml')

class Agent:
  
    def __init__(self, state_size, 
                 action_size, 
                 path,
                 learning_rate=None, 
                 gamma=None, 
                 epsilon_decay=None, 
                 epsilon_max=None, 
                 epsilon_min=None, 
                 memory_size=None,
                 batch_size=None,
                 ):

            self.path = path
          

            self.memory_size = memory_size 
            self.gamma = gamma 
            self.learning_rate = learning_rate 
            self.epsilon_decay = epsilon_decay 

            self.batch_size = batch_size 
            self.epsilon_max = epsilon_max 
            self.epsilon_min = epsilon_min 

            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.policy_net = DQN(state_size, action_size).to(self.device)
            self.target_net = DQN(state_size, action_size).to(self.device)


            self.criterion = nn.HuberLoss()

            self.optimizer = optim.RMSprop(self.policy_net.parameters(),
                                        lr=self.learning_rate, momentum=0.9)
            # self.optimizer = optim.AdamW(self.policy_net.parameters(),
            #                             lr=self.learning_rate, amsgrad=True)
            self.exploration_strategy = exploration.Explorer(self.policy_net, self.epsilon_max, self.epsilon_decay, self.epsilon_min)

            self.memory = replay_memory.ReplayMemory(self.memory_size)
            



    def remember(self, state, action, reward, next_state, done):

        self.memory.remember(state, action, reward, next_state, done)
    
  


    def replay(self, batch_size):

        minibatch = self.memory.replay_batch(batch_size)
        if len(minibatch) == 0:
            return None
        states, actions, next_states, rewards, dones = zip(*minibatch)
        self.perform_training_step(states, actions, next_states, rewards, dones)


    def perform_training_step(self, states, actions, rewards, next_states, dones):

        states =torch.as_tensor(states, device=self.device, dtype=torch.float32) 
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32)


        actions = actions.unsqueeze(-1) if actions.dim() == 1 else actions
        rewards = rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards
        dones = dones.unsqueeze(-1) if dones.dim() == 1 else dones


        current_q_values = self.policy_net(states).gather(1, actions)
       
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)



        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)


        loss = self.criterion(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100, True)

        self.optimizer.step()

        return 

    def choose_action(self, state):

        action, index = self.exploration_strategy.choose_action(state)
        return action, index
    
    def hard_update(self):
    
        policy_net_state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(policy_net_state_dict)
        self.target_net.eval()

    def save_model(self, episode_num):

        filename = f"model"
        filename += f"_ep{episode_num}"
        filename += ".pt"
        path = config['training_settings']['savepath']

        temp_model_path = os.path.join(self.path,path, filename)
        torch.save(self.policy_net.state_dict(), temp_model_path)

    def load_model(self, ep_num):

        filename = f"model"
        filename += f"_ep{ep_num}"
        filename += ".pt"
        path = config['training_settings']['savepath']

        model_path = os.path.join(self.path,path, filename)

        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.to(self.device)

    def decay(self):
     
        self.exploration_strategy.update_epsilon()
    
    def get_epsilon(self):
        
        return self.exploration_strategy.epsilon
    
   
