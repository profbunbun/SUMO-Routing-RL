import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..models.dqn import DQN
from ..exploration import exploration
from ..memory_buffers.permem import PrioritizedExperienceReplayBuffer, Experience
from utils.utils import Utils
config = Utils.load_yaml_config('/home/ahoope5/Desktop/SUMORL/SUMO-Routing-RL/src/configurations/config.yaml')

class PERAgent:
    """
    Agent class for DQN-based reinforcement learning with Prioritized Experience Replay (PER).

    Attributes:
        path (str): Path to save the model.
        memory_size (int): Size of the replay memory.
        gamma (float): Discount factor.
        learning_rate (float): Learning rate for the optimizer.
        epsilon_decay (float): Decay rate for epsilon.
        batch_size (int): Batch size for training.
        epsilon_max (float): Maximum value of epsilon.
        epsilon_min (float): Minimum value of epsilon.
        device (str): Device to run the model (CPU or GPU).
        policy_net (DQN): Policy network.
        target_net (DQN): Target network.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        exploration_strategy (exploration.Explorer): Exploration strategy.
        memory (PrioritizedExperienceReplayBuffer): Replay memory buffer.
    """
  
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
            """
        Initialize the PERAgent.

        Args:
            state_size (int): Size of the state space.
            action_size (int): Size of the action space.
            path (str): Path to save the model.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            epsilon_decay (float): Decay rate for epsilon.
            epsilon_max (float): Maximum value of epsilon.
            epsilon_min (float): Minimum value of epsilon.
            memory_size (int): Size of the replay memory.
            batch_size (int): Batch size for training.
        """

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
            

            alpha = config.get('alpha', 0.6)  # Hyperparameter for prioritized experience replay
            self.memory = PrioritizedExperienceReplayBuffer(batch_size=self.batch_size, buffer_size=memory_size, alpha=alpha)

            self.exploration_strategy = exploration.Explorer(self.policy_net, self.epsilon_max, self.epsilon_decay, self.epsilon_min)

            
            



    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay memory.

        Args:
            state (array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (array): Next state.
            done (bool): Whether the episode is done.
        """

        experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.add(experience)
    
  


    def replay(self, batch_size):
        """
        Train the agent with a batch of experiences.

        Args:
            batch_size (int): Batch size for training.
        """

        beta = config.get('beta', 0.4) 
        if len(self.memory) < self.batch_size:
            return

        idxs, experiences, weights = self.memory.sample(beta=beta)
        states, actions, rewards, next_states, dones = zip(*experiences)
        loss, td_errors = self.perform_training_step(states, actions, rewards, next_states, dones, weights)
        priorities = td_errors + config.get('priority_epsilon', 1e-5)
        self.memory.update_priorities(idxs, priorities.squeeze().cpu().detach().numpy())


    def perform_training_step(self, states, actions, rewards, next_states, dones, weights):
        """
        Perform a training step.

        Args:
            states (array): Batch of current states.
            actions (array): Batch of actions taken.
            rewards (array): Batch of rewards received.
            next_states (array): Batch of next states.
            dones (array): Batch of done flags.
            weights (array): Importance sampling weights.
        """

        states =torch.as_tensor(states, device=self.device, dtype=torch.float32) 
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)


        actions = actions.unsqueeze(-1) if actions.dim() == 1 else actions
        rewards = rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards
        dones = dones.unsqueeze(-1) if dones.dim() == 1 else dones
        weights = weights.unsqueeze(-1) if weights.dim() == 1 else weights
        current_q_values = self.policy_net(states).gather(1, actions)
       
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        td_error = expected_q_values - current_q_values
        loss = self.criterion(current_q_values, expected_q_values) * weights
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100, True)

        self.optimizer.step()

        return loss.item(), td_error.abs()

    def choose_action(self, state):
        """
        Choose an action based on the current state.

        Args:
            state (array): Current state.

        Returns:
            int: Chosen action.
        """

        action = self.exploration_strategy.choose_action(state)
        return action
    
    def hard_update(self):
        """
        Perform a hard update of the target network.
        """
    
        policy_net_state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(policy_net_state_dict)
        self.target_net.eval()

    def save_model(self, episode_num):
        """
        Save the model to a file.

        Args:
            episode_num (int): Episode number.
        """

        filename = f"model"
        filename += f"_ep{episode_num}"
        filename += ".pt"
        path = config['training_settings']['savepath']

        temp_model_path = os.path.join(self.path,path, filename)
        torch.save(self.policy_net.state_dict(), temp_model_path)

    def load_model(self, ep_num):
        """
        Load the model from a file.

        Args:
            ep_num (int): Episode number.
        """

        filename = f"model"
        filename += f"_ep{ep_num}"
        filename += ".pt"
        path = config['training_settings']['savepath']

        model_path = os.path.join(self.path,path, filename)

        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.to(self.device)

    def decay(self):
        """
        Decay the exploration rate.
        """
     
        self.exploration_strategy.update_epsilon()
    
    def get_epsilon(self):
        """
        Get the current value of epsilon.

        Returns:
            float: Current value of epsilon.
        """
        
        return self.exploration_strategy.epsilon
    
   
