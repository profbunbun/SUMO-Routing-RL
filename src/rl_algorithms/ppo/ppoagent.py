import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from .ppoagent import PPOPolicy  # Assume PPOPolicy is your combined actor-critic network
from ..utilities.utils import Utils
config = Utils.load_yaml_config('config.yaml')

class PPOAgent:

    def __init__(self, state_size, action_size, path, learning_rate=1e-3, gamma=0.99, clip_param=0.2, ppo_epochs=4, mini_batch_size=64):
        self.path = path
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPOPolicy(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()  # Sample an action
        log_prob = dist.log_prob(action)  # Get the log probability of the action
        return action.item(), log_prob.item()

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        for _ in range(self.ppo_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(states.size(0))), self.mini_batch_size, False)
            for indices in sampler:
                sampled_states = states[indices]
                sampled_actions = actions[indices]
                sampled_old_log_probs = old_log_probs[indices]
                sampled_returns = returns[indices]
                sampled_advantages = advantages[indices]

                # Get the current predictions
                action_probs, state_values = self.policy(sampled_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(sampled_actions)

                # Calculate the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(new_log_probs - sampled_old_log_probs)

                # Calculate Surrogate Loss
                surr1 = ratios * sampled_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * sampled_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (sampled_returns - state_values).pow(2).mean()

                # Total loss
                loss = 0.5 * value_loss + policy_loss

                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save_model(self, episode_num):
        filename = f"ppo_policy_ep{episode_num}.pt"
        path = config['training_settings']['savepath']
        model_path = os.path.join(self.path, path, filename)
        torch.save(self.policy.state_dict(), model_path)

    def load_model(self, ep_num):
        filename = f"ppo_policy_ep{ep_num}.pt"
        path = config['training_settings']['savepath']
        model_path = os.path.join(self.path, path, filename)
        self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy.to(self.device)
