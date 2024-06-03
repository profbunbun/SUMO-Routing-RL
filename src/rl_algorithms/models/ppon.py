import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOPolicy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(PPOPolicy, self).__init__()
  
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.action_head = nn.Linear(hidden_size, action_size)

        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.shared_layers(state)
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values
