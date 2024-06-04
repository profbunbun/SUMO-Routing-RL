import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) class.

    Attributes:
        layers (nn.Sequential): Sequential model consisting of linear layers, batch normalization, and activation functions.
    """

    def __init__(self, n_observations, n_actions):
        """
        Initialize the DQN model.

        Args:
            n_observations (int): Number of observations (input features).
            n_actions (int): Number of actions (output features).
        """

    
        super().__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.layers = nn.Sequential(
            nn.Linear(n_observations,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
          
            nn.Linear(64,n_actions)
            )

        # self.layers.to(self.device)
    


    def forward(self, x):
        """
        Forward pass of the DQN model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        
        return self.layers(x)
    
