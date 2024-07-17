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
            nn.Linear(n_observations,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,n_actions)
            )
        
        
        # self.layer1 = nn.Linear(n_observations, 32)
        # self.layer2 = nn.Linear(32, 64)
        # self.layer2b = nn.Linear(64, 32)
        # self.layer2c = nn.Linear(32, 16)
        # self.layer3 = nn.Linear(16,  n_actions)

        # self.layers.to(self.device)
    


    def forward(self, x):
        """
        Forward pass of the DQN model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        
        # x_net = F.relu(self.layer1(x_net))
        # # x_net = self.dropout1(x_net)
        # x_net = F.relu(self.layer2(x_net))
        # # x_net = self.dropout2(x_net)
        # x_net = F.relu(self.layer2b(x_net))
        # # x_net = self.dropout3(x_net)
        # x_net = F.relu(self.layer2c(x_net))
        # x_net = self.layer3(x_net)
        # # x_net = F.log_softmax(x_net, dim=1) 
        # return x_net
        
        return self.layers(x)
    
