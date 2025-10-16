import torch
import torch.nn as nn
from typing import List

class RegressionNet(nn.Module):
    """
    Fully connected neural network for regression.
    Takes input features and outputs a continuous value through hidden layers.
    """
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        last_dim = input_dim
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))  # output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)