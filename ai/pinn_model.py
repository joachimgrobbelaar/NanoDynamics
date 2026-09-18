import torch
import torch.nn as nn

class OrbitalPINN(nn.Module):
    def __init__(self, hidden_layers=4, hidden_dim=64):
        super(OrbitalPINN, self).__init__()
        
        # Input: time (1) + initial_state (6) = 7
        # Output: state (6) [x, y, z, vx, vy, vz]
        layers = []
        layers.append(nn.Linear(7, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, 6))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, t, initial_state):
        # t: [batch, 1]
        # initial_state: [batch, 6]
        inputs = torch.cat([t, initial_state], dim=1)
        return self.network(inputs)
