"""Tiny residual transition model: state_t -> state_{t+dt} in normalized units.

The residual form (predict the small delta, add the input) fits orbital flow
far better than absolute prediction and keeps the network small enough for
TinyML deployment (a 2x32 net is ~9k params, ~36 KB as float32).
"""

from torch import nn


class TransitionMLP(nn.Module):
    def __init__(self, hidden_dim=32, hidden_layers=2):
        super().__init__()
        layers = []
        dim = 6
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.Tanh())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, 6] normalized state -> normalized next state
        return x + self.net(x)


def param_count(model):
    return sum(p.numel() for p in model.parameters())
