"""PyTorch training loop for the Embedded PINN Orbit Transition Surrogate."""

from __future__ import annotations

import torch
from torch import nn

from src.ml.dataset_generator import generate_canonical_training_pairs
from src.ml.pinn_loss import pinn_physics_hamiltonian_loss


class ConditionedTransitionPINN(nn.Module):
    """Residual transition MLP mapping canonical state `s_k` (6D) -> `s_{k+1}` (6D).

    Matches the TFLite Micro embedded architecture (`6 -> 32 -> 32 -> 6` with residual skip).
    """

    def __init__(self, hidden_dim: int = 32, hidden_layers: int = 2) -> None:
        super().__init__()
        layers = []
        dim = 6
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.Tanh())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def train_pinn_surrogate(epochs: int = 40, lr: float = 3e-3) -> tuple[ConditionedTransitionPINN, list[float]]:
    """Train a `ConditionedTransitionPINN` on canonical DOP853 trajectories with Hamiltonian loss."""
    x_np, y_np, _ = generate_canonical_training_pairs(n_steps=80)
    x_t = torch.tensor(x_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)

    model = ConditionedTransitionPINN(hidden_dim=32, hidden_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_t)
        loss = pinn_physics_hamiltonian_loss(x_t, y_pred, y_t)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    model.eval()
    return model, loss_history
