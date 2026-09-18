import json
import os
import sys

import torch
import torch.optim as optim

# Allow running as `python3 ai/train.py` from the project root or as
# `python3 train.py` from inside ai/.
AI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(AI_DIR)
for _path in (AI_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pinn_model import OrbitalPINN  # noqa: E402

from leo_simulator.constants import MU_EARTH  # noqa: E402

# Physics loss uses SI units (meters) to match trajectory.json.
MU = MU_EARTH


def time_derivative(vec, t):
    """Per-component time derivative of a [batch, 3] tensor w.r.t. t [batch, 1].

    Rows are independent network evaluations, so differentiating the summed
    component w.r.t. t yields each row's own partial derivative.
    """
    grads = []
    for i in range(vec.shape[1]):
        g = torch.autograd.grad(vec[:, i].sum(), t, create_graph=True)[0]
        grads.append(g)
    return torch.cat(grads, dim=1)


def compute_physics_loss(t, initial_state, model):
    t.requires_grad = True

    # Predict state [batch, 6] -> x, y, z, vx, vy, vz
    pred_state = model(t, initial_state)

    pos = pred_state[:, 0:3]
    vel = pred_state[:, 3:6]

    # Time derivatives with matching [batch, 3] shapes
    dp_dt = time_derivative(pos, t)
    dv_dt = time_derivative(vel, t)

    # r = sqrt(x^2 + y^2 + z^2)
    r = torch.norm(pos, dim=1, keepdim=True)

    # Newtonian gravity: a = -mu / r^3 * r_vec
    expected_acc = -MU / (r**3) * pos
    expected_vel = vel

    # Physics residuals
    vel_residual = torch.mean((dp_dt - expected_vel)**2)
    acc_residual = torch.mean((dv_dt - expected_acc)**2)

    return vel_residual + acc_residual


def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using dummy data for structural test.")
        # Return small dummy data
        t_dummy = torch.linspace(0, 10, 10).unsqueeze(1)
        init_state_dummy = torch.zeros((10, 6))
        true_state_dummy = torch.zeros((10, 6))
        return t_dummy, init_state_dummy, true_state_dummy

    with open(filepath, 'r') as f:
        data = json.load(f)

    times = []
    states = []

    for row in data:
        times.append([row['t']])
        states.append([
            row['x'], row['y'], row['z'],
            row['vx'], row['vy'], row['vz']
        ])

    t_tensor = torch.tensor(times, dtype=torch.float32)
    state_tensor = torch.tensor(states, dtype=torch.float32)

    if len(times) > 0:
        initial_state_tensor = state_tensor[0:1].repeat(len(times), 1)
    else:
        initial_state_tensor = state_tensor

    return t_tensor, initial_state_tensor, state_tensor


def train():
    model = OrbitalPINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss_fn = torch.nn.MSELoss()

    # Load trajectory from parent directory
    trajectory_path = os.path.join(os.path.dirname(__file__), '..', 'trajectory.json')
    t, initial_state, true_state = load_data(trajectory_path)

    epochs = 100
    physics_weight = 1e-4

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. Data Loss
        pred_state = model(t, initial_state)
        data_loss = mse_loss_fn(pred_state, true_state)

        # 2. Physics Loss
        physics_loss = compute_physics_loss(t, initial_state, model)

        # Total Loss
        total_loss = data_loss + physics_weight * physics_loss

        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss={total_loss.item():.4e}, Data Loss={data_loss.item():.4e}, Physics Loss={physics_loss.item():.4e}")

    print("Training loop complete (structural test passed).")


if __name__ == '__main__':
    train()
