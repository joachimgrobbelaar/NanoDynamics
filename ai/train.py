"""Train the orbital PINN on simulator-generated data (with SI normalization).

Data source (first found wins):
  1. ai/data/dataset.npz (+ ai/data/stats.json) from generate_dataset.py
  2. trajectory.json fallback (single trajectory, legacy path)

Inputs are scaled (t/T_SCALE, r/R_EARTH, v/VEL_SCALE) so raw SI magnitudes
(~1e7 m) train stably; the physics loss is evaluated in normalized units
with mu scaled accordingly.

Usage:
    python3 ai/train.py --epochs 200 --hidden 64 --layers 4
    python3 ai/train.py --data ai/data --out ai/checkpoints
"""

import argparse
import json
import os
import sys

import numpy as np
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

DEFAULT_STATS = {"t_scale": 5400.0, "pos_scale": 6378137.0, "vel_scale": 8000.0}


def normalize(t, init_state, state, stats):
    """Scale raw SI tensors to O(1) training units."""
    ts, ps, vs = stats["t_scale"], stats["pos_scale"], stats["vel_scale"]
    sc = np.array([ps, ps, ps, vs, vs, vs], dtype=np.float32)
    return (t / ts, init_state / sc, state / sc)


def load_dataset(data_dir):
    """Load dataset.npz + stats.json produced by generate_dataset.py."""
    blob = np.load(os.path.join(data_dir, "dataset.npz"))
    with open(os.path.join(data_dir, "stats.json")) as f:
        stats = json.load(f)
    t = torch.tensor(blob["t"], dtype=torch.float32)
    init_state = torch.tensor(blob["init_state"], dtype=torch.float32)
    state = torch.tensor(blob["state"], dtype=torch.float32)
    return t, init_state, state, stats


def load_legacy_trajectory(filepath):
    """Legacy single-trajectory loader (trajectory.json)."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using dummy data for structural test.")
        t_dummy = torch.linspace(0, 10, 10).unsqueeze(1)
        return t_dummy, torch.zeros((10, 6)), torch.zeros((10, 6)), dict(DEFAULT_STATS)

    with open(filepath, "r") as f:
        data = json.load(f)
    times = [[row["t"]] for row in data]
    states = [[row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]] for row in data]
    t = torch.tensor(times, dtype=torch.float32)
    state = torch.tensor(states, dtype=torch.float32)
    init = state[0:1].repeat(len(times), 1) if times else state
    return t, init, state, dict(DEFAULT_STATS)


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


def compute_physics_loss(t, initial_state, model, mu):
    """Gravity residual in the (possibly normalized) units of the inputs."""
    t.requires_grad = True
    pred_state = model(t, initial_state)
    pos = pred_state[:, 0:3]
    vel = pred_state[:, 3:6]
    dp_dt = time_derivative(pos, t)
    dv_dt = time_derivative(vel, t)
    r = torch.norm(pos, dim=1, keepdim=True)
    expected_acc = -mu / (r ** 3) * pos
    vel_residual = torch.mean((dp_dt - vel) ** 2)
    acc_residual = torch.mean((dv_dt - expected_acc) ** 2)
    return vel_residual + acc_residual


def train(data_dir=None, out_dir=None, epochs=100, hidden=64, layers=4,
          lr=1e-3, physics_weight=1e-4):
    # mu in training units: mu' = mu / (pos_scale^3 / t_scale^2)
    if data_dir and os.path.exists(os.path.join(data_dir, "dataset.npz")):
        t_raw, init_raw, state_raw, stats = load_dataset(data_dir)
        print(f"Loaded dataset: {len(t_raw)} samples from {data_dir}")
    else:
        traj = os.path.join(PROJECT_ROOT, "trajectory.json")
        t_raw, init_raw, state_raw, stats = load_legacy_trajectory(traj)
    t, init_state, true_state = (torch.as_tensor(x, dtype=torch.float32)
                                 for x in normalize(t_raw, init_raw, state_raw, stats))
    mu = MU_EARTH / (stats["pos_scale"] ** 3 / stats["t_scale"] ** 2)

    model = OrbitalPINN(hidden_layers=layers, hidden_dim=hidden)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_state = model(t, init_state)
        data_loss = mse(pred_state, true_state)
        physics_loss = compute_physics_loss(t, init_state, model, mu)
        total_loss = data_loss + physics_weight * physics_loss
        total_loss.backward()
        optimizer.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Total={total_loss.item():.4e} "
                  f"Data={data_loss.item():.4e} Phys={physics_loss.item():.4e}")

    ckpt = {"state_dict": model.state_dict(), "stats": stats,
            "config": {"hidden_layers": layers, "hidden_dim": hidden}}
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "pinn.pt")
        torch.save(ckpt, path)
        print(f"Checkpoint saved -> {path}")
    print("Training complete.")
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(description="Train orbital PINN")
    ap.add_argument("--data", default=os.path.join(AI_DIR, "data"))
    ap.add_argument("--out", default=os.path.join(AI_DIR, "checkpoints"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(data_dir=args.data, out_dir=args.out, epochs=args.epochs,
          hidden=args.hidden, layers=args.layers, lr=args.lr)


if __name__ == "__main__":
    main()
