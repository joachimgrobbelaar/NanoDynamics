"""Train the one-step transition model and export it for deployment.

Pipeline: dataset_onestep.npz -> TransitionMLP -> ai/checkpoints/transition.pt
(checkpoint + TorchScript trace + stats for the ESP32/TFLite port).

Usage:
    python3 ai/train_transition.py --epochs 300 --hidden 32 --layers 2
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import optim

AI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(AI_DIR)
for _path in (AI_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from transition_model import TransitionMLP, param_count


def train(data_dir=None, out_dir=None, epochs=300, hidden=32, layers=2,
          lr=1e-3, batch_size=256, seed=0):
    data_dir = data_dir or os.path.join(AI_DIR, "data")
    blob = np.load(os.path.join(data_dir, "dataset_onestep.npz"))
    with open(os.path.join(data_dir, "stats_onestep.json")) as f:
        stats = json.load(f)
    sc = np.array([stats["pos_scale"]] * 3 + [stats["vel_scale"]] * 3,
                  dtype=np.float32)
    x = torch.tensor(blob["x"] / sc, dtype=torch.float32)
    y = torch.tensor(blob["y"] / sc, dtype=torch.float32)
    print(f"Loaded {len(x)} one-step pairs (dt={stats['dt']}s) from {data_dir}")

    torch.manual_seed(seed)
    model = TransitionMLP(hidden_dim=hidden, hidden_layers=layers)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    n = len(x)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = mse(model(x[b]), y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: MSE={tot / n:.4e}")

    ckpt = {"state_dict": model.state_dict(), "stats": stats,
            "config": {"hidden_dim": hidden, "hidden_layers": layers}}
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(out_dir, "transition.pt"))
        example = torch.zeros(1, 6)
        traced = torch.jit.trace(model.eval(), example)
        traced.save(os.path.join(out_dir, "transition_traced.pt"))
        print(f"Saved checkpoint + TorchScript trace -> {out_dir}/ "
              f"({param_count(model)} params, ~{param_count(model) * 4 / 1024:.0f} KB fp32)")
    print("Training complete.")
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(description="Train one-step transition model")
    ap.add_argument("--data", default=os.path.join(AI_DIR, "data"))
    ap.add_argument("--out", default=os.path.join(AI_DIR, "checkpoints"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(data_dir=args.data, out_dir=args.out, epochs=args.epochs,
          hidden=args.hidden, layers=args.layers, lr=args.lr)


if __name__ == "__main__":
    main()
