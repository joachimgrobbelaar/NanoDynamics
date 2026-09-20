"""Evaluate a trained PINN against solve_ivp ground truth (the comparison loop).

Propagates a held-out orbit with the numerical propagator, predicts the same
trajectory with the checkpoint, and reports position drift as % of orbital
radius plus velocity error — the <5%-over-one-orbit style acceptance check.

Usage:
    python3 ai/evaluate.py --ckpt ai/checkpoints/pinn.pt [--alt-km 420 --inc-deg 60]
"""

import argparse
import os
import sys

import numpy as np
import torch

AI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(AI_DIR)
for _path in (AI_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pinn_model import OrbitalPINN  # noqa: E402
from train import normalize  # noqa: E402

from leo_simulator import (R_EARTH, ExponentialAtmosphere, OrbitalElements,  # noqa: E402
                           OrbitPropagator, Satellite)


def evaluate(ckpt_path, alt_km=420.0, inc_deg=60.0, orbits=1.0, dt=60.0):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    cfg = ckpt["config"]
    model = OrbitalPINN(hidden_layers=cfg["hidden_layers"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    oe = OrbitalElements(a=R_EARTH + alt_km * 1000.0, e=0.001,
                         i=np.radians(inc_deg), raan=np.radians(30.0),
                         arg_pe=0.0, nu=0.0)
    prop = OrbitPropagator(satellite=Satellite.cubesat_3u(),
                           atmosphere=ExponentialAtmosphere(),
                           include_central_gravity=True, include_j2=True,
                           include_drag=True)
    res = prop.propagate(oe, duration_seconds=float(orbits * oe.period), dt_eval=dt)

    y0 = np.concatenate([res.r[0], res.v[0]])
    n = len(res.t)
    t_raw = torch.tensor(res.t.reshape(n, 1), dtype=torch.float32)
    init_raw = torch.tensor(np.tile(y0, (n, 1)), dtype=torch.float32)
    true_raw = torch.tensor(np.hstack([res.r, res.v]), dtype=torch.float32)
    t, init, _ = (torch.as_tensor(x, dtype=torch.float32)
                  for x in normalize(t_raw, init_raw, true_raw, stats))
    with torch.no_grad():
        pred = model(t, init).numpy()

    ps = stats["pos_scale"]
    pred_r = pred[:, 0:3] * ps
    pos_err = np.linalg.norm(pred_r - res.r, axis=1)
    drift_pct = 100.0 * pos_err / np.linalg.norm(res.r, axis=1)
    vel_err = np.linalg.norm(pred[:, 3:6] * stats["vel_scale"] - res.v, axis=1)

    print(f"Held-out orbit: alt0={alt_km}km inc={inc_deg}deg, {orbits} orbit(s), {n} pts")
    print(f"  final position drift: {drift_pct[-1]:.3f}% "
          f"({'PASS' if drift_pct[-1] < 5.0 else 'FAIL'} vs 5% tolerance)")
    print(f"  mean drift: {drift_pct.mean():.3f}% | max drift: {drift_pct.max():.3f}%")
    print(f"  mean |v| err: {vel_err.mean():.1f} m/s")
    return {"drift_pct": drift_pct, "vel_err": vel_err}


def evaluate_rollout(ckpt_path, alt_km=420.0, inc_deg=60.0, orbits=1.0):
    """Roll out the transition net step-by-step vs solve_ivp ground truth."""
    from transition_model import TransitionMLP  # noqa: E402
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    cfg = ckpt["config"]
    model = TransitionMLP(hidden_dim=cfg["hidden_dim"],
                          hidden_layers=cfg["hidden_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    sc = np.array([stats["pos_scale"]] * 3 + [stats["vel_scale"]] * 3,
                  dtype=np.float32)
    dt = float(stats["dt"])

    oe = OrbitalElements(a=R_EARTH + alt_km * 1000.0, e=0.001,
                         i=np.radians(inc_deg), raan=np.radians(30.0),
                         arg_pe=0.0, nu=0.0)
    prop = OrbitPropagator(satellite=Satellite.cubesat_3u(),
                           atmosphere=ExponentialAtmosphere(),
                           include_central_gravity=True, include_j2=True,
                           include_drag=True)
    res = prop.propagate(oe, duration_seconds=float(orbits * oe.period), dt_eval=dt)

    with torch.no_grad():
        x = torch.tensor(np.concatenate([res.r[0], res.v[0]]) / sc,
                         dtype=torch.float32).unsqueeze(0)
        preds = []
        for _ in range(len(res.t) - 1):
            x = model(x)
            preds.append(x.squeeze(0).numpy())
    preds = np.array(preds) * sc
    pos_err = np.linalg.norm(preds[:, 0:3] - res.r[1:], axis=1)
    drift_pct = 100.0 * pos_err / np.linalg.norm(res.r[1:], axis=1)

    print(f"Rollout on held-out orbit: alt0={alt_km}km inc={inc_deg}deg, "
          f"{orbits} orbit(s), dt={dt}s")
    print(f"  final position drift: {drift_pct[-1]:.3f}% "
          f"({'PASS' if drift_pct[-1] < 5.0 else 'FAIL'} vs 5% tolerance)")
    print(f"  mean drift: {drift_pct.mean():.3f}% | max drift: {drift_pct.max():.3f}%")
    return {"drift_pct": drift_pct}


def main():
    ap = argparse.ArgumentParser(description="Evaluate NN vs simulation")
    ap.add_argument("--model", choices=("pinn", "transition"), default="pinn")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--alt-km", type=float, default=420.0)
    ap.add_argument("--inc-deg", type=float, default=60.0)
    ap.add_argument("--orbits", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=60.0)
    args = ap.parse_args()
    if args.model == "transition":
        ckpt = args.ckpt or os.path.join(AI_DIR, "checkpoints", "transition.pt")
        evaluate_rollout(ckpt, args.alt_km, args.inc_deg, args.orbits)
    else:
        ckpt = args.ckpt or os.path.join(AI_DIR, "checkpoints", "pinn.pt")
        evaluate(ckpt, args.alt_km, args.inc_deg, args.orbits, args.dt)


if __name__ == "__main__":
    main()
