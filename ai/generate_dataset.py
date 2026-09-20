"""Automated training-data collection for the orbital PINN/ML surrogate.

Sweeps randomized LEO initial conditions, propagates each with the verified
numerical propagator (solve_ivp ground truth), and stores a normalized
(t, initial_state) -> state dataset plus scaling stats for training.

Usage:
    python3 ai/generate_dataset.py --n-traj 20 --orbits 1 --dt 60 --seed 0
    python3 ai/generate_dataset.py --help
"""

import argparse
import json
import os
import sys

import numpy as np

AI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(AI_DIR)
for _path in (AI_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from leo_simulator import (R_EARTH, ExponentialAtmosphere, OrbitalElements,  # noqa: E402
                           OrbitPropagator, Satellite)

# Fixed SI scales so raw meter/second values train stably.
T_SCALE = 5400.0        # ~one LEO orbital period [s]
POS_SCALE = R_EARTH     # [m]
VEL_SCALE = 8000.0      # [m/s]


def sample_initial_orbit(rng, alt_min_m, alt_max_m):
    """Draw a random near-circular LEO orbit (angles uniform, e small)."""
    alt = float(rng.uniform(alt_min_m, alt_max_m))
    return OrbitalElements(
        a=R_EARTH + alt,
        e=float(rng.uniform(0.0, 0.01)),
        i=float(rng.uniform(0.0, np.radians(98.0))),
        raan=float(rng.uniform(0.0, 2 * np.pi)),
        arg_pe=float(rng.uniform(0.0, 2 * np.pi)),
        nu=float(rng.uniform(0.0, 2 * np.pi)),
    )


def generate(n_traj=20, orbits=1.0, dt=60.0, alt_min_km=300.0, alt_max_km=600.0,
             seed=0, out_dir=None):
    """Propagate n_traj random orbits; return (t, init_state, state) arrays.

    Rows follow the (t, initial_state) -> state formulation used by
    ai/pinn_model.py. Times are absolute seconds from each trajectory start.
    """
    rng = np.random.default_rng(seed)
    sat = Satellite.cubesat_3u()
    atm = ExponentialAtmosphere()
    prop = OrbitPropagator(satellite=sat, atmosphere=atm,
                           include_central_gravity=True, include_j2=True,
                           include_drag=True)

    t_list, init_list, state_list = [], [], []
    for k in range(n_traj):
        oe = sample_initial_orbit(rng, alt_min_km * 1000.0, alt_max_km * 1000.0)
        duration = float(orbits * oe.period)
        res = prop.propagate(oe, duration_seconds=duration, dt_eval=dt)
        y0 = np.concatenate([res.r[0], res.v[0]])
        n = len(res.t)
        t_list.append(res.t.reshape(n, 1))
        init_list.append(np.tile(y0, (n, 1)))
        state_list.append(np.hstack([res.r, res.v]))
        print(f"  traj {k + 1}/{n_traj}: alt0={res.initial_altitude / 1000:.1f}km "
              f"pts={n} decay={res.altitude_decay:.1f}m")

    data = {
        "t": np.vstack(t_list),
        "init_state": np.vstack(init_list),
        "state": np.vstack(state_list),
    }
    stats = {"t_scale": T_SCALE, "pos_scale": POS_SCALE, "vel_scale": VEL_SCALE,
             "n_traj": n_traj, "orbits": orbits, "dt": dt, "seed": seed,
             "alt_min_km": alt_min_km, "alt_max_km": alt_max_km,
             "physics": "central+J2+drag"}

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "dataset.npz"), **data)
        with open(os.path.join(out_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved {sum(len(v) for v in [data['t']])} samples -> {out_dir}/")

    return data, stats


def main():
    ap = argparse.ArgumentParser(description="Generate PINN training dataset")
    ap.add_argument("--n-traj", type=int, default=20)
    ap.add_argument("--orbits", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--alt-min-km", type=float, default=300.0)
    ap.add_argument("--alt-max-km", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(AI_DIR, "data"))
    args = ap.parse_args()
    generate(n_traj=args.n_traj, orbits=args.orbits, dt=args.dt,
             alt_min_km=args.alt_min_km, alt_max_km=args.alt_max_km,
             seed=args.seed, out_dir=args.out)


if __name__ == "__main__":
    main()
