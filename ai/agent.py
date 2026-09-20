"""Local ML agent: simulation data -> transition net -> comparison report.

Each cycle: ingest every simulation output (saved_sims, trajectory.json,
sim_log.jsonl from POST /simulate, --csv exports), retrain only when new
pairs arrived, then roll out the net on a held-out orbit vs solve_ivp.

Usage:
    python3 ai/agent.py --once [--epochs 300 --hidden 32 --layers 2]
    python3 ai/agent.py --watch --interval 300   # poll for new sim data
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

AI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(AI_DIR)
for _path in (AI_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluate import evaluate_rollout
from ingest import ingest, load_earth_pairs
from train_transition import train as train_transition


def _file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def run_once(data_dir=None, ckpt_dir=None, csv_paths=(), dt=60.0, min_pairs=50,
             epochs=300, hidden=32, layers=2, lr=1e-3,
             alt_km=420.0, inc_deg=60.0, force=False, resume=True):
    data_dir = data_dir or os.path.join(AI_DIR, "data")
    ckpt_dir = ckpt_dir or os.path.join(AI_DIR, "checkpoints", "agent")
    state_path = os.path.join(data_dir, "agent_state.json")
    ckpt_path = os.path.join(ckpt_dir, "transition.pt")

    _, _, manifest = ingest(csv_paths=list(csv_paths), dt_target=dt, out_dir=data_dir)
    pairs_path = os.path.join(data_dir, "sim_pairs.npz")
    if manifest["n_pairs"] < min_pairs or not os.path.exists(pairs_path):
        msg = (f"waiting for data: {manifest['n_pairs']} sim pairs "
               f"(need {min_pairs}). Run simulations (web UI Save or POST "
               f"/simulate) or pass --csv, then re-run.")
        print(msg)
        return {"status": "waiting", "n_pairs": manifest["n_pairs"]}

    data_hash = _file_hash(pairs_path)
    state = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    if not force and state.get("data_hash") == data_hash and os.path.exists(ckpt_path):
        print(f"No new sim data ({manifest['n_pairs']} pairs) — keeping "
              f"existing checkpoint. Last drift: {state.get('final_drift_pct')}%")
        return {"status": "up-to-date", **state}

    # Combine sweep pairs (same dt only) with sim pairs for training.
    sim_x, sim_y, earth_info = load_earth_pairs(data_dir)
    if earth_info["dropped"]:
        print(f"Excluding non-Earth run(s) from training: {earth_info['dropped']}")
    if earth_info["n_kept"] == 0:
        print("No Earth-compatible sim pairs — waiting for data.")
        return {"status": "waiting", "n_pairs": manifest["n_pairs"]}
    xs, ys = [sim_x], [sim_y]
    sweep = os.path.join(data_dir, "dataset_onestep.npz")
    sweep_stats = os.path.join(data_dir, "stats_onestep.json")
    if os.path.exists(sweep) and os.path.exists(sweep_stats):
        with open(sweep_stats) as f:
            sstats = json.load(f)
        if abs(float(sstats.get("dt", -1)) - dt) / dt <= 0.01:
            blob = np.load(sweep)
            xs.append(blob["x"])
            ys.append(blob["y"])
            print(f"Adding {len(blob['x'])} sweep pairs to {earth_info['n_kept']} sim pairs.")
        else:
            print(f"Skipping sweep data (dt={sstats.get('dt')}s != {dt}s).")
    combined = os.path.join(data_dir, f"combined_dt{int(dt)}")
    os.makedirs(combined, exist_ok=True)
    np.savez_compressed(os.path.join(combined, "dataset_onestep.npz"),
                        x=np.vstack(xs), y=np.vstack(ys))
    with open(os.path.join(combined, "stats_onestep.json"), "w") as f:
        json.dump({"pos_scale": 6378137.0, "vel_scale": 8000.0, "dt": dt,
                   "n_sim_pairs": int(earth_info["n_kept"]),
                   "n_total_pairs": int(sum(len(a) for a in xs)),
                   "physics": "central+J2+drag (see sim_manifest.json)"}, f, indent=2)

    _, _ckpt = train_transition(data_dir=combined, out_dir=ckpt_dir,
                                  epochs=epochs, hidden=hidden, layers=layers, lr=lr,
                                  resume=resume)
    res = evaluate_rollout(ckpt_path, alt_km=alt_km, inc_deg=inc_deg, orbits=1.0)
    final_drift = float(res["drift_pct"][-1])
    mean_drift = float(res["drift_pct"].mean())
    max_drift = float(res["drift_pct"].max())

    import datetime
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = {
        "timestamp": now_iso,
        "status": "trained",
        "data_hash": data_hash,
        "n_pairs": int(sum(len(a) for a in xs)),
        "ckpt": ckpt_path,
        "final_drift_pct": round(final_drift, 3),
        "mean_drift_pct": round(mean_drift, 3),
        "max_drift_pct": round(max_drift, 3),
        "pass_5pct": bool(final_drift < 5.0),
        "eval_orbit": {"alt_km": alt_km, "inc_deg": inc_deg, "orbits": 1.0, "dt": dt},
        "resumed": bool(resume),
        "epochs": epochs,
    }
    with open(state_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(data_dir, "agent_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Append to persistent prediction accuracy history log
    eval_log_path = os.path.join(data_dir, "evaluation_history.jsonl")
    try:
        with open(eval_log_path, "a") as f:
            f.write(json.dumps(report) + "\n")
    except Exception:
        pass

    print(f"Agent report: {report['n_pairs']} pairs -> drift {report['final_drift_pct']}% "
          f"({'PASS' if report['pass_5pct'] else 'FAIL'} vs 5%).")
    return report


def main():
    ap = argparse.ArgumentParser(description="Local ML data-collection agent")
    ap.add_argument("--once", action="store_true", help="single ingest→train→evaluate cycle")
    ap.add_argument("--watch", action="store_true", help="repeat cycle on new data")
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--csv", action="append", default=[])
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--min-pairs", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alt-km", type=float, default=420.0)
    ap.add_argument("--inc-deg", type=float, default=60.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.watch:
        print(f"Watching simulation outputs every {args.interval}s (Ctrl-C to stop).")
        while True:
            try:
                run_once(csv_paths=args.csv, dt=args.dt, min_pairs=args.min_pairs,
                         epochs=args.epochs, hidden=args.hidden, layers=args.layers,
                         lr=args.lr, alt_km=args.alt_km, inc_deg=args.inc_deg,
                         force=args.force)
            except Exception as e:  # noqa: BLE001 — a bad cycle must not kill the watch
                print(f"cycle failed: {e}")
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("Stopped.")
                break
    else:  # default: single cycle
        run_once(csv_paths=args.csv, dt=args.dt, min_pairs=args.min_pairs,
                 epochs=args.epochs, hidden=args.hidden, layers=args.layers,
                 lr=args.lr, alt_km=args.alt_km, inc_deg=args.inc_deg,
                 force=args.force)


if __name__ == "__main__":
    main()
