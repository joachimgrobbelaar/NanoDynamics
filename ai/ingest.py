"""Ingest simulation outputs into validated ML training pairs (one-step).

Sources (auto-discovered, all optional):
  visualization_3d/saved_sims/*.json  (old row-list AND new column-array formats)
  trajectory.json                       (row-list, project root)
  ai/data/sim_log.jsonl                 (appended by POST /simulate, best-effort)
  --csv PATH                            (repeatable; simulator CSV exports)

Each run is validated (finite values, above-surface radius, monotonic time)
and split into uniform-dt segments; consecutive state pairs from segments at
the target dt become training rows. Output is rebuilt deterministically from
all sources every run, with content-hash dedup, so re-running never doubles
data. Physics provenance (atmosphere/density_scale/J2/drag/moon flags) is
recorded per run in the manifest — never silently mix configs (cf. the 30x
density mismatch seen in trajectory_export.csv).

Usage:
    python3 ai/ingest.py [--csv PATH] [--dt 60] [--out ai/data]
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from itertools import pairwise

import numpy as np

AI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(AI_DIR)
for _path in (AI_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from leo_simulator.constants import R_EARTH

BODY_RADII = {"Earth": R_EARTH, "Moon": 1737400.0}
ROW_KEYS = ("t", "x", "y", "z", "vx", "vy", "vz")
DT_TOL = 0.01  # segments must hold dt within 1%


def _hash_run(t, y):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(t).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()[:16]


def _rows_to_arrays(rows):
    t = np.array([r["t"] for r in rows], dtype=np.float64)
    y = np.array([[r[k] for k in ("x", "y", "z", "vx", "vy", "vz")] for r in rows],
                 dtype=np.float64)
    return t, y


def _cols_to_arrays(col):
    t = np.array(col["t"], dtype=np.float64)
    y = np.column_stack([np.array(col[k], dtype=np.float64) for k in ("x", "y", "z", "vx", "vy", "vz")])
    return t, y


def parse_saved_sims(path):
    """Parse a saved_sims file (old and new formats) into run dicts."""
    with open(path) as f:
        data = json.load(f)
    runs = []
    for entry in data:
        params = entry.get("params", {}) or {}
        traj = entry.get("trajectory")
        if isinstance(traj, list):  # old: list of row dicts
            t, y = _rows_to_arrays(traj)
        elif isinstance(traj, dict):  # new: dict of column arrays
            t, y = _cols_to_arrays(traj)
        else:
            continue
        runs.append({"source": os.path.basename(path), "sat": entry.get("name", "?"),
                     "params": params, "t": t, "y": y})
    return runs


def parse_trajectory_json(path):
    with open(path) as f:
        data = json.load(f)
    t, y = _rows_to_arrays(data)
    return [{"source": os.path.basename(path), "sat": "trajectory", "params": {},
             "t": t, "y": y}]


def parse_sim_log(path):
    runs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t, y = _cols_to_arrays(rec["trajectory"])
            runs.append({"source": "sim_log.jsonl", "sat": rec.get("name", "?"),
                         "params": rec.get("params", {}) or {},
                         "t": t, "y": y, "dt_hint": rec.get("dt")})
    return runs


def parse_csv_export(path):
    """Parse a simulator CSV export; splits glued runs on time resets/gaps."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    segs, cur = [], []
    for row in rows:
        try:
            t = float(row["Time(s)"])
            y = [float(row[k]) for k in ("X(m)", "Y(m)", "Z(m)",
                                         "VX(m/s)", "VY(m/s)", "VZ(m/s)")]
        except (KeyError, ValueError, TypeError):
            continue
        if cur and t <= cur[-1][0]:
            segs.append(cur)
            cur = []
        cur.append((t, y))
    if cur:
        segs.append(cur)
    return [{"source": f"{os.path.basename(path)}#seg{i + 1}", "sat": "csv",
             "params": {"note": "csv-export"},
             "t": np.array([p[0] for p in s]), "y": np.array([p[1] for p in s])}
            for i, s in enumerate(segs)]


def _physics_provenance(params):
    """Extract the physics knobs that change dynamics; unknowns stay explicit."""
    keys = ("parent_body", "atmosphere_type", "density_scale",
            "include_j2", "include_drag", "include_moon")
    return {k: params.get(k, "unknown") for k in keys}


def split_uniform_segments(t, y, body_radius):
    """Validate a run; return (segments, problems).

    segments: list of (t_seg, y_seg) with strictly increasing, uniform-dt
    time and all points above the surface. problems: human-readable skips.
    """
    problems = []
    if len(t) < 2:
        return [], ["<2 points"]
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        return [], ["non-finite values"]
    rnorm = np.linalg.norm(y[:, 0:3], axis=1)
    if np.any(rnorm <= body_radius):
        problems.append("sub-surface points dropped")
        mask = rnorm > body_radius
        t, y = t[mask], y[mask]
        if len(t) < 2:
            return [], problems
    dt = np.diff(t)
    # split wherever a step is <= 0 or drifts from the segment's opening dt
    seg_bounds = [0]
    for i, d in enumerate(dt):
        if d <= 0 or abs(d - dt[seg_bounds[-1]]) / max(dt[seg_bounds[-1]], 1e-12) > DT_TOL:
            seg_bounds.append(i + 1)
    seg_bounds.append(len(t))
    segments = [(t[a:b], y[a:b]) for a, b in pairwise(seg_bounds) if b - a >= 2]
    if len(segments) > 1:
        problems.append(f"split into {len(segments)} uniform-dt pieces")
    return segments, problems


def ingest(csv_paths=(), dt_target=60.0, out_dir=None):
    """Collect all sources; return (pairs_x, pairs_y, manifest)."""
    out_dir = out_dir or os.path.join(AI_DIR, "data")
    saved_dir = os.path.join(PROJECT_ROOT, "visualization_3d", "saved_sims")
    runs = []
    for path in sorted(glob.glob(os.path.join(saved_dir, "*.json"))):
        try:
            runs.extend(parse_saved_sims(path))
        except (OSError, ValueError, KeyError) as e:
            print(f"  skip {path}: {e}")
    traj = os.path.join(PROJECT_ROOT, "trajectory.json")
    if os.path.exists(traj):
        try:
            runs.extend(parse_trajectory_json(traj))
        except (OSError, ValueError, KeyError) as e:
            print(f"  skip {traj}: {e}")
    sim_log = os.path.join(out_dir, "sim_log.jsonl")
    if os.path.exists(sim_log):
        try:
            runs.extend(parse_sim_log(sim_log))
        except (OSError, ValueError, KeyError) as e:
            print(f"  skip {sim_log}: {e}")
    for cp in csv_paths:
        try:
            runs.extend(parse_csv_export(cp))
        except (OSError, ValueError) as e:
            print(f"  skip {cp}: {e}")

    seen, xs, ys, manifest_runs, skipped = set(), [], [], [], 0
    pair_offset = 0
    for run in runs:
        params = run.get("params", {}) or {}
        radius = BODY_RADII.get(params.get("parent_body", "Earth"), R_EARTH)
        segments, problems = split_uniform_segments(run["t"], run["y"], radius)
        if not segments:
            skipped += 1
            manifest_runs.append({"source": run["source"], "sat": run["sat"],
                                  "status": "skipped", "reason": "; ".join(problems)})
            continue
        rh = _hash_run(run["t"], run["y"])
        if rh in seen:
            manifest_runs.append({"source": run["source"], "sat": run["sat"],
                                  "status": "duplicate"})
            continue
        seen.add(rh)
        n_pairs, seg_dts = 0, set()
        for st, sy in segments:
            seg_dt = float(np.median(np.diff(st)))
            seg_dts.add(round(seg_dt, 3))
            if abs(seg_dt - dt_target) / dt_target > DT_TOL:
                continue
            xs.append(sy[:-1])
            ys.append(sy[1:])
            n_pairs += len(sy) - 1
        manifest_runs.append({"source": run["source"], "sat": run["sat"],
                              "status": "ok", "hash": rh,
                              "n_pts": len(run["t"]), "n_pairs": n_pairs,
                              "pair_start": pair_offset,
                              "segment_dts": sorted(seg_dts),
                              "physics": _physics_provenance(params),
                              "notes": "; ".join(problems)})
        pair_offset += n_pairs
        if n_pairs == 0:
            skipped += 1

    manifest = {"dt_target": dt_target, "n_runs": len(runs),
                "n_pairs": sum(m.get("n_pairs", 0) for m in manifest_runs),
                "n_skipped": skipped, "runs": manifest_runs}
    if out_dir is not None and xs:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "sim_pairs.npz"),
                            x=np.vstack(xs), y=np.vstack(ys))
        with open(os.path.join(out_dir, "sim_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    configs = {json.dumps(m["physics"], sort_keys=True)
               for m in manifest_runs if m.get("physics")}
    if len(configs) > 1:
        print("  NOTE: runs span multiple physics configs — "
              "check sim_manifest.json before joint training:")
        for c in sorted(configs):
            print(f"    {c}")
    print(f"Ingested {manifest['n_pairs']} pairs @dt={dt_target}s "
          f"from {len(runs)} runs ({skipped} skipped/empty).")
    return (np.vstack(xs) if xs else np.zeros((0, 6)),
            np.vstack(ys) if ys else np.zeros((0, 6)), manifest)


def load_earth_pairs(data_dir):
    """Load sim pairs sliced to mu-compatible runs (Earth or unknown legacy).

    Explicit non-Earth bodies (e.g. lunar orbits, different mu) are excluded,
    never silently mixed into Earth training. Returns (x, y, info).
    """
    blob = np.load(os.path.join(data_dir, "sim_pairs.npz"))
    with open(os.path.join(data_dir, "sim_manifest.json")) as f:
        manifest = json.load(f)
    xs, ys, dropped = [], [], []
    for r in manifest["runs"]:
        if r.get("status") != "ok" or r.get("n_pairs", 0) == 0:
            continue
        body = (r.get("physics") or {}).get("parent_body", "unknown")
        s = slice(r["pair_start"], r["pair_start"] + r["n_pairs"])
        if body in ("Earth", "unknown"):
            xs.append(blob["x"][s])
            ys.append(blob["y"][s])
        else:
            dropped.append(f"{r['source']}:{r['sat']} ({body})")
    info = {"n_kept": sum(len(a) for a in xs), "dropped": dropped}
    if not xs:
        return np.zeros((0, 6)), np.zeros((0, 6)), info
    return np.vstack(xs), np.vstack(ys), info


def main():
    ap = argparse.ArgumentParser(description="Ingest simulation outputs for ML")
    ap.add_argument("--csv", action="append", default=[],
                    help="CSV export path (repeatable)")
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--out", default=os.path.join(AI_DIR, "data"))
    args = ap.parse_args()
    ingest(csv_paths=args.csv, dt_target=args.dt, out_dir=args.out)


if __name__ == "__main__":
    main()
