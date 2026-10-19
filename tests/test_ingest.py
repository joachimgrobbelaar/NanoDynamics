"""Tests for ai/ingest.py: formats, validation, dedup (fast, no torch)."""

import json
import os

import numpy as np

from ai.ingest import load_earth_pairs, parse_csv_export, split_uniform_segments


def _row(t, r=6778000.0, v=7700.0):
    return {"t": t, "x": r, "y": 0.0, "z": 0.0,
            "vx": 0.0, "vy": v, "vz": 0.0}


def test_old_row_list_format(tmp_path, monkeypatch):
    import ai.ingest as ing
    sims = tmp_path / "visualization_3d" / "saved_sims"
    sims.mkdir(parents=True)
    with open(sims / "a.json", "w") as f:
        json.dump([{"name": "S1", "params": {"parent_body": "Earth"},
                    "trajectory": [_row(t) for t in (0.0, 60.0, 120.0, 180.0)],
                    "color": 1}], f)
    monkeypatch.setattr(ing, "PROJECT_ROOT", str(tmp_path))
    x, y, m = ing.ingest(csv_paths=(), dt_target=60.0,
                         out_dir=str(tmp_path / "out"))
    assert m["n_pairs"] == 3
    assert x.shape == (3, 6) and y.shape == (3, 6)
    # consecutive pairs share the boundary state
    np.testing.assert_array_equal(x[1], y[0])


def test_uniform_segmentation_and_dt_filter():
    t = np.array([0.0, 60.0, 120.0, 180.0])
    y = np.array([[6778000.0, 0, 0, 0, 7700.0, 0]] * 4, dtype=float)
    segs, problems = split_uniform_segments(t, y, 6378137.0)
    assert len(segs) == 1 and not problems

    # gap splits into two segments
    t2 = np.array([0.0, 60.0, 120.0, 5000.0, 5060.0])
    y2 = np.array([[6778000.0, 0, 0, 0, 7700.0, 0]] * 5, dtype=float)
    segs2, problems2 = split_uniform_segments(t2, y2, 6378137.0)
    assert len(segs2) == 2
    assert any("split" in p for p in problems2)

    # sub-surface points are dropped, not fatal
    y3 = y.copy()
    y3[0, 0] = 1000.0
    segs3, problems3 = split_uniform_segments(t, y3, 6378137.0)
    assert len(segs3) == 1 and any("sub-surface" in p for p in problems3)

    # non-finite run rejected
    y4 = y.copy()
    y4[1, 1] = float("nan")
    segs4, _ = split_uniform_segments(t, y4, 6378137.0)
    assert segs4 == []


def test_csv_export_splits_glued_runs(tmp_path):
    p = tmp_path / "exp.csv"
    with open(p, "w", newline="") as f:
        f.write("Satellite,Time(s),X(m),Y(m),Z(m),VX(m/s),VY(m/s),VZ(m/s),Alt(km)\n")
        f.writelines(f"S1,{t},6778000,0,0,0,7700,0,400\n" for t in (0.0, 60.0, 120.0, 0.0, 60.0, 120.0))
    runs = parse_csv_export(str(p))
    assert len(runs) == 2
    assert all(len(r["t"]) == 3 for r in runs)


def test_ingest_end_to_end_new_column_format(tmp_path, monkeypatch):
    import ai.ingest as ing
    sims = tmp_path / "visualization_3d" / "saved_sims"
    sims.mkdir(parents=True)
    col = {"t": [0.0, 60.0, 120.0], "x": [6778000.0] * 3, "y": [0.0] * 3,
           "z": [0.0] * 3, "vx": [0.0] * 3, "vy": [7700.0] * 3, "vz": [0.0] * 3}
    with open(sims / "n.json", "w") as f:
        json.dump([{"name": "N1",
                    "params": {"parent_body": "Earth", "density_scale": 1,
                               "include_j2": True, "include_drag": True,
                               "include_moon": False},
                    "trajectory": col}], f)
    monkeypatch.setattr(ing, "PROJECT_ROOT", str(tmp_path))
    out = tmp_path / "out"
    x, y, m = ing.ingest(csv_paths=(), dt_target=60.0, out_dir=str(out))
    assert m["n_pairs"] == 2
    assert x.shape == (2, 6) and y.shape == (2, 6)
    assert os.path.exists(out / "sim_pairs.npz")
    assert os.path.exists(out / "sim_manifest.json")

    # same content in a second source file -> duplicate, pairs counted once
    with open(sims / "n2.json", "w") as f:
        json.dump([{"name": "N1-copy", "params": {"parent_body": "Earth"},
                    "trajectory": col}], f)
    _x2, _y2, m2 = ing.ingest(csv_paths=(), dt_target=60.0,
                               out_dir=str(tmp_path / "out2"))
    assert m2["n_pairs"] == 2
    assert sum(1 for r in m2["runs"] if r["status"] == "duplicate") == 1


def test_moon_runs_excluded_from_earth_training(tmp_path, monkeypatch):
    """Lunar (different-mu) pairs must never mix into Earth training data."""
    import ai.ingest as ing
    sims = tmp_path / "visualization_3d" / "saved_sims"
    sims.mkdir(parents=True)
    col = {"t": [0.0, 60.0, 120.0], "x": [6778000.0] * 3, "y": [0.0] * 3,
           "z": [0.0] * 3, "vx": [0.0] * 3, "vy": [7700.0] * 3, "vz": [0.0] * 3}
    moon_col = dict(col, x=[2778000.0, 2779000.0, 2780000.0])  # distinct content
    with open(sims / "mix.json", "w") as f:
        json.dump([{"name": "EarthSat", "params": {"parent_body": "Earth"},
                    "trajectory": col},
                   {"name": "MoonSat", "params": {"parent_body": "Moon"},
                    "trajectory": moon_col}], f)
    monkeypatch.setattr(ing, "PROJECT_ROOT", str(tmp_path))
    out = str(tmp_path / "out")
    _, _, m = ing.ingest(csv_paths=(), dt_target=60.0, out_dir=out)
    assert m["n_pairs"] == 4  # everything ingested ...
    # ... but offsets let training select Earth-only pairs
    starts = [r["pair_start"] for r in m["runs"] if r["status"] == "ok"]
    assert starts == [0, 2]
    x, y, info = load_earth_pairs(out)
    assert info["n_kept"] == 2
    assert len(info["dropped"]) == 1 and "Moon" in info["dropped"][0]
    assert x.shape == (2, 6) and y.shape == (2, 6)
