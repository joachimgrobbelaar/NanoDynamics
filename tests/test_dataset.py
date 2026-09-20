"""Tests for the ML dataset-collection pipeline (fast, CPU-only)."""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai.generate_dataset import generate
from ai.train import normalize


def test_generate_small_dataset_deterministic(tmp_path):
    kwargs = dict(n_traj=2, orbits=0.2, dt=300.0, alt_min_km=400.0,
                  alt_max_km=410.0, seed=7, out_dir=str(tmp_path))
    data1, stats1 = generate(**kwargs)
    data2, stats2 = generate(**kwargs)

    for key in ("t", "init_state", "state"):
        assert data1[key].shape == data2[key].shape
        np.testing.assert_array_equal(data1[key], data2[key])
    assert stats1["physics"] == "central+J2+drag"
    assert os.path.exists(tmp_path / "dataset.npz")
    assert json.loads((tmp_path / "stats.json").read_text())["seed"] == 7


def test_generate_one_step_mode(tmp_path):
    data, stats = generate(n_traj=2, orbits=0.2, dt=300.0, seed=3,
                           out_dir=str(tmp_path), mode="one_step")
    assert stats["mode"] == "one_step"
    assert data["x"].shape == data["y"].shape
    assert data["x"].shape[1] == 6
    assert data["x"].shape[0] > 0
    # consecutive pairs: y[i] is the state after x[i]
    assert np.all(np.isfinite(data["x"])) and np.all(np.isfinite(data["y"]))
    assert os.path.exists(tmp_path / "dataset_onestep.npz")
    with pytest.raises(ValueError):
        generate(n_traj=1, orbits=0.1, dt=300.0, seed=0, out_dir=None,
                 mode="bogus")


def test_dataset_shapes_and_scales(tmp_path):
    data, stats = generate(n_traj=2, orbits=0.2, dt=300.0, seed=1,
                           out_dir=str(tmp_path))
    n = sum(1 for _ in range(len(data["t"])))
    assert data["t"].shape == (n, 1)
    assert data["init_state"].shape == (n, 6)
    assert data["state"].shape == (n, 6)
    assert np.all(np.isfinite(data["state"]))

    import torch
    t, init, state = (torch.as_tensor(x, dtype=torch.float32)
                      for x in normalize(
                          torch.tensor(data["t"]), torch.tensor(data["init_state"]),
                          torch.tensor(data["state"]), stats))
    # normalized magnitudes must be O(1) for stable training
    assert t.abs().max() < 5.0
    assert init.abs().max() < 5.0
    assert state.abs().max() < 5.0
