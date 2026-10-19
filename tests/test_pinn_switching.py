"""Acceptance tests for switching between classical numerical (RK45) and ML PINN propagation."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from leo_simulator.constants import R_EARTH
from leo_simulator.orbit.elements import OrbitalElements
from leo_simulator.orbit.pinn_propagator import PINNPropagator
from visualization_3d.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_pinn_propagator_direct():
    """Verify that PINNPropagator rolls out states without numerical ODE solver."""
    prop = PINNPropagator()
    oe = OrbitalElements(
        a=R_EARTH + 400_000.0,
        e=0.001,
        i=np.radians(51.6),
        raan=np.radians(45.0),
        arg_pe=0.0,
        nu=0.0,
    )
    res = prop.propagate(oe, duration_seconds=3600.0, dt_eval=60.0)
    assert res.success is True
    assert len(res.t) == 61
    assert len(res.r) == 61
    assert len(res.v) == 61
    assert np.all(np.isfinite(res.r))
    assert np.all(np.isfinite(res.v))
    # Altitudes should be in LEO range (~400 km)
    assert 300.0 < (np.linalg.norm(res.r[0]) - R_EARTH) / 1000.0 < 500.0


def test_api_rk45_mode(client):
    """Verify default and explicit rk45 propagation mode."""
    payload = {
        "name": "Sat-RK45-Test",
        "parent_body": "Earth",
        "propagation_mode": "rk45",
        "mass": 4.0,
        "drag_area": 0.03,
        "cd": 2.2,
        "altitude_km": 400.0,
        "eccentricity": 0.001,
        "inclination_deg": 51.6,
        "raan_deg": 45.0,
    }
    res = client.post("/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["engine"] == "rk45"
    assert "trajectory" in data
    assert len(data["trajectory"]["t"]) > 0


def test_api_pinn_mode(client):
    """Verify ML PINN surrogate propagation mode via /simulate endpoint."""
    payload = {
        "name": "Sat-PINN-Test",
        "parent_body": "Earth",
        "propagation_mode": "pinn",
        "mass": 4.0,
        "drag_area": 0.03,
        "cd": 2.2,
        "altitude_km": 400.0,
        "eccentricity": 0.001,
        "inclination_deg": 51.6,
        "raan_deg": 45.0,
    }
    res = client.post("/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["engine"] == "pinn"
    assert "trajectory" in data
    assert len(data["trajectory"]["t"]) > 0
    assert len(data["last_state"]) == 6


def test_api_pinn_rejection_on_moon(client):
    """Verify PINN mode returns 422 if configured for non-Earth parent body."""
    payload = {
        "name": "Sat-Moon-PINN",
        "parent_body": "Moon",
        "propagation_mode": "pinn",
        "mass": 4.0,
        "drag_area": 0.03,
        "cd": 2.2,
        "altitude_km": 100.0,
        "eccentricity": 0.001,
        "inclination_deg": 90.0,
        "raan_deg": 0.0,
    }
    res = client.post("/simulate", json=payload)
    assert res.status_code in (400, 422)
    assert "PINN surrogate model currently only supports Earth" in str(res.json())


def test_stream_continuation_pinn(client):
    """Verify /stream endpoint successfully propagates rolling chunks in PINN mode."""
    params = {
        "name": "Sat-Stream-PINN",
        "parent_body": "Earth",
        "propagation_mode": "pinn",
        "mass": 4.0,
        "drag_area": 0.03,
        "cd": 2.2,
        "altitude_km": 400.0,
        "eccentricity": 0.001,
        "inclination_deg": 51.6,
        "raan_deg": 45.0,
    }
    sim_res = client.post("/simulate", json=params)
    assert sim_res.status_code == 200
    sim_data = sim_res.json()

    stream_payload = {
        "params": params,
        "t_start": sim_data["last_t"],
        "state": sim_data["last_state"],
        "chunk_duration": 1800.0,
        "dt_eval": 60.0,
    }
    stream_res = client.post("/stream", json=stream_payload)
    assert stream_res.status_code == 200
    stream_data = stream_res.json()
    assert stream_data["engine"] == "pinn"
    assert len(stream_data["trajectory"]["t"]) > 0


def test_pinn_long_term_stability():
    """Verify that PINN sustained rollout over 24 hours does not prematurely deorbit."""
    prop = PINNPropagator()
    oe = OrbitalElements(
        a=R_EARTH + 400_000.0,
        e=0.001,
        i=np.radians(51.6),
        raan=np.radians(45.0),
        arg_pe=0.0,
        nu=0.0,
    )
    # Propagate for a full 24 hours (1440 steps at 60s)
    res = prop.propagate(oe, duration_seconds=86_400.0, dt_eval=60.0)
    assert res.reentry_detected is False
    assert len(res.t) == 1441
    # Check altitude remains stably in LEO across all 1440 steps
    alts = (np.linalg.norm(res.r, axis=1) - R_EARTH) / 1000.0
    assert np.all(alts > 200.0)  # No premature deorbit
    assert np.all(alts < 600.0)  # No divergent ejection

