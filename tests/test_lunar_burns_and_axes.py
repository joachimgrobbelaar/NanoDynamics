"""Tests for Lunar Burns, AI Surrogate Endpoints, and Coordinate Systems."""

import pytest
from fastapi.testclient import TestClient
import numpy as np

from visualization_3d.app import app, BODIES
from leo_simulator.models.gravity import MU_MOON
from leo_simulator.coordinates import eci_to_geodetic, extract_trajectory_key_events


@pytest.fixture
def client():
    return TestClient(app)


def test_lunar_burn_preview_and_execute(client):
    """Verify that maneuver burns on Moon satellites propagate using Moon gravity."""
    moon_params = {
        "name": "Artemis-Sat",
        "parent_body": "Moon",
        "mass": 500.0,
        "drag_area": 2.0,
        "cd": 2.0,
        "altitude_km": 100.0,
        "eccentricity": 0.001,
        "inclination_deg": 90.0,
        "raan_deg": 0.0,
        "atmosphere_type": "piecewise",
        "density_scale": 1.0,
        "include_j2": True,
        "include_drag": False,
        "include_moon": False,
        "color": "#00ffaa",
        "icon": "satellite",
        "t_start": 0.0,
    }

    sim_res = client.post("/simulate", json=moon_params)
    assert sim_res.status_code == 200
    sim_data = sim_res.json()
    assert sim_data["parent_body"] == "Moon"
    assert len(sim_data["trajectory"]["x"]) > 10

    # Initial state relative to Moon center
    r0 = [sim_data["trajectory"]["x"][0], sim_data["trajectory"]["y"][0], sim_data["trajectory"]["z"][0]]
    v0 = [sim_data["trajectory"]["vx"][0], sim_data["trajectory"]["vy"][0], sim_data["trajectory"]["vz"][0]]
    initial_state = r0 + v0

    # Preview a prograde burn (+50 m/s)
    burn_payload = {
        "params": moon_params,
        "t_burn": 60.0,
        "current_state": initial_state,
        "dv_prograde": 50.0,
        "dv_radial": 0.0,
        "dv_normal": 0.0,
    }

    prev_res = client.post("/burn/preview", json=burn_payload)
    assert prev_res.status_code == 200
    prev_data = prev_res.json()
    assert prev_data["status"] == "success"
    assert "trajectory" in prev_data
    assert len(prev_data["trajectory"]["x"]) > 10

    # Execute burn
    exec_res = client.post("/burn", json=burn_payload)
    assert exec_res.status_code == 200
    exec_data = exec_res.json()
    assert exec_data["status"] == "success"
    assert len(exec_data["trajectory"]["alt_km"]) > 10
    # Altitude should remain above lunar surface (> 20 km)
    assert min(exec_data["trajectory"]["alt_km"]) > 20.0


def test_ai_status_and_train_endpoints(client):
    """Verify /ai/status and /ai/train endpoints integration."""
    status_res = client.get("/ai/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert "status" in status_data

    # Trigger a 10-epoch smoke training run
    train_res = client.post("/ai/train", json={"epochs": 10, "hidden": 32, "layers": 2, "force": True})
    assert train_res.status_code == 200
    train_data = train_res.json()
    assert train_data["status"] == "success"
    assert "result" in train_data
    assert train_data["result"]["status"] in ("trained", "up-to-date", "waiting")


def test_selenographic_geodetic_coordinates():
    """Verify coordinate transformation on Moon body radius."""
    moon_radius = BODIES["Moon"]["radius_m"]
    moon_omega = BODIES["Moon"]["omega"]

    # Point on prime meridian equator 100 km above lunar surface
    r_eci = np.array([moon_radius + 100000.0, 0.0, 0.0])
    lat, lon, alt = eci_to_geodetic(r_eci, t_seconds=0.0, radius_m=moon_radius, omega_earth=moon_omega)
    assert np.isclose(lat, 0.0, atol=1e-5)
    assert np.isclose(lon, 0.0, atol=1e-5)
    assert np.isclose(alt, 100.0, atol=1e-3)
