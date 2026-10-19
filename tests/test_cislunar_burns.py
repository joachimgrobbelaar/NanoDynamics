import pytest
import numpy as np
from fastapi.testclient import TestClient
from visualization_3d.app import app

client = TestClient(app)

def test_cislunar_tli_burn_preview():
    """Verify preview endpoint handles large Trans-Lunar Injection (TLI) burn."""
    # 1. Simulate initial LEO 400 km orbit
    init_payload = {
        "name": "TLI-Sat",
        "parent_body": "Earth",
        "mass": 500.0,
        "drag_area": 2.0,
        "cd": 2.2,
        "altitude_km": 400.0,
        "eccentricity": 0.001,
        "inclination_deg": 28.5,
        "raan_deg": 0.0,
        "atmosphere_type": "piecewise",
        "density_scale": 1.0,
        "include_j2": True,
        "include_drag": False,
        "include_moon": True
    }
    sim_res = client.post("/simulate", json=init_payload)
    assert sim_res.status_code == 200, sim_res.text
    sim_data = sim_res.json()
    last_state = sim_data["last_state"]
    last_t = sim_data["last_t"]

    # 2. Preview TLI burn (+3100 m/s prograde)
    preview_payload = {
        "params": init_payload,
        "t_burn": last_t,
        "current_state": last_state,
        "dv_prograde": 3100.0,
        "dv_radial": 0.0,
        "dv_normal": 0.0
    }
    prev_res = client.post("/burn/preview", json=preview_payload)
    assert prev_res.status_code == 200, prev_res.text
    prev_data = prev_res.json()
    assert "trajectory" in prev_data
    traj = prev_data["trajectory"]
    assert len(traj["t"]) > 10
    assert "lat_deg" in traj
    assert "lon_deg" in traj

    # Check that maximum altitude reaches cislunar apogee (> 300,000 km)
    max_alt = max(traj["alt_km"])
    assert max_alt > 300000.0, f"Expected cislunar apogee > 300,000 km, got {max_alt} km"

def test_cislunar_tli_burn_execution():
    """Verify /burn endpoint executes large TLI burn and returns trajectory chunk with coordinates."""
    init_payload = {
        "name": "TLI-ExecSat",
        "parent_body": "Earth",
        "mass": 500.0,
        "drag_area": 2.0,
        "cd": 2.2,
        "altitude_km": 400.0,
        "eccentricity": 0.001,
        "inclination_deg": 28.5,
        "raan_deg": 0.0,
        "atmosphere_type": "piecewise",
        "density_scale": 1.0,
        "include_j2": True,
        "include_drag": False,
        "include_moon": True
    }
    sim_res = client.post("/simulate", json=init_payload)
    assert sim_res.status_code == 200, sim_res.text
    sim_data = sim_res.json()
    last_state = sim_data["last_state"]
    last_t = sim_data["last_t"]

    burn_payload = {
        "params": init_payload,
        "t_burn": last_t,
        "current_state": last_state,
        "dv_prograde": 3100.0,
        "dv_radial": 0.0,
        "dv_normal": 0.0
    }
    burn_res = client.post("/burn", json=burn_payload)
    assert burn_res.status_code == 200, burn_res.text
    burn_data = burn_res.json()
    assert "trajectory" in burn_data
    traj = burn_data["trajectory"]
    assert len(traj["t"]) > 10
    assert "lat_deg" in traj
    assert "lon_deg" in traj
    assert "events" in burn_data
    assert traj["t"][0] >= last_t

def test_lunar_orbit_burn():
    """Verify burn maneuver execution on lunar orbit."""
    init_payload = {
        "name": "Lunar-BurnSat",
        "parent_body": "Moon",
        "mass": 500.0,
        "drag_area": 2.0,
        "cd": 2.2,
        "altitude_km": 100.0,
        "eccentricity": 0.001,
        "inclination_deg": 90.0,
        "raan_deg": 0.0,
        "atmosphere_type": "piecewise",
        "density_scale": 1.0,
        "include_j2": False,
        "include_drag": False,
        "include_moon": False
    }
    sim_res = client.post("/simulate", json=init_payload)
    assert sim_res.status_code == 200, sim_res.text
    sim_data = sim_res.json()
    last_state = sim_data["last_state"]
    last_t = sim_data["last_t"]

    burn_payload = {
        "params": init_payload,
        "t_burn": last_t,
        "current_state": last_state,
        "dv_prograde": 50.0,
        "dv_radial": 0.0,
        "dv_normal": 0.0
    }
    burn_res = client.post("/burn", json=burn_payload)
    assert burn_res.status_code == 200, burn_res.text
    burn_data = burn_res.json()
    assert "trajectory" in burn_data
    assert len(burn_data["trajectory"]["t"]) > 0
