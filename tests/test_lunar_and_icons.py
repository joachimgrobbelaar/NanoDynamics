import pytest
import numpy as np
from fastapi.testclient import TestClient
from visualization_3d.app import app

client = TestClient(app)

def test_moon_satellite_propagation_and_icons():
    """Verify satellite orbiting Moon propagates with custom color and icon avatar metadata."""
    payload = {
        "name": "Lunar-Apollo",
        "parent_body": "Moon",
        "mass": 1000.0,
        "drag_area": 5.0,
        "cd": 2.2,
        "altitude_km": 100.0,
        "eccentricity": 0.005,
        "inclination_deg": 90.0,
        "raan_deg": 0.0,
        "atmosphere_type": "piecewise",
        "density_scale": 1.0,
        "include_j2": False,
        "include_drag": False,
        "include_moon": False,
        "color": "#10b981",
        "icon": "astronaut"
    }
    res = client.post("/simulate", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["parent_body"] == "Moon"
    assert data["params"]["color"] == "#10b981"
    assert data["params"]["icon"] == "astronaut"
    
    # Lunar period for 100 km orbit around Moon (R_moon=1737.4 km, mu_moon=4.9048695e12)
    # T = 2*pi*sqrt(1837400^3 / 4.9048695e12) ~ 7074 s
    assert 6900 < data["period_s"] < 7300

    traj = data["trajectory"]
    assert len(traj["t"]) > 10
    # Moon orbit radii are ~1837 km (perigee ~90.8 km with e=0.005)
    assert 85 < traj["alt_km"][0] < 105

def test_alien_satellite_on_earth():
    """Verify Earth-orbiting satellite with alien icon and custom color."""
    payload = {
        "name": "Alien-UFO",
        "parent_body": "Earth",
        "mass": 42.0,
        "drag_area": 1.0,
        "cd": 2.2,
        "altitude_km": 500.0,
        "eccentricity": 0.01,
        "inclination_deg": 45.0,
        "raan_deg": 120.0,
        "atmosphere_type": "piecewise",
        "density_scale": 1.0,
        "include_j2": True,
        "include_drag": True,
        "include_moon": True,
        "color": "#a855f7",
        "icon": "alien"
    }
    res = client.post("/simulate", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["params"]["color"] == "#a855f7"
    assert data["params"]["icon"] == "alien"

def test_moon_satellite_streaming():
    """Verify /stream endpoint continues propagation of a Moon satellite."""
    init_payload = {
        "name": "Lunar-StreamSat",
        "parent_body": "Moon",
        "mass": 500.0,
        "drag_area": 2.0,
        "cd": 2.2,
        "altitude_km": 200.0,
        "eccentricity": 0.001,
        "inclination_deg": 60.0,
        "raan_deg": 30.0,
        "atmosphere_type": "piecewise",
        "density_scale": 1.0,
        "include_j2": False,
        "include_drag": False,
        "include_moon": False,
        "color": "#f59e0b",
        "icon": "rocket"
    }
    sim_res = client.post("/simulate", json=init_payload)
    assert sim_res.status_code == 200
    sim_data = sim_res.json()

    stream_payload = {
        "params": init_payload,
        "t_start": sim_data["last_t"],
        "state": sim_data["last_state"]
    }
    stream_res = client.post("/stream", json=stream_payload)
    assert stream_res.status_code == 200, stream_res.text
    stream_data = stream_res.json()
    assert "trajectory" in stream_data
    assert stream_data["last_t"] > sim_data["last_t"]
