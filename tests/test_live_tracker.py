"""Unit and integration tests for real-time LEO Satellite Tracker and API endpoints."""

import pytest
from fastapi.testclient import TestClient
import numpy as np

from leo_simulator.tle_tracker import (
    GROUPS,
    eccentric_to_true_anomaly,
    get_live_catalog,
    get_satellite_by_norad,
    parse_omm_record,
    solve_kepler,
)
from visualization_3d.app import app


def test_solve_kepler_and_anomaly():
    """Verify Kepler equation solver and eccentric-to-true anomaly conversion."""
    M = 0.5  # rad
    e = 0.001
    E = solve_kepler(M, e)
    # Check Kepler equation satisfaction: E - e*sin(E) == M
    assert abs(E - e * np.sin(E) - M) < 1e-9

    nu = eccentric_to_true_anomaly(E, e)
    assert 0.0 <= nu <= 2.0 * np.pi


def test_parse_omm_record_iss():
    """Verify OMM parsing of ISS record."""
    sample_iss = {
        "OBJECT_NAME": "ISS (ZARYA)",
        "NORAD_CAT_ID": 25544,
        "MEAN_MOTION": 15.50,
        "ECCENTRICITY": 0.0005,
        "INCLINATION": 51.64,
        "RA_OF_ASC_NODE": 120.0,
        "ARG_OF_PERICENTER": 80.0,
        "MEAN_ANOMALY": 280.0,
        "EPOCH": "2026-10-08T00:00:00.000000",
    }
    parsed = parse_omm_record(sample_iss, group="stations")
    assert parsed is not None
    assert parsed["norad_id"] == 25544
    assert 400.0 <= parsed["altitude_km"] <= 430.0
    assert abs(parsed["inclination_deg"] - 51.64) < 1e-3
    assert parsed["mass"] == 450000.0


def test_live_catalog_and_lookup():
    """Verify loading from cache and NORAD lookup."""
    catalog = get_live_catalog(group="stations")
    assert len(catalog) >= 1
    iss = get_satellite_by_norad(25544)
    assert iss is not None
    assert "ISS" in iss["name"]


def test_api_live_endpoints():
    """Verify FastAPI live tracking endpoints."""
    client = TestClient(app)

    # 1. Groups endpoint
    res_groups = client.get("/satellites/live/groups")
    assert res_groups.status_code == 200
    groups_data = res_groups.json()["groups"]
    assert "stations" in groups_data
    assert "cubesat" in groups_data

    # 2. Catalog endpoint
    res_cat = client.get("/satellites/live/catalog?group=stations")
    assert res_cat.status_code == 200
    assert res_cat.json()["count"] >= 1

    # 3. Track endpoint (Numerical RK45)
    res_track = client.post("/satellites/live/track", json={"norad_id": 25544, "propagation_mode": "rk45"})
    assert res_track.status_code == 200
    track_data = res_track.json()
    assert "ISS" in track_data["name"]
    assert track_data["engine"] == "rk45"
    assert len(track_data["trajectory"]["t"]) > 10

    # 4. Track endpoint (PINN Surrogate)
    cubesats = client.get("/satellites/live/catalog?group=cubesat").json()["satellites"]
    cubesat_norad = cubesats[0]["norad_id"] if cubesats else 25544
    res_pinn = client.post("/satellites/live/track", json={"norad_id": cubesat_norad, "propagation_mode": "pinn"})
    assert res_pinn.status_code == 200
    assert res_pinn.json()["engine"] == "pinn"
