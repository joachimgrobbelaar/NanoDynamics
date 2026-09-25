"""Unit and integration tests for orbital collision mechanics and conjunction detection."""

import numpy as np
import pytest

from leo_simulator.orbit.collision import (
    compute_orbital_collision_impulse,
    detect_conjunction,
)


def test_detect_conjunction_approaching():
    """Verify conjunction detection triggers only when closing within threshold."""
    r1 = np.array([7000e3, 0.0, 0.0])
    r2 = np.array([7000e3 + 200.0, 0.0, 0.0])  # 200m separation
    v1 = np.array([0.0, 7500.0, 0.0])
    v2 = np.array([0.0, 7500.0, 0.0])

    # Case 1: approaching along x
    v1_app = np.array([10.0, 7500.0, 0.0])
    v2_app = np.array([-10.0, 7500.0, 0.0])
    is_col, dist, rr = detect_conjunction(r1, r2, v1_app, v2_app, threshold_m=500.0)
    assert is_col is True
    assert pytest.approx(dist, abs=1e-3) == 200.0
    assert rr < 0.0  # Approaching

    # Case 2: receding along x
    v1_rec = np.array([-10.0, 7500.0, 0.0])
    v2_rec = np.array([10.0, 7500.0, 0.0])
    is_col_rec, dist_rec, rr_rec = detect_conjunction(r1, r2, v1_rec, v2_rec, threshold_m=500.0)
    assert is_col_rec is False  # Receding, no collision trigger
    assert rr_rec > 0.0

    # Case 3: outside threshold
    r2_far = np.array([7000e3 + 2000.0, 0.0, 0.0])  # 2km separation
    is_col_far, dist_far, _ = detect_conjunction(r1, r2_far, v1_app, v2_app, threshold_m=500.0)
    assert is_col_far is False


def test_momentum_conservation_head_on():
    """Verify strict linear momentum conservation in a head-on orbital collision."""
    m1 = 12.0  # kg (e.g. 6U CubeSat)
    m2 = 4.0   # kg (e.g. 3U CubeSat)
    r1 = np.array([6800e3, -5.0, 0.0])
    r2 = np.array([6800e3, 5.0, 0.0])
    v1 = np.array([0.0, 7600.0, 0.0])
    v2 = np.array([0.0, -7600.0, 0.0])  # Head-on retrograde encounter ~15.2 km/s rel vel

    res = compute_orbital_collision_impulse(m1, r1, v1, m2, r2, v2, restitution=0.3)

    assert res.momentum_error < 1e-10
    # Mass ratio: satellite 2 should experience 3x the delta-V of satellite 1
    assert pytest.approx(res.dv2_mag / res.dv1_mag, rel=1e-5) == 3.0
    assert res.impulse_mag > 0.0


def test_oblique_collision_impulse():
    """Verify 3D oblique conjunction impulse calculation and momentum balance."""
    m1 = 100.0
    m2 = 50.0
    r1 = np.array([6800e3, 100.0, 50.0])
    r2 = np.array([6800e3 + 30.0, 20.0, 10.0])
    v1 = np.array([50.0, 7500.0, -100.0])
    v2 = np.array([-200.0, 7400.0, 300.0])

    res = compute_orbital_collision_impulse(m1, r1, v1, m2, r2, v2, restitution=0.5)

    p_initial = m1 * v1 + m2 * v2
    p_final = m1 * res.v1_post + m2 * res.v2_post
    np.testing.assert_allclose(p_initial, p_final, atol=1e-9)


def test_invalid_satellite_mass():
    """Verify ValueError is raised if masses are zero or negative."""
    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, 7500.0, 0.0])
    with pytest.raises(ValueError, match="strictly positive"):
        compute_orbital_collision_impulse(0.0, r, v, 10.0, r, v)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_orbital_collision_impulse(10.0, r, v, -5.0, r, v)


def test_api_collision_resolve():
    """Verify /collision/resolve endpoint correctly computes impulse and re-propagates."""
    from fastapi.testclient import TestClient
    from visualization_3d.app import app

    client = TestClient(app)
    payload = {
        "sat1_params": {
            "name": "Sat-Col-1",
            "parent_body": "Earth",
            "propagation_mode": "rk45",
            "mass": 10.0,
            "drag_area": 0.05,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 0.0,
        },
        "sat1_state": [6778137.0, -10.0, 0.0, 0.0, 7670.0, 0.0],
        "sat2_params": {
            "name": "Sat-Col-2",
            "parent_body": "Earth",
            "propagation_mode": "pinn",
            "mass": 5.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 0.0,
        },
        "sat2_state": [6778137.0, 10.0, 0.0, 0.0, -7670.0, 0.0],
        "t_collision": 1200.0,
        "restitution": 0.25,
        "post_collision_duration": 1800.0,
        "dt_eval": 60.0,
    }

    res = client.post("/collision/resolve", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["t_collision"] == 1200.0
    assert data["impulse_ns"] > 0.0
    assert data["sat1"]["dv_mag_m_s"] > 0.0
    assert data["sat2"]["dv_mag_m_s"] > 0.0
    # Sat 2 has half the mass of Sat 1, so its dv must be double
    assert pytest.approx(data["sat2"]["dv_mag_m_s"] / data["sat1"]["dv_mag_m_s"], rel=1e-3) == 2.0
    assert len(data["sat1"]["trajectory"]["t"]) > 0
    assert len(data["sat2"]["trajectory"]["t"]) > 0

