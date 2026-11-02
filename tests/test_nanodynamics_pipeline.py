"""Comprehensive automated test suite for Nanodynamics astrodynamics core, OrbitGuard AI, and FastAPI endpoints."""

from __future__ import annotations

import datetime
import math
import numpy as np
import pytest
from starlette.testclient import TestClient

from core.dynamics import (
    MU_EARTH,
    R_EARTH,
    OrbitalElements,
    coe_to_rv,
    compute_vnc_basis,
    ecef_to_geodetic,
    eci_to_ecef,
    rv_to_coe,
)
from core.propagator import NumericalCowellPropagator, SGP4Propagator
from ai.orbitguard import (
    ConjunctionEvent,
    OrbitGuardScreener,
    calculate_avoidance_maneuver,
    calculate_collision_probability,
    compute_collision_plane_basis,
    evaluate_encounter_risk,
)
from api.main import app
from workers.background_tasks import TLEIngestionWorker


@pytest.fixture
def api_client():
    return TestClient(app)


def test_coe_to_rv_and_back():
    """Verify Keplerian elements to Cartesian state vector round-trip conversion."""
    coe_orig = OrbitalElements(
        a=7000000.0,
        e=0.015,
        i=math.radians(51.6),
        raan=math.radians(120.0),
        arg_pe=math.radians(45.0),
        true_anomaly=math.radians(30.0),
    )

    r, v = coe_to_rv(coe_orig)
    assert np.linalg.norm(r) > R_EARTH
    assert 6000.0 < np.linalg.norm(v) < 8500.0

    coe_recovered = rv_to_coe(r, v)
    assert math.isclose(coe_recovered.a, coe_orig.a, rel_tol=1e-3)
    assert math.isclose(coe_recovered.e, coe_orig.e, rel_tol=1e-3)
    assert math.isclose(coe_recovered.i, coe_orig.i, rel_tol=1e-3)
    assert math.isclose(coe_recovered.raan, coe_orig.raan, rel_tol=1e-3)
    assert math.isclose(coe_recovered.arg_pe, coe_orig.arg_pe, rel_tol=1e-3)


def test_cowell_numerical_propagation():
    """Verify numerical Cowell integration with J2 and thermospheric drag."""
    propagator = NumericalCowellPropagator(
        mass_kg=4.0,
        drag_area_m2=0.03,
        cd=2.2,
        include_j2=True,
        include_drag=True,
    )

    # 500 km circular orbit
    r0 = np.array([R_EARTH + 500000.0, 0.0, 0.0])
    v_circ = math.sqrt(MU_EARTH / (R_EARTH + 500000.0))
    v0 = np.array([0.0, v_circ, 0.0])

    traj = propagator.propagate(r0, v0, duration_seconds=1200.0, step_seconds=60.0)
    assert traj.success
    assert len(traj.t) > 10
    # Altitude remains within acceptable range for 1200s
    for alt in traj.altitudes:
        assert 450000.0 < alt < 550000.0


def test_sgp4_tle_propagation():
    """Verify analytical SGP4 ephemeris generation using ISS TLE."""
    line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9001"
    line2 = "2 25544  51.6400 120.0000 0003000  45.0000  80.0000 15.50000000430001"

    sgp = SGP4Propagator(line1, line2)
    start_dt = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    traj = sgp.propagate(start_dt, duration_seconds=600.0, step_seconds=60.0)

    assert traj.success
    assert len(traj.t) == 11
    # Check geocentric altitude is approx 400-430 km
    for alt in traj.altitudes:
        assert 350000.0 < alt < 450000.0


def test_collision_probability_and_risk_evaluation():
    """Verify 2D B-plane encounter risk scoring."""
    # Near-zero miss distance -> High Pc & Critical tier
    pc_near, tier_near = evaluate_encounter_risk(miss_distance_km=0.05)
    assert tier_near == "CRITICAL"
    assert pc_near > 1e-4

    # Moderate miss distance (800 m)
    pc_mod, tier_mod = evaluate_encounter_risk(miss_distance_km=0.8)
    assert tier_mod in ["MODERATE", "LOW"]

    # Wide miss distance (15 km) -> Negligible tier
    pc_far, tier_far = evaluate_encounter_risk(miss_distance_km=15.0)
    assert tier_far == "NEGLIGIBLE"
    assert pc_far == 0.0


def test_conjunction_screener_and_cdm():
    """Verify pairwise trajectory screening and CCSDS CDM formatting."""
    screener = OrbitGuardScreener(screening_threshold_km=20.0)

    # Primary in 500 km circular equatorial orbit
    r1 = np.array([R_EARTH + 500000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7600.0, 0.0])

    # Secondary offset by 2 km in radial and slight velocity difference
    r2 = np.array([R_EARTH + 502000.0, 0.0, 0.0])
    v2 = np.array([0.0, 7590.0, 50.0])

    events = screener.screen_encounter(
        r1_0=r1,
        v1_0=v1,
        r2_0=r2,
        v2_0=v2,
        duration_hours=1.0,
        primary_id="SAT-01",
        secondary_id="DEB-02",
    )

    assert len(events) >= 1
    best_event = events[0]
    assert best_event.miss_distance_km <= 20.0

    cdm_text = best_event.generate_cdm()
    assert "CCSDS_CDM_VERS = 1.0" in cdm_text
    assert "OBJECT_1 = SAT-01" in cdm_text
    assert "MISS_DISTANCE" in cdm_text


def test_avoidance_maneuver_calculation():
    """Verify prograde/retrograde collision avoidance burn calculation."""
    event = ConjunctionEvent(
        tca_seconds=3600.0,
        tca_datetime=datetime.datetime.now(datetime.timezone.utc),
        miss_distance_km=0.15,
        radial_miss_km=0.05,
        in_track_miss_km=0.1,
        cross_track_miss_km=0.02,
        relative_speed_km_s=14.0,
        r1_tca=np.array([R_EARTH + 500000.0, 0.0, 0.0]),
        v1_tca=np.array([0.0, 7600.0, 0.0]),
        r2_tca=np.array([R_EARTH + 500150.0, 0.0, 0.0]),
        v2_tca=np.array([0.0, -7600.0, 0.0]),
        collision_probability=0.002,
        risk_tier="CRITICAL",
    )

    maneuver = calculate_avoidance_maneuver(event, lead_time_seconds=1800.0, target_clearance_km=3.0)
    assert maneuver["status"] == "MANEUVER_RECOMMENDED"
    assert maneuver["delta_v_magnitude_m_s"] > 0.0
    assert maneuver["projected_miss_km"] > 3.0


def test_api_health_endpoint(api_client):
    """Verify GET /api/health returns 200 and telemetry metadata."""
    res = api_client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "nanodynamics-platform"
    assert data["features"]["sgp4_tle"] is True
    assert data["features"]["orbitguard_conjunction_screening"] is True


def test_api_simulation_propagate(api_client):
    """Verify POST /api/simulation/propagate returns valid trajectory."""
    payload = {
        "r0": [R_EARTH + 500000.0, 0.0, 0.0],
        "v0": [0.0, 7600.0, 0.0],
        "duration_seconds": 600.0,
        "step_seconds": 60.0,
        "mass_kg": 4.0,
        "drag_area_m2": 0.03,
        "cd": 2.2,
        "include_j2": True,
        "include_drag": True,
    }
    res = api_client.post("/api/simulation/propagate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert len(data["trajectory"]) > 0


def test_api_conjunction_evaluate(api_client):
    """Verify POST /api/conjunction/evaluate risk calculations."""
    payload = {
        "miss_distance_km": 0.25,
        "combined_radius_m": 2.0,
        "sigma_radial_m": 50.0,
        "sigma_in_track_m": 250.0,
        "sigma_cross_track_m": 80.0,
    }
    res = api_client.post("/api/conjunction/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_tier"] in ["CRITICAL", "HIGH", "MODERATE"]
    assert "collision_probability" in data


def test_api_conjunction_cdm_format(api_client):
    """Verify POST /api/conjunction/cdm returns CCSDS standard plaintext."""
    payload = {
        "miss_distance_km": 0.35,
        "combined_radius_m": 2.5,
        "sigma_radial_m": 40.0,
        "sigma_in_track_m": 200.0,
        "sigma_cross_track_m": 60.0,
    }
    res = api_client.post("/api/conjunction/cdm", json=payload)
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    assert "CCSDS_CDM_VERS = 1.0" in res.text
