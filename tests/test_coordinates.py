import numpy as np
import pytest
from leo_simulator.constants import OMEGA_EARTH, R_EARTH
from leo_simulator.coordinates import (
    ecef_to_geodetic,
    eci_to_ecef,
    eci_to_geodetic,
    extract_trajectory_key_events,
)


def test_eci_to_ecef_t0():
    r_eci = np.array([7000e3, 0.0, 0.0])
    r_ecef = eci_to_ecef(r_eci, t_seconds=0.0)
    assert np.allclose(r_ecef, r_eci)


def test_eci_to_ecef_quarter_day():
    # After a quarter rotation (T = 2pi / omega), theta = pi/2
    t_quarter = (np.pi / 2.0) / OMEGA_EARTH
    r_eci = np.array([7000e3, 0.0, 0.0])
    r_ecef = eci_to_ecef(r_eci, t_seconds=t_quarter)
    # [x, y, z] -> [0, -7000e3, 0]
    assert np.allclose(r_ecef, [0.0, -7000e3, 0.0], atol=1e-3)


def test_ecef_to_geodetic_equator_prime_meridian():
    r_ecef = np.array([R_EARTH + 400e3, 0.0, 0.0])
    lat, lon, alt = ecef_to_geodetic(r_ecef)
    assert pytest.approx(lat, abs=1e-5) == 0.0
    assert pytest.approx(lon, abs=1e-5) == 0.0
    assert pytest.approx(alt, abs=1e-3) == 400.0


def test_ecef_to_geodetic_north_pole():
    r_ecef = np.array([0.0, 0.0, R_EARTH + 500e3])
    lat, lon, alt = ecef_to_geodetic(r_ecef)
    assert pytest.approx(lat, abs=1e-5) == 90.0
    assert pytest.approx(alt, abs=1e-3) == 500.0


def test_vectorized_coordinates():
    t_arr = np.linspace(0, 3600, 10)
    r_arr = np.zeros((10, 3))
    r_arr[:, 0] = R_EARTH + 300e3
    
    lat, lon, alt = eci_to_geodetic(r_arr, t_arr)
    assert len(lat) == 10
    assert len(lon) == 10
    assert len(alt) == 10
    assert np.allclose(lat, 0.0)
    assert np.allclose(alt, 300.0)


def test_extract_trajectory_key_events():
    t_arr = np.array([0.0, 100.0, 200.0, 300.0])
    # Altitude descending from 200km -> 120km -> 50km -> 0km
    r_eci_arr = np.array([
        [R_EARTH + 200e3, 0.0, 0.0],
        [R_EARTH + 120e3, 0.0, 0.0],
        [R_EARTH + 50e3, 0.0, 0.0],
        [R_EARTH + 0.0, 0.0, 0.0],
    ])
    
    events = extract_trajectory_key_events(t_arr, r_eci_arr, reentry_alt_km=120.0)
    assert events["initial"] is not None
    assert events["initial"]["alt_km"] == 200.0
    
    assert events["reentry"] is not None
    assert events["reentry"]["time_s"] == 100.0
    assert events["reentry"]["alt_km"] == 120.0
    
    assert events["impact"] is not None
    assert events["impact"]["time_s"] == 300.0
    assert events["impact"]["alt_km"] == 0.0
