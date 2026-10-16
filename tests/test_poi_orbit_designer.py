"""Unit tests for POI Orbit Designer and Specialized Orbit Presets."""

import math
import numpy as np
import pytest

from leo_simulator.coordinates import eci_to_geodetic
from leo_simulator.orbit.poi_orbit_designer import (
    MAJOR_POIS,
    design_orbit_for_poi,
    design_orbit_for_poi_cluster,
    find_poi_by_name,
    get_all_pois,
    get_specialized_orbit_presets,
)


def test_poi_catalog_and_lookup():
    pois = get_all_pois()
    assert len(pois) >= 15
    tokyo = find_poi_by_name("Tokyo")
    assert tokyo is not None
    assert tokyo.country == "Japan"
    assert tokyo.lat_deg == pytest.approx(35.6762, abs=0.01)

    london = find_poi_by_name("london")
    assert london is not None
    assert london.country == "United Kingdom"

    none_poi = find_poi_by_name("NonExistentAtlantis")
    assert none_poi is None


def test_design_orbit_for_poi_direct_flyover():
    # Test for Paris: Lat 48.8566, Lon 2.3522
    target_lat = 48.8566
    target_lon = 2.3522
    target_alt = 500.0

    res = design_orbit_for_poi(lat_deg=target_lat, lon_deg=target_lon, target_alt_km=target_alt)
    elem = res["elements"]
    assert elem["inclination_deg"] >= target_lat
    assert elem["alt_km"] == pytest.approx(target_alt, abs=1e-3)

    # Initial Cartesian position in ECI at t=0
    r0 = np.array(res["initial_state"]["r"], dtype=np.float64)
    lat_calc, lon_calc, alt_calc = eci_to_geodetic(r0, t_seconds=0.0)

    # At t=0, the sub-satellite point must match target coordinates
    assert lat_calc == pytest.approx(target_lat, abs=0.1)
    assert lon_calc == pytest.approx(target_lon, abs=0.1)
    assert alt_calc == pytest.approx(target_alt, abs=5.0)


def test_design_orbit_for_poi_cluster():
    cluster = ["Tokyo", "San Francisco", "Honolulu"]
    res = design_orbit_for_poi_cluster(cluster, target_alt_km=600.0)
    assert res["cluster_count"] == 3
    assert res["elements"]["inclination_deg"] >= 37.0


def test_specialized_orbit_presets():
    presets = get_specialized_orbit_presets()
    assert "GEO" in presets
    assert "GSO_INCLINED" in presets
    assert "MOLNIYA" in presets
    assert "TUNDRA" in presets
    assert "SSO" in presets
    assert "POLAR_LEO" in presets

    geo = presets["GEO"]
    assert geo["inclination_deg"] == 0.0
    assert geo["altitude_km"] == pytest.approx(35786.0, abs=50.0)
    assert geo["period_s"] == pytest.approx(86164.0, abs=10.0)

    molniya = presets["MOLNIYA"]
    assert molniya["inclination_deg"] == pytest.approx(63.435, abs=0.01)
    assert molniya["eccentricity"] > 0.7
