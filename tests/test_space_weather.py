"""Unit tests for SpaceWeatherManager and dynamic ionospheric TEC modeling."""

from fastapi.testclient import TestClient
import pytest

from leo_simulator.space_weather import SpaceWeatherIndices, SpaceWeatherManager
from visualization_3d.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_density_multiplier_scaling():
    # Solar quiet baseline
    mult_quiet = SpaceWeatherManager.compute_density_multiplier(f107=70.0, kp=0.0)
    assert mult_quiet == pytest.approx(1.0, abs=1e-4)

    # Moderate solar condition
    mult_mod = SpaceWeatherManager.compute_density_multiplier(f107=120.0, kp=2.0)
    expected_mod = 1.0 + (0.008 * 50.0) + (0.15 * 2.0)
    assert mult_mod == pytest.approx(expected_mod, abs=1e-4)

    # Extreme solar storm
    mult_storm = SpaceWeatherManager.compute_density_multiplier(f107=250.0, kp=8.0)
    assert mult_storm > 3.0


def test_ionospheric_tec_and_delays():
    # Equatorial Anomaly peak (15 deg lat) vs mid-latitude (45 deg lat)
    tec_eia = SpaceWeatherManager.compute_ionospheric_tec(lat_deg=15.0, lon_deg=0.0, alt_km=350.0, f107=100.0)
    tec_mid = SpaceWeatherManager.compute_ionospheric_tec(lat_deg=45.0, lon_deg=0.0, alt_km=350.0, f107=100.0)

    assert tec_eia["tec_tecu"] > tec_mid["tec_tecu"]

    # RF group delay scaling: delay proportional to 1/f^2
    # UHF (435 MHz) delay must be substantially higher than L1 (1.575 GHz) and S-band (2.4 GHz)
    assert tec_eia["iono_delay_uhf_ns"] > tec_eia["iono_delay_l1_ns"]
    assert tec_eia["iono_delay_l1_ns"] > tec_eia["iono_delay_s_band_ns"]


def test_space_weather_fetch():
    mgr = SpaceWeatherManager.get_instance()
    indices = mgr.fetch_indices(force_refresh=False)
    assert isinstance(indices, SpaceWeatherIndices)
    assert indices.f107 >= 50.0
    assert 0.0 <= indices.kp <= 9.0
    assert indices.density_scale >= 1.0
    assert indices.source in {"noaa_swpc_live", "disk_cache", "default_climatology"}


def test_api_space_weather_endpoint(client):
    response = client.get("/api/space_weather")
    assert response.status_code == 200
    data = response.json()
    assert "f107" in data
    assert "kp" in data
    assert "density_scale" in data
    assert "source" in data
    assert "timestamp" in data
    assert data["density_scale"] > 0.5
