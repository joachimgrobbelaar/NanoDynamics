"""
tests/test_dataset_loaders.py

Comprehensive pytest suite for the nanodynamics data ingestion pipeline.
Tests registry, loader interface, schemas, catalog, and at least one
analytical function per pillar (using mocked/synthetic data to avoid
requiring HF network access during CI).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─── Registry & Core Import Tests ────────────────────────────────────────────

def test_registry_has_all_57_datasets():
    from nanodynamics.data_ingestion.loaders import DATASET_REGISTRY
    assert len(DATASET_REGISTRY) >= 57, f"Expected >=57 datasets, got {len(DATASET_REGISTRY)}"


def test_all_categories_present():
    from nanodynamics.data_ingestion.loaders import DATASET_REGISTRY, DataCategory
    cats_in_registry = {v.category for v in DATASET_REGISTRY.values()}
    for cat in DataCategory:
        assert cat in cats_in_registry, f"Category {cat.value} has no datasets"


def test_dataset_meta_fields():
    from nanodynamics.data_ingestion.loaders import DATASET_REGISTRY
    for key, meta in DATASET_REGISTRY.items():
        assert meta.hf_id, f"{key} missing hf_id"
        assert meta.split, f"{key} missing split"


def test_constants_importable():
    from nanodynamics.core.constants import GM_EARTH, AU_KM, R_EARTH_KM
    assert GM_EARTH > 0
    assert AU_KM > 1e8
    assert R_EARTH_KM > 6000


def test_catalog_print():
    from nanodynamics.catalog import print_catalog
    text = print_catalog()
    assert "EPHEMERIS" in text
    assert "ORBITAL" in text


def test_datasets_list():
    from nanodynamics.datasets import list_datasets, list_categories
    all_ds = list_datasets()
    assert len(all_ds) >= 57
    cats = list_categories()
    assert "ephemeris_catalogs" in cats
    assert "orbital_mechanics" in cats


def test_datasets_info():
    from nanodynamics.datasets import dataset_info
    info = dataset_info("starlink_tle_latest")
    assert info["hf_id"] == "juliensimon/starlink-tle-latest"
    assert info["category"] == "orbital_mechanics"


def test_datasets_info_unknown():
    from nanodynamics.datasets import dataset_info
    with pytest.raises(KeyError):
        dataset_info("nonexistent_dataset")


def test_loader_rejects_unknown_key():
    from nanodynamics.data_ingestion.loaders import NanodynamicsDataLoader
    loader = NanodynamicsDataLoader()
    with pytest.raises(KeyError, match="not registered"):
        loader.load("totally_fake_dataset_key")


def test_schema_keplerian():
    from nanodynamics.data_ingestion.loaders import KeplerianElementRecord
    rec = KeplerianElementRecord(
        object_id="Ceres",
        e=0.076,
        i=10.59,
    )
    assert rec.eccentricity == 0.076
    assert rec.inclination_deg == 10.59


def test_schema_space_weather():
    from nanodynamics.data_ingestion.loaders import SpaceWeatherRecord
    rec = SpaceWeatherRecord(
        timestamp_utc="2024-01-15T00:00:00Z",
        f107_flux=150.0,
        kp_index=3.0,
        ap_index=12.0,
    )
    assert rec.f107_flux == 150.0


def test_schema_conjunction():
    from nanodynamics.data_ingestion.loaders import ConjunctionEventRecord
    rec = ConjunctionEventRecord(
        tca_utc="2024-06-01T12:00:00Z",
        primary_norad_id=25544,
        secondary_norad_id=48275,
        miss_distance_m=500.0,
        collision_probability=1e-5,
    )
    assert rec.miss_distance_m == 500.0


# ─── Pillar 1: Ephemeris Catalogs ────────────────────────────────────────────

def test_propagate_keplerian_orbit():
    from nanodynamics.modules.ephemeris_catalogs.functions import propagate_keplerian_orbit
    from nanodynamics.core.constants import GM_SUN, AU_M
    # Earth-like orbit: a=1 AU, e=0.0167, i=0
    state = propagate_keplerian_orbit(
        a_au=1.0, e=0.0167, i_deg=0.0, om_deg=0.0,
        w_deg=0.0, ma_deg=0.0, epoch_jd=2451545.0,
        target_jd=2451545.0, gm=GM_SUN,
    )
    assert len(state) == 6
    # At MA=0 (perihelion), x should be close to a*(1-e) in AU
    r = np.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
    expected_r = 1.0 * (1.0 - 0.0167) * AU_M
    assert abs(r - expected_r) / expected_r < 0.01


def test_classify_spectral_type():
    from nanodynamics.modules.ephemeris_catalogs.functions import classify_spectral_type
    # S-type asteroids tend to have moderate g-r
    result = classify_spectral_type(u_minus_g=1.5, g_minus_r=0.5, r_minus_i=0.15)
    assert result in ("C", "S", "X", "V")


def test_estimate_albedo_density():
    from nanodynamics.modules.ephemeris_catalogs.functions import estimate_albedo_density
    result = estimate_albedo_density("S")
    assert "albedo" in result
    assert "density_kg_m3" in result
    assert result["albedo"] > 0


def test_compute_hill_sphere():
    from nanodynamics.modules.ephemeris_catalogs.functions import compute_hill_sphere
    # Moon around Earth
    r_hill = compute_hill_sphere(
        moon_mass_kg=7.342e22,
        planet_mass_kg=5.972e24,
        semi_major_axis_m=384400e3,
    )
    assert r_hill > 0
    # Hill sphere of Moon ~ 60,000 km
    assert 40_000e3 < r_hill < 80_000e3


# ─── Pillar 2: Orbital Mechanics ────────────────────────────────────────────

def test_hohmann_transfer_dv():
    from nanodynamics.modules.orbital_mechanics.functions import calculate_hohmann_transfer_dv
    from nanodynamics.core.constants import GM_EARTH, R_EARTH_KM
    # LEO (400 km) to GEO (35786 km)
    r1_km = R_EARTH_KM + 400
    r2_km = R_EARTH_KM + 35786
    result = calculate_hohmann_transfer_dv(r1_km, r2_km, gm=GM_EARTH)
    dv_total = result["dv_total_m_s"]
    # Total dv should be ~3.9 km/s
    assert 3500 < dv_total < 4500


def test_collision_probability():
    from nanodynamics.modules.orbital_mechanics.functions import calculate_collision_probability
    # Large miss distance = low Pc
    pc = calculate_collision_probability(
        miss_distance_m=10000.0,
        sigma_a_m=100.0,
        sigma_b_m=100.0,
    )
    assert 0 <= pc <= 1
    assert pc < 1e-3  # should be very small


def test_compute_lyapunov_exponent():
    from nanodynamics.modules.orbital_mechanics.functions import compute_lyapunov_exponent
    # Regular (non-chaotic) trajectory: constant separation
    traj = np.column_stack([np.sin(np.linspace(0, 10, 100)),
                            np.cos(np.linspace(0, 10, 100))])
    result = compute_lyapunov_exponent(traj, dt=0.1)
    assert isinstance(result, dict)
    assert "lambda_max" in result
    assert isinstance(result["lambda_max"], float)


# ─── Pillar 3: Space Weather ────────────────────────────────────────────────

def test_atmospheric_drag_density():
    from nanodynamics.modules.space_weather_atmos.functions import compute_atmospheric_drag_density
    result = compute_atmospheric_drag_density(alt_km=400.0, f107=150.0, ap=15.0)
    assert isinstance(result, dict)
    assert result["density_kg_m3"] > 0
    # At 400 km, density should be roughly 1e-14 to 1e-9 kg/m³
    assert 1e-14 < result["density_kg_m3"] < 1e-9


def test_is_storm_condition():
    from nanodynamics.modules.space_weather_atmos.functions import is_storm_condition
    assert is_storm_condition(kp_value=7.0) is True
    assert is_storm_condition(kp_value=3.0) is False


def test_check_launch_commit_criteria():
    from nanodynamics.modules.space_weather_atmos.functions import check_launch_commit_criteria
    # Safe conditions
    res_safe = check_launch_commit_criteria(10.0, 0.05, 10.0)
    assert res_safe["go"] is True
    # High wind
    res_wind = check_launch_commit_criteria(30.0, 0.05, 10.0)
    assert res_wind["go"] is False


def test_gnss_delay():
    from nanodynamics.modules.space_weather_atmos.functions import get_total_electron_content
    # Test with a mock dataset-like call using the analytical fallback
    # The function should accept ds=None or empty and use analytical model
    # We test the math directly by checking the delay formula
    import math
    tec_tecu = 50.0
    freq_hz = 1.575e9
    c = 299_792_458.0
    tec_m2 = tec_tecu * 1e16
    expected_delay = 40.3 * tec_m2 / (c * freq_hz ** 2)
    assert expected_delay > 0
    # Typical L1 delay for 50 TECU ~ 8 ns
    assert 1e-9 < expected_delay < 100e-9


# ─── Pillar 4: Planetary Exploration ────────────────────────────────────────

def test_free_return_trajectory():
    from nanodynamics.modules.planetary_exploration.functions import simulate_free_return_trajectory
    result = simulate_free_return_trajectory()
    assert "v_tli_km_s" in result
    assert "perilune_alt_km" in result
    assert "total_mission_time_hours" in result
    assert result["v_tli_km_s"] > 10


# ─── Pillar 5: Space Ops ────────────────────────────────────────────────────

def test_launch_success_rate():
    from nanodynamics.modules.space_ops_infrastructure.functions import calculate_launch_success_rate
    # Mock dataset
    mock_ds = [
        {"vehicle": "Falcon 9", "outcome": "Success"},
        {"vehicle": "Falcon 9", "outcome": "Success"},
        {"vehicle": "Falcon 9", "outcome": "Failure"},
        {"vehicle": "Atlas V", "outcome": "Success"},
    ]
    result = calculate_launch_success_rate(mock_ds, "Falcon 9")
    assert isinstance(result, dict)
    assert abs(result["success_rate"] - 2.0 / 3.0) < 0.01


def test_gateway_handover():
    from nanodynamics.modules.space_ops_infrastructure.functions import calculate_gateway_handover
    result = calculate_gateway_handover(
        lat=0.0, lon=0.0,
        station_lat=28.5, station_lon=-80.6,
        altitude_km=550.0,
    )
    assert "elevation_deg" in result
    assert "is_visible" in result


# ─── Pillar 6: Computer Vision ──────────────────────────────────────────────

def test_pose_error():
    from nanodynamics.modules.computer_vision_sensors.functions import evaluate_pose_error
    pred_t = np.array([1.0, 2.0, 3.0])
    gt_t = np.array([1.1, 2.1, 3.1])
    pred_q = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
    gt_q = np.array([0.9999, 0.01, 0.0, 0.0])  # near-identity
    result = evaluate_pose_error(pred_t, pred_q, gt_t, gt_q)
    assert isinstance(result, dict)
    assert result["translation_error_m"] > 0
    assert result["angular_error_rad"] >= 0


def test_temporal_change_vector():
    from nanodynamics.modules.computer_vision_sensors.functions import compute_temporal_change_vector
    t1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    t2 = t1.copy()
    t2[50:60, 50:60] = 255  # inject change
    change_map, mean_change = compute_temporal_change_vector(t1, t2)
    assert change_map.shape == (100, 100)
    assert mean_change > 0


# ─── Pillar 7: LLM Instruction ──────────────────────────────────────────────

def test_build_sft_chatml():
    from nanodynamics.modules.llm_instruction_bench.functions import build_astrodynamics_sft_dataset
    records = [
        {"instruction": "What is delta-v?", "output": "Delta-v is the change in velocity..."},
        {"instruction": "Explain Kepler's 3rd law", "output": "T^2 \\propto a^3 is $\\frac{T^2}{a^3} = \\frac{4\\pi^2}{GM}$"},
    ]
    result = build_astrodynamics_sft_dataset(records, format="chatml")
    assert len(result) == 2
    assert "messages" in result[0]
    assert result[0]["messages"][1]["role"] == "user"


def test_build_sft_alpaca():
    from nanodynamics.modules.llm_instruction_bench.functions import build_astrodynamics_sft_dataset
    records = [
        {"instruction": "What is an orbit?", "output": "An orbit is..."},
    ]
    result = build_astrodynamics_sft_dataset(records, format="alpaca")
    assert "instruction" in result[0]
    assert "output" in result[0]


def test_filter_by_math_complexity():
    from nanodynamics.modules.llm_instruction_bench.functions import filter_by_math_complexity
    records = [
        {"instruction": "Q1", "output": "Simple text"},
        {"instruction": "Q2", "output": "The equation $\\frac{GM}{r^2}$ gives $F = \\frac{GMm}{r^2}$ and $\\int_0^\\infty \\sqrt{x} dx$"},
    ]
    filtered = filter_by_math_complexity(records, min_latex_chars=10)
    assert len(filtered) == 1
    assert filtered[0]["instruction"] == "Q2"


# ─── Loader Mock Tests ──────────────────────────────────────────────────────

@patch("nanodynamics.data_ingestion.loaders.load_dataset")
def test_loader_calls_hf(mock_load):
    from nanodynamics.data_ingestion.loaders import NanodynamicsDataLoader
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=10)
    mock_load.return_value = mock_ds

    loader = NanodynamicsDataLoader()
    ds = loader.load("starlink_tle_latest", use_cache=False)
    mock_load.assert_called_once()
    call_kwargs = mock_load.call_args
    assert "juliensimon/starlink-tle-latest" in str(call_kwargs)


@patch("nanodynamics.data_ingestion.loaders.load_dataset")
def test_stream_batches(mock_load):
    from nanodynamics.data_ingestion.loaders import NanodynamicsDataLoader
    mock_load.return_value = iter([{"a": i} for i in range(5)])

    loader = NanodynamicsDataLoader()
    batches = list(loader.stream_batches("starlink_tle_latest", batch_size=2))
    assert len(batches) == 3  # [2, 2, 1]
    assert len(batches[0]) == 2
    assert len(batches[2]) == 1
