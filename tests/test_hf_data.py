"""Offline tests for leo_simulator.hf_data using synthetic parquet fixtures."""

import importlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")


@pytest.fixture
def hf(tmp_path, monkeypatch):
    monkeypatch.setenv("LEO_HF_CACHE", str(tmp_path))
    import leo_simulator.hf_data as m
    return importlib.reload(m)


def _write(tmp, repo, rel, df):
    p = tmp / repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)


def test_mean_motion_iss(hf):
    a = hf.mean_motion_to_sma_m(15.5)
    assert 6.7e6 < a < 6.85e6


def test_tle_history_filter(hf, tmp_path):
    df = pd.DataFrame({
        "norad_id": [25544, 25544, 1],
        "epoch": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01"], utc=True),
        "inclination": [51.6] * 3, "raan": [10.0] * 3, "eccentricity": [0.0005] * 3,
        "arg_perigee": [90.0] * 3, "mean_anomaly": [0.0] * 3, "mean_motion": [15.5] * 3,
        "mean_motion_dot": [1e-5] * 3, "bstar": [1e-4] * 3,
    })
    _write(tmp_path, "space-track-tle-history", "data/tle_2024.parquet", df)
    recs = hf.load_tle_history(25544, 2024)
    assert len(recs) == 2 and recs[0]["epoch"] < recs[1]["epoch"]


def test_space_weather_lookup(hf, tmp_path):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "f107_obs": [150.0, np.nan], "f107_obs_center81": [140.0, 141.0], "ap_avg": [10.0, 12.0],
    })
    _write(tmp_path, "space-weather-indices", "data/space_weather_indices.parquet", df)
    tab = hf.load_space_weather("2024-01-01", "2024-01-02")
    day2 = pd.Timestamp("2024-01-02").timestamp()
    out = hf.space_weather_lookup(day2, tab)
    assert out["f107"] == 150.0 and out["ap"] == 12.0  # NaN falls back to default


def test_status_and_missing_file(hf):
    assert set(hf.dataset_status()) == set(hf.DATASETS)
    with pytest.raises(FileNotFoundError):
        hf.load_starlink_tles()


def test_historical_space_weather_and_validate_arc(hf, tmp_path):
    from leo_simulator.space_weather import SpaceWeatherManager
    from scripts.hf_validate_pinn import validate_tle_arc

    df_sw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "f107_obs": [165.0, 170.0], "f107_obs_center81": [160.0, 162.0], "ap_avg": [12.0, 15.0],
    })
    _write(tmp_path, "space-weather-indices", "data/space_weather_indices.parquet", df_sw)
    mgr = SpaceWeatherManager()
    idx = mgr.get_historical_indices(pd.Timestamp("2024-01-01").timestamp())
    assert idx.source == "hf_space_weather_indices"
    assert idx.f107 == pytest.approx(165.0)

    df_tle = pd.DataFrame({
        "norad_id": [25544, 25544],
        "epoch": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T01:30:00Z"], utc=True),
        "inclination": [51.6, 51.6], "raan": [10.0, 10.0], "eccentricity": [0.0005, 0.0005],
        "arg_perigee": [90.0, 90.0], "mean_anomaly": [0.0, 5.0], "mean_motion": [15.5, 15.5001],
        "mean_motion_dot": [1e-5, 1e-5], "bstar": [1e-4, 1e-4],
    })
    _write(tmp_path, "space-track-tle-history", "data/tle_2024.parquet", df_tle)
    res = validate_tle_arc(norad_id=25544, year=2024, max_pairs=1)
    assert res["evaluated_arcs"] == 1
    assert res["rk45_radial_rmse_km"] >= 0.0
    assert res["pinn_vs_rk45_3d_rmse_km"] >= 0.0

