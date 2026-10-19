"""Unit and integration tests for the parametric experimentation suite and atmospheric models."""

import os
import pytest
from fastapi.testclient import TestClient

from leo_simulator.experiment import (
    ScaledAtmosphere,
    get_atmosphere_by_name,
    run_parametric_sweep,
    run_single_simulation,
)
from leo_simulator.models.drag import (
    ExponentialAtmosphere,
    PiecewiseExponentialAtmosphere,
)
from visualization_3d.app import EXPERIMENTS_DIR, app

client = TestClient(app)


class TestAtmosphereConsistency:
    """Validate atmospheric density models across reference altitudes."""

    def test_piecewise_vs_exponential_densities(self):
        """Verify piecewise US Standard 1976 atmosphere produces physically expected densities."""
        atm_piecewise = PiecewiseExponentialAtmosphere()
        atm_high = get_atmosphere_by_name("high_solar")
        atm_low = get_atmosphere_by_name("low_solar")

        # 200 km altitude density sanity check
        rho_200 = atm_piecewise.density(200_000.0)
        assert 1e-10 < rho_200 < 5e-10, f"Expected rho(200km) ~ 2.8e-10, got {rho_200}"

        # 400 km altitude density sanity check
        rho_400 = atm_piecewise.density(400_000.0)
        assert 1e-12 < rho_400 < 1e-11, f"Expected rho(400km) ~ 3.7e-12, got {rho_400}"

        # Scale factor
        scaled = ScaledAtmosphere(atm_piecewise, scale_factor=2.0)
        assert abs(scaled.density(200_000.0) - 2.0 * rho_200) < 1e-14


class TestParametricSweeps:
    """Test parametric sweeps over spacecraft and orbital independent variables."""

    def test_sweep_mass_linear_lifetime_scaling(self):
        """Lifetime scales monotonically with mass at a low decaying altitude (250 km)."""
        res = run_parametric_sweep(
            param_name="mass",
            values=[2.0, 4.0],
            base_params={
                "name": "MassSweepSat",
                "mass": 4.0,
                "drag_area": 0.03,
                "cd": 2.2,
                "altitude_km": 250.0,
                "eccentricity": 0.001,
                "inclination_deg": 51.6,
                "raan_deg": 45.0,
                "include_j2": True,
                "include_drag": True,
                "include_moon": False,
            },
            atmosphere_type="piecewise",
            max_duration_seconds=5 * 86400.0,
            quiet=True,
            output_dir=EXPERIMENTS_DIR,
        )

        assert len(res.results) == 2
        r1, r2 = res.results[0], res.results[1]
        assert r2.mass_kg > r1.mass_kg
        assert r2.ballistic_coeff_kg_m2 > r1.ballistic_coeff_kg_m2
        assert r2.lifetime_seconds > r1.lifetime_seconds

        # Clean up
        if res.csv_filepath and os.path.exists(res.csv_filepath):
            os.remove(res.csv_filepath)

    def test_sweep_ballistic_coefficient(self):
        """Test sweep over ballistic coefficient B."""
        res = run_parametric_sweep(
            param_name="ballistic_coefficient",
            values=[20.0, 60.0],
            base_params={
                "mass": 4.0,
                "drag_area": 0.03,
                "cd": 2.2,
                "altitude_km": 250.0,
                "eccentricity": 0.001,
                "inclination_deg": 51.6,
                "raan_deg": 45.0,
            },
            max_duration_seconds=5 * 86400.0,
            quiet=True,
        )
        assert len(res.results) == 2
        assert res.results[1].lifetime_seconds > res.results[0].lifetime_seconds


class TestExperimentationEndpoints:
    """Test FastAPI /experiment endpoints."""

    def test_api_run_experiment_endpoint(self):
        payload = {
            "param_name": "drag_area",
            "values": [0.02, 0.05],
            "base_params": {
                "name": "ApiExpSat",
                "parent_body": "Earth",
                "mass": 4.0,
                "drag_area": 0.03,
                "cd": 2.2,
                "altitude_km": 250.0,
                "eccentricity": 0.001,
                "inclination_deg": 51.6,
                "raan_deg": 45.0,
                "atmosphere_type": "piecewise",
                "density_scale": 1.0,
                "include_j2": True,
                "include_drag": True,
                "include_moon": False,
            },
            "atmosphere_type": "piecewise",
            "density_scale": 1.0,
            "max_duration_days": 5.0,
            "quiet": True,
        }
        res = client.post("/experiment/run", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert len(data["results"]) == 2
        assert "csv_filename" in data

        # Test download
        csv_name = data["csv_filename"]
        dl_res = client.get(f"/experiment/download/{csv_name}")
        assert dl_res.status_code == 200
        assert "run_id" in dl_res.text

        # Test list
        list_res = client.get("/experiment/list")
        assert list_res.status_code == 200
        assert csv_name in list_res.json()["experiments"]

        # Clean up
        csv_path = os.path.join(EXPERIMENTS_DIR, csv_name)
        if os.path.exists(csv_path):
            os.remove(csv_path)
