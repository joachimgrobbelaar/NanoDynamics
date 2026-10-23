"""Unit tests for PINN-OD (Physics-Informed Orbit Determination)."""

import math
import numpy as np
import pytest
import torch

from ai.orbit_determination import (
    PINNOrbitDeterminator,
    compute_differentiable_accel,
    ground_station_eci,
    propagate_rk4_differentiable,
    topocentric_ra_dec,
)
from leo_simulator.constants import MU_EARTH, R_EARTH


def test_ground_station_eci():
    t_sec = torch.tensor([0.0, 100.0, 200.0], dtype=torch.float64)
    station_eci = ground_station_eci(lat_deg=0.0, lon_deg=0.0, alt_m=0.0, t_sec=t_sec)
    assert station_eci.shape == (3, 3)
    # Equatorial ground station radius must equal R_EARTH
    r_mag = torch.norm(station_eci, dim=-1)
    assert torch.allclose(r_mag, torch.full_like(r_mag, R_EARTH), atol=1.0)


def test_differentiable_propagation_autograd():
    # 400 km circular orbit
    r0 = torch.tensor([R_EARTH + 400_000.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    v_circ = math.sqrt(MU_EARTH / (R_EARTH + 400_000.0))
    v0 = torch.tensor([0.0, v_circ, 0.0], dtype=torch.float64, requires_grad=True)
    b_coeff = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)

    time_steps = torch.linspace(0, 300, 4, dtype=torch.float64)
    pos, vel = propagate_rk4_differentiable(r0, v0, b_coeff, time_steps)

    assert pos.shape == (4, 3)
    assert vel.shape == (4, 3)

    # Autograd graph check
    loss = torch.sum(pos) + torch.sum(vel)
    loss.backward()

    assert r0.grad is not None
    assert v0.grad is not None
    assert b_coeff.grad is not None
    assert not torch.isnan(r0.grad).any()


def test_pinn_od_fitting_convergence():
    pod = PINNOrbitDeterminator(station_lat_deg=28.5, station_lon_deg=-80.6, station_alt_m=10.0)

    # Ground truth initial state
    r_true = np.array([R_EARTH + 450_000.0, 0.0, 100_000.0], dtype=np.float64)
    v_circ = math.sqrt(MU_EARTH / np.linalg.norm(r_true))
    v_true = np.array([0.0, v_circ, 1000.0], dtype=np.float64)
    b_true = 0.015  # m^2/kg

    epochs = np.linspace(0, 180, 5)
    obs = pod.generate_synthetic_observations(r_true, v_true, b_true, epochs, noise_std_rad=0.0)

    # Initial guess perturbed by 5 km and 10 m/s
    r_guess = r_true + np.array([5000.0, -3000.0, 2000.0])
    v_guess = v_true + np.array([5.0, -10.0, 8.0])
    b_guess = 0.03

    result = pod.fit(
        epochs_sec=epochs,
        observed_ra_rad=obs["ra_rad"],
        observed_dec_rad=obs["dec_rad"],
        initial_guess_r0=r_guess,
        initial_guess_v0=v_guess,
        initial_guess_b=b_guess,
        iterations=50,
        learning_rate=0.001,
    )

    # Optimization must substantially decrease angular tracking loss
    assert result["final_loss"] < result["loss_history"][0]
    assert result["estimated_b"] > 0.0
