"""PINN-OD: Physics-Informed Orbit Determination for LEO Space Debris.

Uses PyTorch autograd to solve the inverse problem of estimating initial Cartesian
orbital state [r0, v0] and unknown ballistic coefficient B = (Cd * A / m) from
sparse angles-only (Right Ascension, Declination) ground tracking observations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim

from leo_simulator.constants import (
    ATMOSPHERE_DEFAULT_H0,
    ATMOSPHERE_DEFAULT_RHO0,
    ATMOSPHERE_DEFAULT_SCALE_HEIGHT,
    J2_EARTH,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH,
)


def ground_station_eci(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    t_sec: torch.Tensor,
    gmst0_rad: float = 0.0,
) -> torch.Tensor:
    """Compute ground station position vector in ECI frame across time epochs.

    Args:
        lat_deg: Station geodetic latitude in degrees.
        lon_deg: Station longitude in degrees.
        alt_m: Station altitude above sea level in meters.
        t_sec: 1D PyTorch tensor of epoch times in seconds.
        gmst0_rad: Greenwich Mean Sidereal Time at t=0 in radians.

    Returns:
        Tensor of shape [N, 3] representing ECI coordinates in meters.
    """
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    r_station_xy = (R_EARTH + alt_m) * math.cos(phi)
    r_station_z = (R_EARTH + alt_m) * math.sin(phi)

    # Earth rotation angle
    theta = gmst0_rad + lam + OMEGA_EARTH * t_sec
    x_eci = r_station_xy * torch.cos(theta)
    y_eci = r_station_xy * torch.sin(theta)
    z_eci = torch.full_like(x_eci, r_station_z)

    return torch.stack([x_eci, y_eci, z_eci], dim=-1)


def topocentric_ra_dec(sat_pos: torch.Tensor, station_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute topocentric Right Ascension (RA) and Declination (Dec) angles in radians."""
    rho = sat_pos - station_pos
    rho_x = rho[:, 0]
    rho_y = rho[:, 1]
    rho_z = rho[:, 2]

    ra = torch.atan2(rho_y, rho_x)
    # Wrap RA to [0, 2*pi)
    ra = torch.remainder(ra, 2.0 * math.pi)

    rho_xy = torch.sqrt(rho_x ** 2 + rho_y ** 2)
    dec = torch.atan2(rho_z, rho_xy)
    return ra, dec


def compute_differentiable_accel(
    pos: torch.Tensor,
    vel: torch.Tensor,
    ballistic_coeff: torch.Tensor,
    mu: float = MU_EARTH,
    j2: float = J2_EARTH,
    r_e: float = R_EARTH,
    rho0: float = ATMOSPHERE_DEFAULT_RHO0,
    h0: float = ATMOSPHERE_DEFAULT_H0,
    scale_h: float = ATMOSPHERE_DEFAULT_SCALE_HEIGHT,
) -> torch.Tensor:
    """Compute total gravitational (point-mass + J2) and aerodynamic drag acceleration."""
    r_norm = torch.norm(pos, dim=-1, keepdim=True)
    r_val = r_norm.squeeze(-1)

    # Point-mass gravity
    a_grav = -mu / (r_norm ** 3) * pos

    # J2 Geopotential perturbation
    z = pos[:, 2:3]
    z_ratio = z / r_norm
    j2_factor = -1.5 * mu * j2 * (r_e ** 2) / (r_norm ** 5)

    a_j2_x = j2_factor * pos[:, 0:1] * (1.0 - 5.0 * (z_ratio ** 2))
    a_j2_y = j2_factor * pos[:, 1:2] * (1.0 - 5.0 * (z_ratio ** 2))
    a_j2_z = j2_factor * pos[:, 2:3] * (3.0 - 5.0 * (z_ratio ** 2))
    a_j2 = torch.cat([a_j2_x, a_j2_y, a_j2_z], dim=-1)

    # Atmospheric drag with rotating atmosphere
    omega_vec = torch.tensor([0.0, 0.0, OMEGA_EARTH], device=pos.device, dtype=pos.dtype)
    v_atm = torch.cross(omega_vec.expand_as(pos), pos, dim=-1)
    v_rel = vel - v_atm
    v_rel_norm = torch.norm(v_rel, dim=-1, keepdim=True)

    alt = r_norm - r_e
    # Clamp altitude to avoid numerical underflow/overflow
    alt_clamped = torch.clamp(alt, min=100_000.0, max=1_000_000.0)
    density = rho0 * torch.exp(-(alt_clamped - h0) / scale_h)

    # Drag: -0.5 * rho * B * |v_rel| * v_rel
    b_val = ballistic_coeff.view(1, 1)
    a_drag = -0.5 * density * b_val * v_rel_norm * v_rel

    return a_grav + a_j2 + a_drag


def propagate_rk4_differentiable(
    r0: torch.Tensor,
    v0: torch.Tensor,
    b_coeff: torch.Tensor,
    time_steps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Propagate trajectory sequentially via differentiable RK4 integration."""
    positions = [r0.unsqueeze(0)]
    velocities = [v0.unsqueeze(0)]

    curr_r = r0.unsqueeze(0)
    curr_v = v0.unsqueeze(0)

    for i in range(len(time_steps) - 1):
        dt = time_steps[i + 1] - time_steps[i]

        k1_r = curr_v
        k1_v = compute_differentiable_accel(curr_r, curr_v, b_coeff)

        k2_r = curr_v + 0.5 * dt * k1_v
        k2_v = compute_differentiable_accel(curr_r + 0.5 * dt * k1_r, curr_v + 0.5 * dt * k1_v, b_coeff)

        k3_r = curr_v + 0.5 * dt * k2_v
        k3_v = compute_differentiable_accel(curr_r + 0.5 * dt * k2_r, curr_v + 0.5 * dt * k2_v, b_coeff)

        k4_r = curr_v + dt * k3_v
        k4_v = compute_differentiable_accel(curr_r + dt * k3_r, curr_v + dt * k3_v, b_coeff)

        curr_r = curr_r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        curr_v = curr_v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        positions.append(curr_r)
        velocities.append(curr_v)

    pos_traj = torch.cat(positions, dim=0)
    vel_traj = torch.cat(velocities, dim=0)
    return pos_traj, vel_traj


class PINNOrbitDeterminator:
    """Physics-Informed Orbit Determinator (PINN-OD) for LEO space debris."""

    def __init__(
        self,
        station_lat_deg: float = 28.5729,  # Cape Canaveral default
        station_lon_deg: float = -80.6490,
        station_alt_m: float = 10.0,
    ):
        self.lat_deg = station_lat_deg
        self.lon_deg = station_lon_deg
        self.alt_m = station_alt_m

    def generate_synthetic_observations(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        b_coeff: float,
        epochs_sec: np.ndarray,
        noise_std_rad: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Generate ground-truth synthetic topocentric angles for testing."""
        t_t = torch.tensor(epochs_sec, dtype=torch.float64)
        r0_t = torch.tensor(r0, dtype=torch.float64)
        v0_t = torch.tensor(v0, dtype=torch.float64)
        b_t = torch.tensor(b_coeff, dtype=torch.float64)

        with torch.no_grad():
            pos, vel = propagate_rk4_differentiable(r0_t, v0_t, b_t, t_t)
            station_pos = ground_station_eci(self.lat_deg, self.lon_deg, self.alt_m, t_t)
            ra, dec = topocentric_ra_dec(pos, station_pos)

        ra_np = ra.cpu().numpy()
        dec_np = dec.cpu().numpy()

        if noise_std_rad > 0:
            ra_np += np.random.normal(0, noise_std_rad, size=ra_np.shape)
            dec_np += np.random.normal(0, noise_std_rad, size=dec_np.shape)

        return {
            "t": epochs_sec,
            "ra_rad": ra_np,
            "dec_rad": dec_np,
            "true_r": pos.cpu().numpy(),
            "true_v": vel.cpu().numpy(),
            "true_b": b_coeff,
        }

    def fit(
        self,
        epochs_sec: np.ndarray,
        observed_ra_rad: np.ndarray,
        observed_dec_rad: np.ndarray,
        initial_guess_r0: np.ndarray,
        initial_guess_v0: np.ndarray,
        initial_guess_b: float = 0.01,
        iterations: int = 150,
        learning_rate: float = 1e-3,
    ) -> Dict[str, Any]:
        """Solve the inverse problem: optimize [r0, v0, B] to match observed angles."""
        t_tensor = torch.tensor(epochs_sec, dtype=torch.float64)
        ra_obs = torch.tensor(observed_ra_rad, dtype=torch.float64)
        dec_obs = torch.tensor(observed_dec_rad, dtype=torch.float64)

        station_pos = ground_station_eci(self.lat_deg, self.lon_deg, self.alt_m, t_tensor)

        # Normalization scale factors for well-conditioned optimization
        pos_scale = R_EARTH
        vel_scale = 8000.0

        r0_norm = torch.tensor(initial_guess_r0 / pos_scale, dtype=torch.float64, requires_grad=True)
        v0_norm = torch.tensor(initial_guess_v0 / vel_scale, dtype=torch.float64, requires_grad=True)
        log10_b = torch.tensor(math.log10(max(1e-6, initial_guess_b)), dtype=torch.float64, requires_grad=True)

        optimizer = optim.Adam([r0_norm, v0_norm, log10_b], lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=learning_rate * 0.05)
        loss_history: List[float] = []

        best_loss = float("inf")
        best_r0_norm = r0_norm.detach().clone()
        best_v0_norm = v0_norm.detach().clone()
        best_log10_b = log10_b.detach().clone()

        for _ in range(iterations):
            optimizer.zero_grad()
            curr_r0 = r0_norm * pos_scale
            curr_v0 = v0_norm * vel_scale
            curr_b = 10.0 ** log10_b

            pred_pos, pred_vel = propagate_rk4_differentiable(curr_r0, curr_v0, curr_b, t_tensor)
            pred_ra, pred_dec = topocentric_ra_dec(pred_pos, station_pos)

            # Continuous angular distance loss avoiding 2*pi jump
            ra_diff = torch.atan2(torch.sin(pred_ra - ra_obs), torch.cos(pred_ra - ra_obs))
            dec_diff = pred_dec - dec_obs

            loss_ra = torch.mean(ra_diff ** 2)
            loss_dec = torch.mean(dec_diff ** 2)
            loss = loss_ra + loss_dec

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = float(loss.item())
            loss_history.append(loss_val)
            if loss_val < best_loss:
                best_loss = loss_val
                best_r0_norm = r0_norm.detach().clone()
                best_v0_norm = v0_norm.detach().clone()
                best_log10_b = log10_b.detach().clone()

        with torch.no_grad():
            final_r0 = (best_r0_norm * pos_scale).cpu().numpy()
            final_v0 = (best_v0_norm * vel_scale).cpu().numpy()
            final_b = float((10.0 ** best_log10_b).item())
            final_pos, final_vel = propagate_rk4_differentiable(
                best_r0_norm * pos_scale, best_v0_norm * vel_scale, 10.0 ** best_log10_b, t_tensor
            )

        return {
            "estimated_r0": final_r0,
            "estimated_v0": final_v0,
            "estimated_b": final_b,
            "final_loss": loss_history[-1],
            "loss_history": loss_history,
            "fitted_pos": final_pos.cpu().numpy(),
            "fitted_vel": final_vel.cpu().numpy(),
        }
