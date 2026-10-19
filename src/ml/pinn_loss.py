"""Differentiable physics-informed residual loss and Symplectic Hamiltonian manifold projection."""

from __future__ import annotations

import math
import numpy as np
import torch


def hamiltonian_manifold_project(
    r_next_si: np.ndarray,
    v_nn_si: np.ndarray,
    e_target_j_kg: float,
    mu: float = 3.986004418e14,
) -> np.ndarray:
    """Project predicted velocity `v_nn_si` onto the exact Hamiltonian energy manifold `e_target_j_kg`:

        v_target = sqrt(2 * (E_target + mu / ||r_{k+1}||))
        v_{k+1}  = v_NN * (v_target / ||v_NN||)
    """
    r_norm = float(np.linalg.norm(r_next_si))
    v_norm = float(np.linalg.norm(v_nn_si))
    if r_norm <= 1.0 or v_norm <= 1e-12:
        return np.asarray(v_nn_si, dtype=np.float64)
    two_term = 2.0 * (float(e_target_j_kg) + float(mu) / r_norm)
    if two_term <= 0.0:
        return np.asarray(v_nn_si, dtype=np.float64)
    v_target = math.sqrt(two_term)
    return np.asarray(v_nn_si, dtype=np.float64) * (v_target / v_norm)


def pinn_physics_hamiltonian_loss(
    state_in_nd: torch.Tensor,
    state_pred_nd: torch.Tensor,
    state_true_nd: torch.Tensor,
    dt_nd: float = 0.01239,
    j2: float = 1.08262668e-3,
    lambda_phys: float = 0.15,
    lambda_ham: float = 0.25,
) -> torch.Tensor:
    """Compute composite PINN loss: Data MSE + Canonical J2 ODE Residual + Hamiltonian Energy Conservation."""
    mse_data = torch.mean((state_pred_nd - state_true_nd) ** 2)

    r_in = state_in_nd[:, 0:3]
    v_in = state_in_nd[:, 3:6]
    r_pred = state_pred_nd[:, 0:3]
    v_pred = state_pred_nd[:, 3:6]

    # Canonical two-body + J2 acceleration at midpoint
    r_mid = 0.5 * (r_in + r_pred)
    r_mag = torch.clamp(torch.linalg.norm(r_mid, dim=1, keepdim=True), min=0.5)
    z_r2 = (r_mid[:, 2:3] / r_mag) ** 2

    a_kepler = -r_mid / (r_mag**3)
    f_j2 = -1.5 * j2 / (r_mag**5)
    a_j2_xy = f_j2 * r_mid[:, 0:2] * (1.0 - 5.0 * z_r2)
    a_j2_z = f_j2 * r_mid[:, 2:3] * (3.0 - 5.0 * z_r2)
    a_tot = a_kepler + torch.cat([a_j2_xy, a_j2_z], dim=1)

    res_r = (r_pred - r_in) - 0.5 * (v_in + v_pred) * dt_nd
    res_v = (v_pred - v_in) - a_tot * dt_nd
    loss_phys = torch.mean(res_r**2 + res_v**2)

    # Canonical specific mechanical energy H = 0.5 * ||v*||^2 - 1 / ||r*||
    r_pred_mag = torch.clamp(torch.linalg.norm(r_pred, dim=1), min=0.5)
    r_true_mag = torch.clamp(torch.linalg.norm(state_true_nd[:, 0:3], dim=1), min=0.5)
    e_pred = 0.5 * torch.sum(v_pred**2, dim=1) - (1.0 / r_pred_mag)
    e_true = 0.5 * torch.sum(state_true_nd[:, 3:6] ** 2, dim=1) - (1.0 / r_true_mag)
    loss_ham = torch.mean((e_pred - e_true) ** 2)

    return mse_data + lambda_phys * loss_phys + lambda_ham * loss_ham
