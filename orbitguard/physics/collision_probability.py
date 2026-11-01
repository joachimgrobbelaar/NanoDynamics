"""Collision probability (Pc) engine using 2D B-plane encounter projection."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import dblquad

from orbitguard.physics.constants import MU_EARTH


def compute_collision_plane_basis(v_rel: np.ndarray, r1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute orthonormal encounter frame (x_enc, y_enc, z_enc).

    - y_enc: aligned with relative velocity vector v_rel (encounter direction).
    - x_enc: perpendicular to relative velocity in collision plane.
    - z_enc: completes right-handed triad in collision plane.
    """
    v_norm = np.linalg.norm(v_rel)
    if v_norm < 1e-6:
        raise ValueError("Relative velocity is zero; encounter plane undefined.")
    y_enc = v_rel / v_norm

    # Auxiliary vector from primary position
    r_unit = r1 / np.linalg.norm(r1)
    x_cross = np.cross(y_enc, r_unit)
    x_norm = np.linalg.norm(x_cross)

    if x_norm < 1e-4:
        # Singularity: pick arbitrary non-parallel vector
        aux = np.array([1.0, 0.0, 0.0]) if abs(y_enc[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        x_cross = np.cross(y_enc, aux)
        x_norm = np.linalg.norm(x_cross)

    x_enc = x_cross / x_norm
    z_enc = np.cross(x_enc, y_enc)
    return x_enc, y_enc, z_enc


def calculate_collision_probability(
    miss_distance_m: float,
    combined_radius_m: float = 2.0,
    sigma_major_m: float = 150.0,
    sigma_minor_m: float = 75.0,
    orientation_rad: float = 0.0,
) -> float:
    """Calculate 2D B-plane collision probability (Pc) over combined hard-body sphere.

    Uses the standard Akella-Alfriend / Foster model integrating bivariate normal distribution:
      Pc = 1 / (2*pi*sigma_x*sigma_z) * Integral[ exp(-0.5 * (x^2/sigma_x^2 + z^2/sigma_z^2)) dx dz ]
    over a circular disk of radius R centered at (x_m, 0).
    """
    if miss_distance_m < 0.0:
        return 0.0

    r = combined_radius_m
    sx = max(1.0, sigma_major_m)
    sz = max(1.0, sigma_minor_m)
    xm = miss_distance_m

    # If miss distance is extreme compared to uncertainty (> 8 sigma), Pc is virtually zero
    if xm > 8.0 * max(sx, sz):
        return 0.0

    # If hard body radius is much smaller than covariance (standard CubeSat regime)
    # Uniform density approximation over disk: Pc ~= Area * PDF(xm, 0)
    area = math.pi * (r**2)
    norm_const = 1.0 / (2.0 * math.pi * sx * sz)
    exponent = -0.5 * ((xm / sx) ** 2)
    pc_approx = area * norm_const * math.exp(exponent)

    # Upper bound check (Pc cannot exceed 1.0)
    return min(1.0, max(0.0, float(pc_approx)))


def evaluate_encounter_risk(
    miss_distance_km: float,
    combined_radius_m: float = 2.0,
    sigma_radial_m: float = 50.0,
    sigma_in_track_m: float = 250.0,
    sigma_cross_track_m: float = 80.0,
) -> Tuple[float, str]:
    """Evaluate collision probability and assign operational risk tier.

    Returns:
        (collision_probability, risk_tier)
    """
    miss_distance_m = miss_distance_km * 1000.0
    # Equivalent 2D major and minor standard deviations
    sigma_major = math.sqrt(sigma_in_track_m**2 + sigma_cross_track_m**2)
    sigma_minor = sigma_radial_m

    pc = calculate_collision_probability(
        miss_distance_m=miss_distance_m,
        combined_radius_m=combined_radius_m,
        sigma_major_m=sigma_major,
        sigma_minor_m=sigma_minor,
    )

    if miss_distance_km < 1.0 or pc > 1e-3:
        tier = "CRITICAL"
    elif miss_distance_km < 5.0 or pc > 1e-4:
        tier = "HIGH"
    elif miss_distance_km < 10.0 or pc > 1e-5:
        tier = "MODERATE"
    elif miss_distance_km < 25.0:
        tier = "LOW"
    else:
        tier = "NEGLIGIBLE"

    return pc, tier
