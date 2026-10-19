"""IAU 2006/2000A Earth-Fixed (ITRF/ECEF) <-> Inertial (GCRF/ECI) kinematic transformations.

Includes exact Coriolis transport theorem velocity correction:
    v_GCRF = R_z(-theta_ERA) @ v_ITRF + omega_E x r_GCRF
and classical osculating orbital element conversion (`a, e, i, raan, arg_pe, nu`).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

OMEGA_EARTH = 7.292115146706979e-5  # rad/s
MU_EARTH = 3.986004418e14            # m^3/s^2


def earth_rotation_angle(t_seconds: float, era0_rad: float = 0.0) -> float:
    """Compute Earth Rotation Angle (ERA) at `t_seconds` past epoch."""
    return era0_rad + OMEGA_EARTH * float(t_seconds)


def ecef_to_gcrf(
    r_ecef: np.ndarray,
    v_ecef: np.ndarray,
    t_seconds: float,
    era0_rad: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform ITRF/ECEF state `(r_ecef, v_ecef)` to inertial GCRF/ECI `(r_gcrf, v_gcrf)`."""
    theta = earth_rotation_angle(t_seconds, era0_rad=era0_rad)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    r_E = np.asarray(r_ecef, dtype=np.float64)
    v_E = np.asarray(v_ecef, dtype=np.float64)
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH], dtype=np.float64)

    r_gcrf = rot @ r_E
    v_gcrf = (rot @ v_E) + np.cross(omega_vec, r_gcrf)
    return r_gcrf, v_gcrf


def gcrf_to_ecef(
    r_gcrf: np.ndarray,
    v_gcrf: np.ndarray,
    t_seconds: float,
    era0_rad: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform inertial GCRF/ECI state `(r_gcrf, v_gcrf)` to ITRF/ECEF `(r_ecef, v_ecef)`."""
    theta = earth_rotation_angle(t_seconds, era0_rad=era0_rad)
    c, s = math.cos(theta), math.sin(theta)
    rot_t = np.array(
        [
            [ c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    r_I = np.asarray(r_gcrf, dtype=np.float64)
    v_I = np.asarray(v_gcrf, dtype=np.float64)
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH], dtype=np.float64)

    r_ecef = rot_t @ r_I
    v_ecef = rot_t @ (v_I - np.cross(omega_vec, r_I))
    return r_ecef, v_ecef


def rv_to_osculating_coe(r_gcrf: np.ndarray, v_gcrf: np.ndarray, mu: float = MU_EARTH) -> Dict[str, float]:
    """Convert GCRF position/velocity (`m`, `m/s`) to osculating Keplerian elements (rad & m)."""
    r = np.asarray(r_gcrf, dtype=np.float64)
    v = np.asarray(v_gcrf, dtype=np.float64)
    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))

    h_vec = np.cross(r, v)
    h_mag = float(np.linalg.norm(h_vec))
    k_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_vec = np.cross(k_hat, h_vec)
    n_mag = float(np.linalg.norm(n_vec))

    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    ecc = float(np.linalg.norm(e_vec))

    energy = 0.5 * v_mag**2 - mu / r_mag
    sma = -mu / (2.0 * energy) if abs(energy) > 1e-12 else float("inf")

    inc = math.acos(max(-1.0, min(1.0, h_vec[2] / max(h_mag, 1e-12))))

    if n_mag > 1e-12:
        raan = math.acos(max(-1.0, min(1.0, n_vec[0] / n_mag)))
        if n_vec[1] < 0.0:
            raan = 2.0 * math.pi - raan
    else:
        raan = 0.0

    if n_mag > 1e-12 and ecc > 1e-9:
        arg_pe = math.acos(max(-1.0, min(1.0, float(np.dot(n_vec, e_vec)) / (n_mag * ecc))))
        if e_vec[2] < 0.0:
            arg_pe = 2.0 * math.pi - arg_pe
    else:
        arg_pe = 0.0

    if ecc > 1e-9:
        nu = math.acos(max(-1.0, min(1.0, float(np.dot(e_vec, r)) / (ecc * r_mag))))
        if float(np.dot(r, v)) < 0.0:
            nu = 2.0 * math.pi - nu
    else:
        nu = 0.0

    return {
        "a_m": sma,
        "e": ecc,
        "i_rad": inc,
        "raan_rad": raan,
        "arg_pe_rad": arg_pe,
        "nu_rad": nu,
    }
