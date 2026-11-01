"""Orbital elements conversion and frame transformations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from orbitguard.physics.constants import MU_EARTH


@dataclass
class OrbitalElements:
    """Classical Keplerian Orbital Elements (COE)."""

    a: float          # Semi-major axis [m]
    e: float          # Eccentricity [-]
    i: float          # Inclination [rad]
    raan: float       # Right Ascension of Ascending Node [rad]
    arg_pe: float     # Argument of Periapsis [rad]
    true_anomaly: float # True Anomaly [rad]
    epoch: float = 0.0 # Time epoch [s]

    @property
    def perigee_altitude(self) -> float:
        """Altitude at perigee [m] relative to Earth equatorial radius."""
        from orbitguard.physics.constants import R_EARTH
        return self.a * (1.0 - self.e) - R_EARTH

    @property
    def apogee_altitude(self) -> float:
        """Altitude at apogee [m] relative to Earth equatorial radius."""
        from orbitguard.physics.constants import R_EARTH
        return self.a * (1.0 + self.e) - R_EARTH

    @property
    def period(self) -> float:
        """Orbital period in seconds."""
        return 2.0 * math.pi * math.sqrt(self.a**3 / MU_EARTH)


def coe_to_rv(coe: OrbitalElements, mu: float = MU_EARTH) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Classical Orbital Elements to Cartesian ECI state vectors (r, v)."""
    p = coe.a * (1.0 - coe.e**2)
    nu = coe.true_anomaly
    cos_nu = math.cos(nu)
    sin_nu = math.sin(nu)

    # Position and velocity in perifocal frame (PQW)
    r_pqw = np.array([
        (p * cos_nu) / (1.0 + coe.e * cos_nu),
        (p * sin_nu) / (1.0 + coe.e * cos_nu),
        0.0,
    ], dtype=np.float64)

    sqrt_mu_p = math.sqrt(mu / p)
    v_pqw = np.array([
        -sqrt_mu_p * sin_nu,
        sqrt_mu_p * (coe.e + cos_nu),
        0.0,
    ], dtype=np.float64)

    # Rotation matrix PQW -> ECI: R = R3(-raan) * R1(-i) * R3(-arg_pe)
    cos_o = math.cos(coe.raan)
    sin_o = math.sin(coe.raan)
    cos_i = math.cos(coe.i)
    sin_i = math.sin(coe.i)
    cos_w = math.cos(coe.arg_pe)
    sin_w = math.sin(coe.arg_pe)

    P11 = cos_o * cos_w - sin_o * sin_w * cos_i
    P12 = -cos_o * sin_w - sin_o * cos_w * cos_i
    P13 = sin_o * sin_i

    P21 = sin_o * cos_w + cos_o * sin_w * cos_i
    P22 = -sin_o * sin_w + cos_o * cos_w * cos_i
    P23 = -cos_o * sin_i

    P31 = sin_w * sin_i
    P32 = cos_w * sin_i
    P33 = cos_i

    R_pqw_to_eci = np.array([
        [P11, P12, P13],
        [P21, P22, P23],
        [P31, P32, P33],
    ], dtype=np.float64)

    r_eci = R_pqw_to_eci @ r_pqw
    v_eci = R_pqw_to_eci @ v_pqw
    return r_eci, v_eci


def rv_to_coe(r: np.ndarray, v: np.ndarray, mu: float = MU_EARTH) -> OrbitalElements:
    """Convert Cartesian ECI state vectors (r, v) to Classical Orbital Elements (COE)."""
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))

    # Angular momentum vector
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))

    # Specific mechanical energy
    energy = 0.5 * v_norm**2 - mu / r_norm
    if abs(energy) < 1e-12:
        raise ValueError("Parabolic trajectory detected; cannot compute elliptical COE.")
    a = -mu / (2.0 * energy)

    # Eccentricity vector
    e_vec = ((v_norm**2 - mu / r_norm) * r - np.dot(r, v) * v) / mu
    e = float(np.linalg.norm(e_vec))

    # Inclination
    i = math.acos(max(-1.0, min(1.0, h[2] / h_norm)))

    # Node vector n = k x h
    k_unit = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_unit, h)
    n_norm = float(np.linalg.norm(n_vec))

    # RAAN
    if n_norm > 1e-8:
        raan = math.acos(max(-1.0, min(1.0, n_vec[0] / n_norm)))
        if n_vec[1] < 0:
            raan = 2.0 * math.pi - raan
    else:
        raan = 0.0

    # Argument of periapsis
    if n_norm > 1e-8 and e > 1e-6:
        cos_w = np.dot(n_vec, e_vec) / (n_norm * e)
        arg_pe = math.acos(max(-1.0, min(1.0, cos_w)))
        if e_vec[2] < 0:
            arg_pe = 2.0 * math.pi - arg_pe
    else:
        arg_pe = 0.0

    # True anomaly
    if e > 1e-6:
        cos_nu = np.dot(e_vec, r) / (e * r_norm)
        true_anomaly = math.acos(max(-1.0, min(1.0, cos_nu)))
        if np.dot(r, v) < 0:
            true_anomaly = 2.0 * math.pi - true_anomaly
    else:
        true_anomaly = 0.0

    return OrbitalElements(
        a=a,
        e=e,
        i=i,
        raan=raan,
        arg_pe=arg_pe,
        true_anomaly=true_anomaly,
    )


def compute_vnc_basis(r: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute orthonormal VNC (Velocity, Normal, Co-Normal) basis vectors.

    - V (Tangential/In-track): along velocity vector v.
    - N (Normal/Out-of-plane): along angular momentum h = r x v.
    - C (Co-Normal/Radial): completes the right-handed triad C = V x N.
    """
    v_norm = float(np.linalg.norm(v))
    u_v = v / v_norm

    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    u_n = h / h_norm

    u_c = np.cross(u_v, u_n)
    return u_v, u_n, u_c
