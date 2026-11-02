"""Nanosatellite orbital mechanics and astrodynamics coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

# Standard WGS-84 and IERS Earth Physical Constants
MU_EARTH: float = 3.986004418e14        # Earth gravitational parameter [m^3/s^2]
R_EARTH: float = 6378137.0              # Earth equatorial radius [m]
J2_EARTH: float = 1.08262668e-3         # Earth second zonal harmonic [-]
OMEGA_EARTH: float = 7.2921150e-5       # Earth rotational angular velocity [rad/s]
RHO_0: float = 1.225                    # Sea level atmospheric density [kg/m^3]
SCALE_HEIGHT_LEO: float = 37500.0       # Effective thermospheric scale height [m]


@dataclass
class OrbitalElements:
    """Classical Keplerian Orbital Elements (COE)."""

    a: float               # Semi-major axis [m]
    e: float               # Eccentricity [-]
    i: float               # Inclination [rad]
    raan: float            # Right Ascension of Ascending Node [rad]
    arg_pe: float          # Argument of Periapsis [rad]
    true_anomaly: float    # True Anomaly [rad]
    epoch: float = 0.0     # Time epoch [s]

    @property
    def perigee_altitude(self) -> float:
        """Altitude at perigee [m] relative to Earth equatorial radius."""
        return self.a * (1.0 - self.e) - R_EARTH

    @property
    def apogee_altitude(self) -> float:
        """Altitude at apogee [m] relative to Earth equatorial radius."""
        return self.a * (1.0 + self.e) - R_EARTH

    @property
    def period(self) -> float:
        """Orbital period in seconds."""
        if self.a <= 0:
            return 0.0
        return 2.0 * math.pi * math.sqrt(self.a**3 / MU_EARTH)

    @property
    def mean_motion(self) -> float:
        """Mean motion in rad/s."""
        if self.a <= 0:
            return 0.0
        return math.sqrt(MU_EARTH / (self.a**3))

    def to_dict(self) -> dict:
        return {
            "a": float(self.a),
            "e": float(self.e),
            "i": float(self.i),
            "i_deg": float(math.degrees(self.i)),
            "raan": float(self.raan),
            "raan_deg": float(math.degrees(self.raan)),
            "arg_pe": float(self.arg_pe),
            "arg_pe_deg": float(math.degrees(self.arg_pe)),
            "true_anomaly": float(self.true_anomaly),
            "true_anomaly_deg": float(math.degrees(self.true_anomaly)),
            "perigee_altitude_km": float(self.perigee_altitude / 1000.0),
            "apogee_altitude_km": float(self.apogee_altitude / 1000.0),
            "period_minutes": float(self.period / 60.0),
            "epoch": float(self.epoch),
        }


def coe_to_rv(coe: OrbitalElements, mu: float = MU_EARTH) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Classical Orbital Elements to Cartesian ECI state vectors (r, v)."""
    p = coe.a * (1.0 - coe.e**2)
    nu = coe.true_anomaly
    cos_nu = math.cos(nu)
    sin_nu = math.sin(nu)

    denom = 1.0 + coe.e * cos_nu
    if abs(denom) < 1e-12:
        denom = 1e-12

    # Position and velocity in perifocal frame (PQW)
    r_pqw = np.array([
        (p * cos_nu) / denom,
        (p * sin_nu) / denom,
        0.0,
    ], dtype=np.float64)

    sqrt_mu_p = math.sqrt(mu / max(p, 1e-6))
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

    r_eci = np.array([
        P11 * r_pqw[0] + P12 * r_pqw[1],
        P21 * r_pqw[0] + P22 * r_pqw[1],
        P31 * r_pqw[0] + P32 * r_pqw[1],
    ], dtype=np.float64)

    v_eci = np.array([
        P11 * v_pqw[0] + P12 * v_pqw[1],
        P21 * v_pqw[0] + P22 * v_pqw[1],
        P31 * v_pqw[0] + P32 * v_pqw[1],
    ], dtype=np.float64)

    return r_eci, v_eci


def rv_to_coe(r: np.ndarray, v: np.ndarray, mu: float = MU_EARTH) -> OrbitalElements:
    """Convert Cartesian ECI state vectors (r, v) to Classical Orbital Elements."""
    r_vec = np.asarray(r, dtype=np.float64)
    v_vec = np.asarray(v, dtype=np.float64)

    r_norm = float(np.linalg.norm(r_vec))
    v_norm = float(np.linalg.norm(v_vec))

    # Angular momentum vector
    h_vec = np.cross(r_vec, v_vec)
    h_norm = float(np.linalg.norm(h_vec))

    # Node vector
    k_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_vec = np.cross(k_unit, h_vec)
    n_norm = float(np.linalg.norm(n_vec))

    # Specific energy & Semi-major axis
    energy = 0.5 * (v_norm**2) - (mu / r_norm)
    if abs(energy) < 1e-12:
        a = float("inf")
    else:
        a = -mu / (2.0 * energy)

    # Eccentricity vector
    e_vec = ((v_norm**2 - mu / r_norm) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = float(np.linalg.norm(e_vec))

    # Inclination
    i = math.acos(np.clip(h_vec[2] / max(h_norm, 1e-12), -1.0, 1.0))

    # RAAN
    if n_norm > 1e-8:
        raan = math.acos(np.clip(n_vec[0] / n_norm, -1.0, 1.0))
        if n_vec[1] < 0.0:
            raan = 2.0 * math.pi - raan
    else:
        raan = 0.0

    # Argument of Periapsis
    if n_norm > 1e-8 and e > 1e-6:
        arg_pe = math.acos(np.clip(np.dot(n_vec, e_vec) / (n_norm * e), -1.0, 1.0))
        if e_vec[2] < 0.0:
            arg_pe = 2.0 * math.pi - arg_pe
    else:
        arg_pe = 0.0

    # True Anomaly
    if e > 1e-6:
        true_anom = math.acos(np.clip(np.dot(e_vec, r_vec) / (e * r_norm), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0.0:
            true_anom = 2.0 * math.pi - true_anom
    else:
        true_anom = 0.0

    return OrbitalElements(
        a=a,
        e=e,
        i=i,
        raan=raan,
        arg_pe=arg_pe,
        true_anomaly=true_anom,
    )


def compute_vnc_basis(r: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute orthonormal Velocity-Normal-Cross (VNC / RTN) coordinate frame basis vectors.

    - v_hat: aligned along velocity vector (In-track / Tangential)
    - n_hat: aligned with orbital angular momentum (Cross-track / Out-of-plane)
    - c_hat: completes right-handed triad = v_hat x n_hat (Radial-like)
    """
    v_norm = float(np.linalg.norm(v))
    v_hat = v / max(v_norm, 1e-12)

    h_vec = np.cross(r, v)
    h_norm = float(np.linalg.norm(h_vec))
    n_hat = h_vec / max(h_norm, 1e-12)

    c_hat = np.cross(v_hat, n_hat)
    return v_hat, n_hat, c_hat


def eci_to_ecef(r_eci: np.ndarray, epoch_seconds: float) -> np.ndarray:
    """Rotate ECI position vector to ECEF frame assuming uniform Earth rotation."""
    theta = OMEGA_EARTH * epoch_seconds
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    R_z = np.array([
        [cos_t, sin_t, 0.0],
        [-sin_t, cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return R_z @ r_eci


def ecef_to_geodetic(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    """Convert ECEF coordinates (x, y, z) [m] to geodetic latitude [deg], longitude [deg], altitude [m]."""
    x, y, z = float(r_ecef[0]), float(r_ecef[1]), float(r_ecef[2])
    lon = math.atan2(y, x)

    # WGS-84 ellipsoid parameters
    a_wgs84 = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f

    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(5):
        n = a_wgs84 / math.sqrt(1.0 - e2 * (math.sin(lat) ** 2))
        lat = math.atan2(z + e2 * n * math.sin(lat), p)

    n = a_wgs84 / math.sqrt(1.0 - e2 * (math.sin(lat) ** 2))
    alt = p / math.cos(lat) - n if abs(math.cos(lat)) > 1e-6 else z / math.sin(lat) - n * (1.0 - e2)

    return math.degrees(lat), math.degrees(lon), alt
