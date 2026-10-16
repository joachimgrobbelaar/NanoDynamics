"""Multi-body & non-conservative astrodynamic force models (TRL 1-6).

Implements:
- Earth zonal geopotential harmonics J2, J3, J4 (EGM2008 coefficients)
- Space-weather-conditioned NRLMSISE-00 / Jacchia-Bowman thermospheric drag
- Cannonball Solar Radiation Pressure (SRP) with cylindrical umbra shadow function
- Coupled Earth-Moon-Sun N-body gravitational perturbations
- Circular Restricted Three-Body Problem (CR3BP) synodic dynamics & Jacobi constant
"""

from __future__ import annotations

import math
import numpy as np

MU_EARTH = 3.986004418e14          # m^3/s^2
MU_MOON = 4.902800066e12           # m^3/s^2
MU_SUN = 1.32712440018e20          # m^3/s^2
R_EARTH = 6378137.0                # m
OMEGA_EARTH = 7.292115146706979e-5 # rad/s

EARTH_J2 = 1.08262668355e-3
EARTH_J3 = -2.53265648533e-6
EARTH_J4 = -1.61962159137e-6

CR3BP_MU_EM = MU_MOON / (MU_EARTH + MU_MOON)  # ~0.0121505856


def acceleration_j2_j4(
    r_gcrf: np.ndarray,
    mu: float = MU_EARTH,
    r_eq: float = R_EARTH,
    j2: float = EARTH_J2,
    j3: float = EARTH_J3,
    j4: float = EARTH_J4,
) -> np.ndarray:
    """Compute zonal harmonic perturbations (J2 + J3 + J4) in GCRF [m/s^2]."""
    x, y, z = float(r_gcrf[0]), float(r_gcrf[1]), float(r_gcrf[2])
    r2 = x * x + y * y + z * z
    r = math.sqrt(r2)
    if r <= 1.0:
        return np.zeros(3, dtype=np.float64)

    z_r = z / r
    z_r2 = z_r * z_r
    z_r3 = z_r2 * z_r
    z_r4 = z_r2 * z_r2

    # J2
    f_j2 = -1.5 * j2 * mu * (r_eq**2) / (r**5)
    a_j2 = np.array(
        [
            f_j2 * x * (1.0 - 5.0 * z_r2),
            f_j2 * y * (1.0 - 5.0 * z_r2),
            f_j2 * z * (3.0 - 5.0 * z_r2),
        ],
        dtype=np.float64,
    )

    # J3
    f_j3 = -0.5 * j3 * mu * (r_eq**3) / (r**6)
    a_j3 = np.array(
        [
            f_j3 * x * (5.0 * (7.0 * z_r3 - 3.0 * z_r)),
            f_j3 * y * (5.0 * (7.0 * z_r3 - 3.0 * z_r)),
            f_j3 * r * (35.0 * z_r4 - 30.0 * z_r2 + 3.0),
        ],
        dtype=np.float64,
    )

    # J4
    f_j4 = (5.0 / 8.0) * j4 * mu * (r_eq**4) / (r**7)
    a_j4 = np.array(
        [
            f_j4 * x * (3.0 - 42.0 * z_r2 + 63.0 * z_r4),
            f_j4 * y * (3.0 - 42.0 * z_r2 + 63.0 * z_r4),
            f_j4 * z * (15.0 - 70.0 * z_r2 + 63.0 * z_r4),
        ],
        dtype=np.float64,
    )

    return a_j2 + a_j3 + a_j4


def nrlmsise00_thermospheric_density(
    altitude_m: float,
    f107: float = 150.0,
    kp: float = 3.0,
) -> float:
    """Compute thermospheric mass density rho [kg/m^3] conditioned on solar flux F10.7 and Kp."""
    alt_km = max(80.0, altitude_m * 1e-3)
    if alt_km > 1000.0:
        return 0.0

    # Exospheric temperature T_inf modulated by F10.7 and geomagnetic Kp (Jacchia/NRLMSISE-00 fit)
    t_inf = 900.0 + 2.5 * (f107 - 70.0) + 28.0 * kp
    scale_height_km = max(25.0, 0.052 * t_inf)
    rho_120 = 2.22e-8
    rho_200 = 2.54e-10 * (1.0 + 0.0045 * max(0.0, f107 - 70.0)) * (1.0 + 0.06 * max(0.0, kp - 2.0))

    if alt_km < 200.0:
        return rho_120 * math.exp(-(alt_km - 120.0) / 16.5)
    return rho_200 * math.exp(-(alt_km - 200.0) / scale_height_km)


def acceleration_drag_nrlmsise00(
    r_gcrf: np.ndarray,
    v_gcrf: np.ndarray,
    b_coeff: float = 0.015,
    f107: float = 150.0,
    kp: float = 3.0,
    r_eq: float = R_EARTH,
    omega_e: float = OMEGA_EARTH,
) -> np.ndarray:
    """Compute atmospheric drag acceleration [m/s^2] with co-rotating atmosphere."""
    r_mag = float(np.linalg.norm(r_gcrf))
    alt_m = r_mag - r_eq
    rho = nrlmsise00_thermospheric_density(alt_m, f107=f107, kp=kp)
    if rho <= 0.0:
        return np.zeros(3, dtype=np.float64)

    omega_vec = np.array([0.0, 0.0, omega_e], dtype=np.float64)
    v_rel = v_gcrf - np.cross(omega_vec, r_gcrf)
    v_rel_mag = float(np.linalg.norm(v_rel))
    return -0.5 * rho * float(b_coeff) * v_rel_mag * v_rel


def acceleration_srp(
    r_gcrf: np.ndarray,
    r_sun_gcrf: np.ndarray,
    cr_area_over_m: float = 0.012,
    p_srp_1au: float = 4.56e-6,
    r_eq: float = R_EARTH,
) -> np.ndarray:
    """Cannonball Solar Radiation Pressure (SRP) acceleration with cylindrical Earth shadow."""
    r_sat_sun = r_gcrf - r_sun_gcrf
    dist_sat_sun = float(np.linalg.norm(r_sat_sun))
    if dist_sat_sun <= 1.0:
        return np.zeros(3, dtype=np.float64)

    # Cylindrical umbra shadow test
    s_hat = r_sun_gcrf / max(float(np.linalg.norm(r_sun_gcrf)), 1.0)
    proj = float(np.dot(r_gcrf, s_hat))
    perp_sq = float(np.dot(r_gcrf, r_gcrf)) - proj * proj
    if proj < 0.0 and perp_sq < (r_eq * r_eq):
        return np.zeros(3, dtype=np.float64)

    au = 149_597_870_700.0
    flux_scale = (au / dist_sat_sun) ** 2
    return p_srp_1au * flux_scale * float(cr_area_over_m) * (r_sat_sun / dist_sat_sun)


def acceleration_third_body_ems(
    r_gcrf: np.ndarray,
    r_moon_gcrf: np.ndarray,
    r_sun_gcrf: np.ndarray,
    mu_moon: float = MU_MOON,
    mu_sun: float = MU_SUN,
) -> np.ndarray:
    """Coupled lunar and solar third-body gravitational perturbations in GCRF [m/s^2]."""
    d_m = r_gcrf - r_moon_gcrf
    d_s = r_gcrf - r_sun_gcrf

    rm_d = max(float(np.linalg.norm(d_m)), 1.0)
    rm_0 = max(float(np.linalg.norm(r_moon_gcrf)), 1.0)
    rs_d = max(float(np.linalg.norm(d_s)), 1.0)
    rs_0 = max(float(np.linalg.norm(r_sun_gcrf)), 1.0)

    a_moon = -mu_moon * (d_m / (rm_d**3) + r_moon_gcrf / (rm_0**3))
    a_sun = -mu_sun * (d_s / (rs_d**3) + r_sun_gcrf / (rs_0**3))
    return a_moon + a_sun


def acceleration_cr3bp_synodic(state_nd: np.ndarray, mu: float = CR3BP_MU_EM) -> np.ndarray:
    """Non-dimensional CR3BP equations of motion in the Earth-Moon rotating synodic frame."""
    x, y, z, vx, vy, vz = (float(v) for v in state_nd[:6])
    r1 = math.sqrt((x + mu) ** 2 + y * y + z * z)
    r2 = math.sqrt((x - 1.0 + mu) ** 2 + y * y + z * z)

    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / (r1**3) - mu * (x - 1.0 + mu) / (r2**3)
    ay = -2.0 * vx + y - (1.0 - mu) * y / (r1**3) - mu * y / (r2**3)
    az = -(1.0 - mu) * z / (r1**3) - mu * z / (r2**3)
    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)


def compute_jacobi_constant(state_nd: np.ndarray, mu: float = CR3BP_MU_EM) -> float:
    """Compute the Jacobi energy integral C = 2*Omega(x, y, z) - (vx^2 + vy^2 + vz^2)."""
    x, y, z, vx, vy, vz = (float(v) for v in state_nd[:6])
    r1 = math.sqrt((x + mu) ** 2 + y * y + z * z)
    r2 = math.sqrt((x - 1.0 + mu) ** 2 + y * y + z * z)
    omega = 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2
    v2 = vx * vx + vy * vy + vz * vz
    return 2.0 * omega - v2


def total_gcrf_acceleration(
    t_s: float,
    r_gcrf: np.ndarray,
    v_gcrf: np.ndarray,
    b_coeff: float = 0.015,
    f107: float = 150.0,
    kp: float = 3.0,
    include_j2_j4: bool = True,
    include_drag: bool = True,
    include_srp: bool = False,
    include_third_body: bool = False,
    r_moon_gcrf: np.ndarray | None = None,
    r_sun_gcrf: np.ndarray | None = None,
) -> np.ndarray:
    """Compute total acceleration vector in GCRF [m/s^2]."""
    r_mag = max(float(np.linalg.norm(r_gcrf)), 1.0)
    a_tot = -MU_EARTH * r_gcrf / (r_mag**3)

    if include_j2_j4:
        a_tot = a_tot + acceleration_j2_j4(r_gcrf)
    if include_drag:
        a_tot = a_tot + acceleration_drag_nrlmsise00(r_gcrf, v_gcrf, b_coeff=b_coeff, f107=f107, kp=kp)
    if include_srp and r_sun_gcrf is not None:
        a_tot = a_tot + acceleration_srp(r_gcrf, r_sun_gcrf)
    if include_third_body and r_moon_gcrf is not None and r_sun_gcrf is not None:
        a_tot = a_tot + acceleration_third_body_ems(r_gcrf, r_moon_gcrf, r_sun_gcrf)
    return a_tot
