"""Gravitational acceleration models: central two-body gravity and J2 oblateness perturbation.

Coordinates are defined in an Earth-Centered Inertial (ECI) Cartesian frame [x, y, z].
"""


import numpy as np

from leo_simulator.constants import (
    B0_EARTH_SURFACE,
    C22_EARTH,
    J2_EARTH,
    J3_EARTH,
    J4_EARTH,
    K2_EARTH_TIDE,
    LP165P_HARMONICS_4X4,
    MAG_POLE_LON_RAD,
    MAG_POLE_TILT_RAD,
    MU_EARTH,
    MU_MOON,
    MU_SUN,
    OMEGA_EARTH,
    R_EARTH,
    R_MOON,
    R_MOON_ORBIT,
    R_SUN_ORBIT,
    S22_EARTH,
    T_MOON_ORBIT,
    T_SUN_ORBIT,
)


def central_gravity_acceleration(
    r_vec: np.ndarray | list,
    mu: float = MU_EARTH,
) -> np.ndarray:
    """Compute central two-body gravitational acceleration toward Earth center.

    a_grav = - (mu / r^3) * r_vec

    Args:
        r_vec: Position vector in ECI frame [x, y, z] in meters.
        mu: Standard gravitational parameter in m^3 / s^2.

    Returns:
        np.ndarray: Acceleration vector in ECI frame [ax, ay, az] in m / s^2.

    Raises:
        ValueError: If position vector magnitude is non-positive or coordinates are invalid.
    """
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError(f"Gravitational parameter mu must be strictly positive and finite, got {mu}")

    r = np.asarray(r_vec, dtype=np.float64)
    if r.shape != (3,):
        raise ValueError(f"Position vector must have shape (3,), got {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("Position vector coordinates must be finite.")

    r_norm = np.linalg.norm(r)
    if r_norm <= 0.0:
        raise ValueError("Position vector norm must be positive.")

    return -(mu / (r_norm**3)) * r


def j2_perturbation_acceleration(
    r_vec: np.ndarray | list,
    mu: float = MU_EARTH,
    r_earth: float = R_EARTH,
    j2: float = J2_EARTH,
) -> np.ndarray:
    """Compute perturbative acceleration due to Earth's J2 oblateness (second zonal harmonic).

    Standard Cartesian formulation:
        factor = - (3/2) * J2 * mu * R_E^2 / r^5
        a_x = factor * x * (1 - 5 * z^2 / r^2)
        a_y = factor * y * (1 - 5 * z^2 / r^2)
        a_z = factor * z * (3 - 5 * z^2 / r^2)

    Args:
        r_vec: Position vector in ECI frame [x, y, z] in meters.
        mu: Standard gravitational parameter in m^3 / s^2.
        r_earth: Earth equatorial radius in meters.
        j2: J2 harmonic coefficient (dimensionless).

    Returns:
        np.ndarray: Perturbative acceleration vector [ax, ay, az] in m / s^2.

    Raises:
        ValueError: If position vector norm is non-positive or shape is invalid.
    """
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError(f"Gravitational parameter mu must be strictly positive and finite, got {mu}")
    if not np.isfinite(r_earth) or r_earth <= 0.0:
        raise ValueError(f"Earth radius must be strictly positive and finite, got {r_earth}")
    if not np.isfinite(j2):
        raise ValueError(f"J2 harmonic coefficient must be finite, got {j2}")

    r = np.asarray(r_vec, dtype=np.float64)
    if r.shape != (3,):
        raise ValueError(f"Position vector must have shape (3,), got {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("Position vector coordinates must be finite.")

    r_norm = np.linalg.norm(r)
    if r_norm <= 0.0:
        raise ValueError("Position vector norm must be positive.")

    x, y, z = r[0], r[1], r[2]
    r2 = r_norm * r_norm
    z2_over_r2 = (z * z) / r2

    factor = -1.5 * j2 * mu * (r_earth**2) / (r_norm**5)

    ax = factor * x * (1.0 - 5.0 * z2_over_r2)
    ay = factor * y * (1.0 - 5.0 * z2_over_r2)
    az = factor * z * (3.0 - 5.0 * z2_over_r2)

    return np.array([ax, ay, az], dtype=np.float64)


def third_body_acceleration(
    r_sat: np.ndarray | list,
    r_moon: np.ndarray | list,
    mu_moon: float = MU_MOON,
) -> np.ndarray:
    """Compute third-body gravitational perturbation acceleration.

    a_3rd = mu_moon * ( (r_moon - r_sat) / ||r_moon - r_sat||^3 - r_moon / ||r_moon||^3 )

    Args:
        r_sat: Satellite position vector in ECI frame [x, y, z] in meters.
        r_moon: Moon position vector in ECI frame [x, y, z] in meters.
        mu_moon: Moon standard gravitational parameter in m^3 / s^2.

    Returns:
        np.ndarray: Perturbative acceleration vector [ax, ay, az] in m / s^2.

    Raises:
        ValueError: If mu_moon is non-positive/non-finite, shapes are invalid,
            coordinates are non-finite, or vectors are degenerate.
    """
    if not np.isfinite(mu_moon) or mu_moon <= 0.0:
        raise ValueError(f"Moon gravitational parameter mu_moon must be strictly positive and finite, got {mu_moon}")

    r_s = np.asarray(r_sat, dtype=np.float64)
    r_m = np.asarray(r_moon, dtype=np.float64)
    if r_s.shape != (3,):
        raise ValueError(f"Satellite position vector must have shape (3,), got {r_s.shape}")
    if r_m.shape != (3,):
        raise ValueError(f"Moon position vector must have shape (3,), got {r_m.shape}")
    if not np.all(np.isfinite(r_s)):
        raise ValueError("Satellite position vector coordinates must be finite.")
    if not np.all(np.isfinite(r_m)):
        raise ValueError("Moon position vector coordinates must be finite.")

    d = r_m - r_s
    d_norm = np.linalg.norm(d)
    m_norm = np.linalg.norm(r_m)

    if d_norm <= 0.0 or m_norm <= 0.0:
        raise ValueError("Position vectors must have positive norms and not be co-located.")

    return mu_moon * (d / (d_norm**3) - r_m / (m_norm**3))


def moon_position(t: float) -> np.ndarray:
    """Compute approximate Moon position in ECI frame for a given time t in seconds.

    Assumes a circular orbit in the equatorial plane at 384,400 km distance
    with a period of ~27.32 days.

    Args:
        t: Time in seconds from epoch.

    Returns:
        np.ndarray: Position vector [x, y, z] of the Moon in meters.

    Raises:
        ValueError: If t is non-finite.
    """
    if not np.isfinite(t):
        raise ValueError(f"Epoch time t must be finite, got {t}")

    omega = 2 * np.pi / T_MOON_ORBIT
    angle = omega * float(t)

    return np.array([
        R_MOON_ORBIT * np.cos(angle),
        R_MOON_ORBIT * np.sin(angle),
        0.0
    ], dtype=np.float64)


def sun_position(t: float, obliquity_rad: float = np.radians(23.4392911)) -> np.ndarray:
    """Compute approximate Sun position in ECI frame at time t in seconds."""
    if not np.isfinite(t):
        raise ValueError(f"Epoch time t must be finite, got {t}")

    omega_s = 2.0 * np.pi / T_SUN_ORBIT
    lam = omega_s * float(t)
    return np.array([
        R_SUN_ORBIT * np.cos(lam),
        R_SUN_ORBIT * np.sin(lam) * np.cos(obliquity_rad),
        R_SUN_ORBIT * np.sin(lam) * np.sin(obliquity_rad),
    ], dtype=np.float64)


def coupled_ems_acceleration(
    r_sat: np.ndarray | list,
    t: float = 0.0,
    r_earth: np.ndarray | list | None = None,
    r_moon: np.ndarray | list | None = None,
    r_sun: np.ndarray | list | None = None,
    mu_earth: float = MU_EARTH,
    mu_moon: float = MU_MOON,
    mu_sun: float = MU_SUN,
    include_sun: bool = True,
) -> np.ndarray:
    """Compute coupled Earth-Moon-Sun (EMS) N-body gravitational acceleration:

    r_ddot = -mu_E (r - r_E)/||r - r_E||^3
             -mu_M [ (r - r_M)/||r - r_M||^3 + r_M/||r_M||^3 ]
             -mu_S [ (r - r_S)/||r - r_S||^3 + r_S/||r_S||^3 ]
    """
    r = np.asarray(r_sat, dtype=np.float64)
    r_e = np.zeros(3, dtype=np.float64) if r_earth is None else np.asarray(r_earth, dtype=np.float64)
    r_m = moon_position(t) if r_moon is None else np.asarray(r_moon, dtype=np.float64)
    r_s = sun_position(t) if r_sun is None else np.asarray(r_sun, dtype=np.float64)

    d_e = r - r_e
    norm_e = float(np.linalg.norm(d_e))
    if norm_e <= 0.0:
        raise ValueError("Satellite coincides with Earth center.")

    a_earth = -mu_earth * d_e / (norm_e**3)
    a_moon = third_body_acceleration(r, r_m, mu_moon=mu_moon)
    if not include_sun:
        return a_earth + a_moon
    a_sun = third_body_acceleration(r, r_s, mu_moon=mu_sun)
    return a_earth + a_moon + a_sun


def jacobi_constant_synodic(
    r_syn: np.ndarray | list,
    v_syn: np.ndarray | list,
    mu_1: float = MU_EARTH,
    mu_2: float = MU_MOON,
    distance: float = R_MOON_ORBIT,
) -> float:
    """Compute Jacobi Energy integral C in the Earth-Moon synodic barycentric frame.

    Accepts either normalized CR3BP coordinates (when distance == 1.0) or dimensional SI
    synodic coordinates [m, m/s] (when distance == R_MOON_ORBIT):
        C = 2 * Omega(x, y, z) - (vx^2 + vy^2 + vz^2)
    """
    r = np.asarray(r_syn, dtype=np.float64)
    v = np.asarray(v_syn, dtype=np.float64)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("Synodic position and velocity must have shape (3,).")

    x, y, z = float(r[0]), float(r[1]), float(r[2])
    v_sq = float(np.dot(v, v))

    if np.isclose(distance, 1.0):
        # Normalized CR3BP units: mu = mu_2 / (mu_1 + mu_2), n = 1
        mu = mu_2 / (mu_1 + mu_2)
        r1 = np.sqrt((x + mu) ** 2 + y * y + z * z)
        r2 = np.sqrt((x - 1.0 + mu) ** 2 + y * y + z * z)
        omega = 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2
        return float(2.0 * omega - v_sq)

    mu_tot = mu_1 + mu_2
    pi_2 = mu_2 / mu_tot
    x1 = -pi_2 * distance
    x2 = (1.0 - pi_2) * distance
    n = np.sqrt(mu_tot / (distance**3))

    r1 = np.sqrt((x - x1) ** 2 + y * y + z * z)
    r2 = np.sqrt((x - x2) ** 2 + y * y + z * z)
    if r1 <= 0.0 or r2 <= 0.0:
        raise ValueError("Position coincides with primary or secondary body in synodic frame.")

    omega = 0.5 * (n * n) * (x * x + y * y) + (mu_1 / r1) + (mu_2 / r2)
    return float(2.0 * omega - v_sq)


def cr3bp_synodic_derivatives(
    t: float,
    state: np.ndarray | list,
    mu_1: float = MU_EARTH,
    mu_2: float = MU_MOON,
    distance: float = R_MOON_ORBIT,
) -> np.ndarray:
    """Equations of motion in the dimensional Earth-Moon synodic barycentric frame."""
    s = np.asarray(state, dtype=np.float64)
    x, y, z, vx, vy, vz = s[0], s[1], s[2], s[3], s[4], s[5]

    mu_tot = mu_1 + mu_2
    pi_2 = mu_2 / mu_tot
    x1 = -pi_2 * distance
    x2 = (1.0 - pi_2) * distance
    n = np.sqrt(mu_tot / (distance**3))

    r1 = np.sqrt((x - x1) ** 2 + y * y + z * z)
    r2 = np.sqrt((x - x2) ** 2 + y * y + z * z)

    ax = 2.0 * n * vy + (n * n) * x - mu_1 * (x - x1) / (r1**3) - mu_2 * (x - x2) / (r2**3)
    ay = -2.0 * n * vx + (n * n) * y - mu_1 * y / (r1**3) - mu_2 * y / (r2**3)
    az = -mu_1 * z / (r1**3) - mu_2 * z / (r2**3)

    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)


def detect_l1_l2_gateway(
    r_eci: np.ndarray | list,
    v_eci: np.ndarray | list,
    t: float = 0.0,
    r_moon: np.ndarray | list | None = None,
) -> dict[str, object]:
    """Detect Earth-Moon L1/L2 gateway transit regimes (r ~ 320,000 km to 384,400+ km)
    and Weak Stability Boundary (WSB) ballistic capture eligibility.
    """
    r = np.asarray(r_eci, dtype=np.float64)
    v = np.asarray(v_eci, dtype=np.float64)
    r_m = moon_position(t) if r_moon is None else np.asarray(r_moon, dtype=np.float64)

    r_norm = float(np.linalg.norm(r))
    d_moon_vec = r - r_m
    d_moon = float(np.linalg.norm(d_moon_vec))

    # Moon velocity in circular ECI orbit
    omega_m = np.sqrt((MU_EARTH + MU_MOON) / (R_MOON_ORBIT**3))
    v_moon = np.array([-omega_m * r_m[1], omega_m * r_m[0], 0.0], dtype=np.float64)
    v_rel_moon = v - v_moon
    v_rel_sq = float(np.dot(v_rel_moon, v_rel_moon))

    # Specific selenocentric Keplerian energy
    eps_moon = 0.5 * v_rel_sq - (MU_MOON / max(d_moon, 1.0))

    # Transform to dimensional synodic barycentric frame to compute Jacobi constant
    u_x = r_m / np.linalg.norm(r_m)
    u_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u_y = np.cross(u_z, u_x)
    rot = np.vstack([u_x, u_y, u_z])

    pi_2 = MU_MOON / (MU_EARTH + MU_MOON)
    r_bary_eci = r - pi_2 * r_m
    r_syn = rot @ r_bary_eci
    omega_vec = np.array([0.0, 0.0, omega_m], dtype=np.float64)
    v_bary_eci = v - pi_2 * v_moon
    v_syn = rot @ (v_bary_eci - np.cross(omega_vec, r_bary_eci))

    c_jacobi = jacobi_constant_synodic(r_syn, v_syn)

    # L1 ~ 326,390 km from Earth center, L2 ~ 448,900 km; gateway transit band 320,000 - 384,400 km (plus L2 corridor)
    in_radial_band = 3.15e8 <= r_norm <= 4.55e8
    r_soi_moon = R_MOON_ORBIT * ((MU_MOON / MU_EARTH) ** (2.0 / 5.0))  # ~66,183 km
    near_moon_gateway = d_moon <= 1.25 * r_soi_moon

    gateway = "NONE"
    if in_radial_band and near_moon_gateway:
        gateway = "L1" if r_norm <= R_MOON_ORBIT else "L2"

    wsb_capture = bool(near_moon_gateway and eps_moon < 0.0)

    return {
        "in_gateway_regime": bool(in_radial_band and near_moon_gateway),
        "gateway": gateway,
        "wsb_ballistic_capture": wsb_capture,
        "r_earth_km": r_norm / 1000.0,
        "r_moon_km": d_moon / 1000.0,
        "selenocentric_energy": eps_moon,
        "jacobi_constant": c_jacobi,
    }


def _associated_legendre_4x4(sin_phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute unnormalized Associated Legendre polynomials P_nm(sin_phi) and dP_nm/dphi for n <= 4."""
    cos_phi = float(np.sqrt(max(0.0, 1.0 - sin_phi * sin_phi)))
    P = np.zeros((5, 5), dtype=np.float64)
    dP = np.zeros((5, 5), dtype=np.float64)

    P[0, 0] = 1.0
    P[1, 0] = sin_phi
    P[1, 1] = cos_phi
    dP[1, 0] = cos_phi
    dP[1, 1] = -sin_phi

    for n in range(2, 5):
        P[n, n] = (2 * n - 1) * cos_phi * P[n - 1, n - 1]
        for m in range(0, n):
            if n - 1 >= m:
                P[n, m] = ((2 * n - 1) * sin_phi * P[n - 1, m] - (n + m - 1) * P[n - 2, m]) / (n - m)

    # Derivative with respect to latitude phi: dP_nm/dphi = n*tan(phi)*P_nm - (n+m)/cos(phi)*P_{n-1,m}
    if cos_phi > 1e-12:
        tan_phi = sin_phi / cos_phi
        for n in range(2, 5):
            for m in range(0, n + 1):
                prev = P[n - 1, m] if m <= n - 1 else 0.0
                dP[n, m] = n * tan_phi * P[n, m] - ((n + m) / cos_phi) * prev

    return P, dP


def lunar_harmonics_acceleration(
    r_selenocentric: np.ndarray | list,
    mu_moon: float = MU_MOON,
    r_moon: float = R_MOON,
    harmonics: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> np.ndarray:
    """Compute lunar zonal and tesseral gravitational harmonic perturbations (LP165P 4x4)
    in the selenocentric frame.
    """
    r_vec = np.asarray(r_selenocentric, dtype=np.float64)
    if r_vec.shape != (3,):
        raise ValueError(f"Selenocentric position vector must have shape (3,), got {r_vec.shape}")
    r = float(np.linalg.norm(r_vec))
    if r <= 0.0:
        raise ValueError("Selenocentric radius must be positive.")

    x, y, z = float(r_vec[0]), float(r_vec[1]), float(r_vec[2])
    r_xy = float(np.hypot(x, y))
    sin_phi = np.clip(z / r, -1.0, 1.0)
    cos_phi = float(np.sqrt(max(0.0, 1.0 - sin_phi * sin_phi)))
    lam = float(np.arctan2(y, x))

    coeffs = LP165P_HARMONICS_4X4 if harmonics is None else harmonics
    P, dP = _associated_legendre_4x4(sin_phi)

    dU_dr = 0.0
    dU_dphi = 0.0
    dU_dlam = 0.0

    for (n, m), (c_nm, s_nm) in coeffs.items():
        if n < 2 or n > 4 or m < 0 or m > n:
            continue
        ratio_n = (r_moon / r) ** n
        cos_ml = np.cos(m * lam)
        sin_ml = np.sin(m * lam)
        cs_term = c_nm * cos_ml + s_nm * sin_ml
        d_cs_term = m * (-c_nm * sin_ml + s_nm * cos_ml)

        dU_dr += -(mu_moon / (r * r)) * (n + 1) * ratio_n * P[n, m] * cs_term
        dU_dphi += (mu_moon / r) * ratio_n * dP[n, m] * cs_term
        dU_dlam += (mu_moon / r) * ratio_n * P[n, m] * d_cs_term

    if r_xy > 1e-9:
        ax = (dU_dr / r - (z / (r * r * r_xy)) * dU_dphi) * x - (dU_dlam / (r_xy * r_xy)) * y
        ay = (dU_dr / r - (z / (r * r * r_xy)) * dU_dphi) * y + (dU_dlam / (r_xy * r_xy)) * x
        az = (dU_dr / r) * z + (r_xy / (r * r)) * dU_dphi
    else:
        ax = 0.0
        ay = 0.0
        az = dU_dr * np.sign(z if z != 0.0 else 1.0)

    return np.array([ax, ay, az], dtype=np.float64)


def total_gravity_acceleration(
    r_vec: np.ndarray | list,
    include_j2: bool = True,
    mu: float = MU_EARTH,
    r_earth: float = R_EARTH,
    j2: float = J2_EARTH,
) -> np.ndarray:
    """Compute total gravitational acceleration (central + optional J2).

    Args:
        r_vec: Position vector in ECI frame [x, y, z] in meters.
        include_j2: Whether to include the J2 oblateness perturbation.
        mu: Standard gravitational parameter in m^3 / s^2.
        r_earth: Earth equatorial radius in meters.
        j2: J2 harmonic coefficient.

    Returns:
        np.ndarray: Combined gravitational acceleration [ax, ay, az] in m / s^2.
    """
    a_central = central_gravity_acceleration(r_vec, mu=mu)
    if not include_j2:
        return a_central

    a_j2 = j2_perturbation_acceleration(r_vec, mu=mu, r_earth=r_earth, j2=j2)
    return a_central + a_j2


def j3_perturbation_acceleration(
    r_vec: np.ndarray | list,
    mu: float = MU_EARTH,
    r_earth: float = R_EARTH,
    j3: float = J3_EARTH,
) -> np.ndarray:
    """Compute perturbative acceleration due to Earth's J3 pear-shaped zonal harmonic.

    Standard formulation:
        factor = - (5/2) * J3 * mu * R_E^3 / r^7
        a_x = factor * x * z * (3 - 7 * z^2 / r^2)
        a_y = factor * y * z * (3 - 7 * z^2 / r^2)
        a_z = factor * (6 * z^2 - 7 * z^4 / r^2 - (3/5) * r^2)
    """
    r = np.asarray(r_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    x, y, z = r[0], r[1], r[2]
    r2 = r_norm * r_norm
    z2 = z * z
    z2_over_r2 = z2 / r2

    factor = -2.5 * j3 * mu * (r_earth**3) / (r_norm**7)
    ax = factor * x * z * (3.0 - 7.0 * z2_over_r2)
    ay = factor * y * z * (3.0 - 7.0 * z2_over_r2)
    az = factor * (6.0 * z2 - 7.0 * (z2 * z2_over_r2) - 0.6 * r2)
    return np.array([ax, ay, az], dtype=np.float64)


def j4_perturbation_acceleration(
    r_vec: np.ndarray | list,
    mu: float = MU_EARTH,
    r_earth: float = R_EARTH,
    j4: float = J4_EARTH,
) -> np.ndarray:
    """Compute perturbative acceleration due to Earth's J4 fourth zonal harmonic."""
    r = np.asarray(r_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    x, y, z = r[0], r[1], r[2]
    r2 = r_norm * r_norm
    z2_over_r2 = (z * z) / r2

    factor = (5.0 / 8.0) * j4 * mu * (r_earth**4) / (r_norm**7)
    term_xy = 3.0 - 42.0 * z2_over_r2 + 63.0 * (z2_over_r2**2)
    term_z = 15.0 - 70.0 * z2_over_r2 + 63.0 * (z2_over_r2**2)

    ax = factor * x * term_xy
    ay = factor * y * term_xy
    az = factor * z * term_z
    return np.array([ax, ay, az], dtype=np.float64)


def tesseral_j22_acceleration(
    r_vec: np.ndarray | list,
    t: float = 0.0,
    mu: float = MU_EARTH,
    r_earth: float = R_EARTH,
    c22: float = C22_EARTH,
    s22: float = S22_EARTH,
    omega_earth: float = OMEGA_EARTH,
) -> np.ndarray:
    """Compute perturbative acceleration from Earth's equatorial ellipticity (C22, S22).

    Drives longitudinal drift of geostationary orbits toward stable points at 75.1 deg E and 105.3 deg W.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    theta_gmst = (omega_earth * t) % (2.0 * np.pi)
    cos_th, sin_th = np.cos(theta_gmst), np.sin(theta_gmst)

    # Rotate ECI position to ECEF
    x_ecef = cos_th * r[0] + sin_th * r[1]
    y_ecef = -sin_th * r[0] + cos_th * r[1]
    z_ecef = r[2]

    # Partial derivatives of V_22 in ECEF
    # V22 = (3 * mu * R_E^2 / r^5) * (C22 * (x^2 - y^2) + 2 * S22 * x * y)
    factor = 3.0 * mu * (r_earth**2) / (r_norm**5)
    quad = c22 * (x_ecef**2 - y_ecef**2) + 2.0 * s22 * x_ecef * y_ecef

    dV_dx = factor * (2.0 * c22 * x_ecef + 2.0 * s22 * y_ecef) - (5.0 * x_ecef / (r_norm**2)) * factor * quad
    dV_dy = factor * (-2.0 * c22 * y_ecef + 2.0 * s22 * x_ecef) - (5.0 * y_ecef / (r_norm**2)) * factor * quad
    dV_dz = - (5.0 * z_ecef / (r_norm**2)) * factor * quad

    # Rotate back to ECI
    ax = cos_th * dV_dx - sin_th * dV_dy
    ay = sin_th * dV_dx + cos_th * dV_dy
    az = dV_dz
    return np.array([ax, ay, az], dtype=np.float64)


def solid_earth_tide_acceleration(
    r_vec: np.ndarray | list,
    r_moon: np.ndarray | None = None,
    r_sun: np.ndarray | None = None,
    mu_moon: float = MU_MOON,
    mu_sun: float = MU_SUN,
    r_earth: float = R_EARTH,
    k2: float = K2_EARTH_TIDE,
) -> np.ndarray:
    """Compute solid Earth tidal potential perturbation on satellite due to Moon & Sun tidal bulges."""
    r = np.asarray(r_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    a_tide = np.zeros(3, dtype=np.float64)
    bodies = []
    if r_moon is not None:
        bodies.append((np.asarray(r_moon, dtype=np.float64), mu_moon))
    if r_sun is not None:
        bodies.append((np.asarray(r_sun, dtype=np.float64), mu_sun))

    for r_b, mu_b in bodies:
        d_b = float(np.linalg.norm(r_b))
        if d_b <= 0.0:
            continue
        u_b = r_b / d_b
        cos_psi = float(np.dot(r / r_norm, u_b))
        factor = (k2 * mu_b * (r_earth**5)) / (2.0 * (d_b**3) * (r_norm**4))
        # Gradient of P2(cos psi) term
        grad_term = factor * (3.0 * (3.0 * cos_psi**2 - 1.0) * (r / r_norm) - 6.0 * cos_psi * u_b)
        a_tide += grad_term

    return a_tide


def earth_magnetic_field_vector(
    r_vec: np.ndarray | list,
    t: float = 0.0,
    r_earth: float = R_EARTH,
    b0: float = B0_EARTH_SURFACE,
    tilt_rad: float = MAG_POLE_TILT_RAD,
    lon_pole_rad: float = MAG_POLE_LON_RAD,
    omega_earth: float = OMEGA_EARTH,
) -> np.ndarray:
    """Compute geomagnetic dipole magnetic field vector B in Tesla at satellite position in ECI frame.

    B(r) = (B0 * R_E^3 / r^3) * [3 * (m_hat . r_hat) * r_hat - m_hat]
    """
    r = np.asarray(r_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    # Current geographic longitude of magnetic dipole in ECI
    pole_lon_eci = lon_pole_rad + (omega_earth * t)
    # Unit vector pointing toward North Magnetic Pole in ECI
    m_hat = np.array([
        np.sin(tilt_rad) * np.cos(pole_lon_eci),
        np.sin(tilt_rad) * np.sin(pole_lon_eci),
        np.cos(tilt_rad),
    ], dtype=np.float64)

    r_hat = r / r_norm
    dipole_scale = b0 * ((r_earth / r_norm) ** 3)
    b_vec = dipole_scale * (3.0 * float(np.dot(m_hat, r_hat)) * r_hat - m_hat)
    return b_vec


def lorentz_magnetic_acceleration(
    r_vec: np.ndarray | list,
    v_vec: np.ndarray | list,
    q_charge: float = 1e-4,  # Net residual satellite charge in Coulombs
    mass: float = 4.0,       # Satellite mass in kg
    t: float = 0.0,
) -> np.ndarray:
    """Compute Lorentz acceleration a = (q/m) * (v x B) from geomagnetic interaction."""
    if mass <= 0.0:
        return np.zeros(3, dtype=np.float64)
    b_vec = earth_magnetic_field_vector(r_vec, t=t)
    v = np.asarray(v_vec, dtype=np.float64)
    f_lorentz = q_charge * np.cross(v, b_vec)
    return f_lorentz / mass

