"""Gravitational acceleration models: central two-body gravity and J2 oblateness perturbation.

Coordinates are defined in an Earth-Centered Inertial (ECI) Cartesian frame [x, y, z].
"""


import numpy as np

from leo_simulator.constants import J2_EARTH, MU_EARTH, R_EARTH


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
