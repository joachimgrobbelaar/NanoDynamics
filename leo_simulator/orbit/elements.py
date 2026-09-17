"""Keplerian orbital elements conversions and analytical perturbation rates.

Supports round-trip conversions between Cartesian state vectors [r, v] and
classical orbital elements [a, e, i, Omega, omega, nu], with robust handling
of circular and equatorial orbit singularities.
"""

from dataclasses import dataclass

import numpy as np

from leo_simulator.constants import J2_EARTH, MU_EARTH, R_EARTH


@dataclass(frozen=True)
class OrbitalElements:
    """Classical Keplerian Orbital Elements.

    Attributes:
        a: Semi-major axis [m]
        e: Eccentricity [dimensionless]
        i: Inclination [rad]
        raan: Right Ascension of Ascending Node (Omega) [rad]
        arg_pe: Argument of Periapsis (omega) [rad]
        nu: True Anomaly [rad]
    """

    a: float
    e: float
    i: float
    raan: float
    arg_pe: float
    nu: float

    @property
    def periapsis_radius(self) -> float:
        """Periapsis distance rp = a * (1 - e) in meters."""
        return self.a * (1.0 - self.e)

    @property
    def apoapsis_radius(self) -> float:
        """Apoapsis distance ra = a * (1 + e) in meters."""
        return self.a * (1.0 + self.e)

    @property
    def semi_latus_rectum(self) -> float:
        """p = a * (1 - e^2) in meters."""
        return self.a * (1.0 - self.e**2)

    @property
    def period(self) -> float:
        """Orbital period T = 2 * pi * sqrt(a^3 / mu) in seconds."""
        if self.a <= 0.0:
            return float("inf")
        return float(2.0 * np.pi * np.sqrt(self.a**3 / MU_EARTH))

    @property
    def mean_motion(self) -> float:
        """Mean motion n = sqrt(mu / a^3) in rad / s."""
        if self.a <= 0.0:
            return 0.0
        return float(np.sqrt(MU_EARTH / (self.a**3)))

    def to_degrees(self) -> dict[str, float]:
        """Return angles in degrees for reporting."""
        return {
            "a_km": self.a / 1000.0,
            "e": self.e,
            "i_deg": float(np.degrees(self.i)),
            "raan_deg": float(np.degrees(self.raan)),
            "arg_pe_deg": float(np.degrees(self.arg_pe)),
            "nu_deg": float(np.degrees(self.nu)),
        }


def rv_to_coe(
    r_vec: np.ndarray | list,
    v_vec: np.ndarray | list,
    mu: float = MU_EARTH,
    tol_e: float = 1e-10,
    tol_i: float = 1e-10,
) -> OrbitalElements:
    """Convert Cartesian state vector [r, v] to classical Keplerian elements.

    Handles circular (e < tol_e) and equatorial (i < tol_i) edge cases gracefully.

    Args:
        r_vec: Position vector in ECI meters [x, y, z].
        v_vec: Velocity vector in ECI m/s [vx, vy, vz].
        mu: Gravitational parameter in m^3 / s^2.
        tol_e: Threshold below which orbit is treated as circular.
        tol_i: Threshold below which orbit is treated as equatorial.

    Returns:
        OrbitalElements dataclass instance.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)

    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    if r_norm <= 0.0:
        raise ValueError("Position norm must be strictly positive.")

    # Specific angular momentum vector h = r x v
    h_vec = np.cross(r, v)
    h_norm = np.linalg.norm(h_vec)

    if h_norm <= 1e-12:
        raise ValueError("Radial or rectilinear orbit; angular momentum is zero.")

    # Specific energy epsilon = v^2 / 2 - mu / r
    energy = (v_norm**2) / 2.0 - mu / r_norm

    # Semi-major axis a = - mu / (2 * energy)
    if abs(energy) < 1e-14:
        # Parabolic orbit
        a = float("inf")
    else:
        a = -mu / (2.0 * energy)

    # Eccentricity vector e_vec = ((v^2 - mu/r)*r - (r.v)*v) / mu
    rdotv = np.dot(r, v)
    e_vec = ((v_norm**2 - mu / r_norm) * r - rdotv * v) / mu
    e = float(np.linalg.norm(e_vec))

    # Inclination i = arccos(hz / h)
    hz_ratio = np.clip(h_vec[2] / h_norm, -1.0, 1.0)
    i = float(np.arccos(hz_ratio))

    # Node vector N = k x h = [-hy, hx, 0]
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0], dtype=np.float64)
    n_norm = np.linalg.norm(n_vec)

    is_equatorial = (n_norm < tol_i) or (i < tol_i) or (abs(i - np.pi) < tol_i)
    is_circular = e < tol_e

    # Right Ascension of Ascending Node (RAAN, Omega)
    if is_equatorial:
        raan = 0.0
    else:
        nx_ratio = np.clip(n_vec[0] / n_norm, -1.0, 1.0)
        raan = float(np.arccos(nx_ratio))
        if n_vec[1] < 0.0:
            raan = 2.0 * np.pi - raan

    # Argument of Periapsis (omega, arg_pe)
    if is_circular and is_equatorial:
        # True longitude
        arg_pe = 0.0
    elif is_circular:
        # Argument of latitude u = arg_pe + nu
        arg_pe = 0.0
    elif is_equatorial:
        # Longitude of periapsis pi = Omega + omega = omega (since Omega = 0)
        ex_ratio = np.clip(e_vec[0] / e, -1.0, 1.0)
        arg_pe = float(np.arccos(ex_ratio))
        if (h_vec[2] >= 0.0 and e_vec[1] < 0.0) or (h_vec[2] < 0.0 and e_vec[1] > 0.0):
            arg_pe = 2.0 * np.pi - arg_pe
    else:
        ndote = np.clip(np.dot(n_vec, e_vec) / (n_norm * e), -1.0, 1.0)
        arg_pe = float(np.arccos(ndote))
        if e_vec[2] < 0.0:
            arg_pe = 2.0 * np.pi - arg_pe

    # True Anomaly (nu)
    if is_circular and is_equatorial:
        # Angle of position vector from x-axis
        rx_ratio = np.clip(r[0] / r_norm, -1.0, 1.0)
        nu = float(np.arccos(rx_ratio))
        if (h_vec[2] >= 0.0 and r[1] < 0.0) or (h_vec[2] < 0.0 and r[1] > 0.0):
            nu = 2.0 * np.pi - nu
    elif is_circular:
        # Angle from ascending node vector
        ndotr = np.clip(np.dot(n_vec, r) / (n_norm * r_norm), -1.0, 1.0)
        nu = float(np.arccos(ndotr))
        if r[2] < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        edotr = np.clip(np.dot(e_vec, r) / (e * r_norm), -1.0, 1.0)
        nu = float(np.arccos(edotr))
        if rdotv < 0.0:
            nu = 2.0 * np.pi - nu

    return OrbitalElements(
        a=float(a),
        e=float(e),
        i=float(i),
        raan=float(raan % (2.0 * np.pi)),
        arg_pe=float(arg_pe % (2.0 * np.pi)),
        nu=float(nu % (2.0 * np.pi)),
    )


def coe_to_rv(
    elements: OrbitalElements,
    mu: float = MU_EARTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert classical Keplerian orbital elements to Cartesian state vector [r, v].

    Args:
        elements: OrbitalElements object.
        mu: Gravitational parameter in m^3 / s^2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Position vector [m] and velocity vector [m/s] in ECI.
    """
    a = elements.a
    e = elements.e
    i = elements.i
    raan = elements.raan
    arg_pe = elements.arg_pe
    nu = elements.nu

    if a <= 0.0:
        raise ValueError("Semi-major axis must be positive for bound elliptic orbits.")
    if e < 0.0 or e >= 1.0:
        raise ValueError(f"Eccentricity must be in range [0, 1), got {e}")

    # Semi-latus rectum p
    p = a * (1.0 - e**2)

    # Position in perifocal coordinate system (PQW frame)
    r_pqw_norm = p / (1.0 + e * np.cos(nu))
    r_pqw = np.array(
        [r_pqw_norm * np.cos(nu), r_pqw_norm * np.sin(nu), 0.0],
        dtype=np.float64,
    )

    # Velocity in perifocal coordinate system
    v_factor = np.sqrt(mu / p)
    v_pqw = np.array(
        [-v_factor * np.sin(nu), v_factor * (e + np.cos(nu)), 0.0],
        dtype=np.float64,
    )

    # Rotation matrix from perifocal to ECI frame:
    # R = Rz(-raan) * Rx(-i) * Rz(-arg_pe)
    c_O, s_O = np.cos(raan), np.sin(raan)
    c_i, s_i = np.cos(i), np.sin(i)
    c_w, s_w = np.cos(arg_pe), np.sin(arg_pe)

    # Perifocal unit vectors P, Q expressed in ECI coordinates
    P = np.array(
        [
            c_O * c_w - s_O * s_w * c_i,
            s_O * c_w + c_O * s_w * c_i,
            s_w * s_i,
        ],
        dtype=np.float64,
    )

    Q = np.array(
        [
            -c_O * s_w - s_O * c_w * c_i,
            -s_O * s_w + c_O * c_w * c_i,
            c_w * s_i,
        ],
        dtype=np.float64,
    )

    r_eci = r_pqw[0] * P + r_pqw[1] * Q
    v_eci = v_pqw[0] * P + v_pqw[1] * Q

    return r_eci, v_eci


def analytical_j2_raan_rate(
    a: float,
    e: float,
    i: float,
    mu: float = MU_EARTH,
    r_earth: float = R_EARTH,
    j2: float = J2_EARTH,
) -> float:
    """Compute first-order secular nodal regression rate (dOmega/dt) due to J2 oblateness.

    Formula:
        dot_Omega = - (3/2) * J2 * (R_E / p)^2 * n * cos(i)
        where p = a * (1 - e^2) and n = sqrt(mu / a^3).

    Args:
        a: Semi-major axis in meters.
        e: Eccentricity.
        i: Inclination in radians.
        mu: Gravitational parameter.
        r_earth: Earth equatorial radius.
        j2: Earth J2 coefficient.

    Returns:
        float: Nodal regression rate in rad / s. (Negative for prograde orbits i < 90 deg).
    """
    if a <= 0.0 or e >= 1.0:
        return 0.0

    p = a * (1.0 - e**2)
    n = np.sqrt(mu / (a**3))
    rate = -1.5 * j2 * ((r_earth / p) ** 2) * n * np.cos(i)
    return float(rate)


def circular_velocity(altitude_m: float, mu: float = MU_EARTH, r_earth: float = R_EARTH) -> float:
    """Calculate circular orbital velocity at a given altitude in m/s."""
    r = r_earth + altitude_m
    return float(np.sqrt(mu / r))
