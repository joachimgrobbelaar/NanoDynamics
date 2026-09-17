"""Atmospheric density and aerodynamic drag models for LEO orbits.

Implements exponential atmospheric models and aerodynamic drag force
opposing the spacecraft velocity relative to the rotating atmosphere.
"""

from abc import ABC, abstractmethod

import numpy as np

from leo_simulator.constants import (
    ATMOSPHERE_DEFAULT_H0,
    ATMOSPHERE_DEFAULT_RHO0,
    ATMOSPHERE_DEFAULT_SCALE_HEIGHT,
    OMEGA_EARTH,
    R_EARTH,
)


class BaseAtmosphere(ABC):
    """Abstract base class for atmospheric density models."""

    @abstractmethod
    def density(self, altitude_m: float) -> float:
        """Compute atmospheric density at a given geometric altitude above Earth surface.

        Args:
            altitude_m: Altitude above spherical Earth radius in meters.

        Returns:
            float: Atmospheric density in kg / m^3 (non-negative).
        """

    def __call__(self, altitude_m: float) -> float:
        return self.density(altitude_m)


class ExponentialAtmosphere(BaseAtmosphere):
    """Single-scale-height exponential atmospheric model.

    rho(h) = rho_0 * exp(-(h - h_0) / H)

    For altitude below 0 m, density is clamped at the surface value (or h=0) to prevent
    unbounded exponential growth if numerical steps dip slightly below reference.
    """

    def __init__(
        self,
        h0: float = ATMOSPHERE_DEFAULT_H0,
        rho0: float = ATMOSPHERE_DEFAULT_RHO0,
        scale_height: float = ATMOSPHERE_DEFAULT_SCALE_HEIGHT,
        min_altitude: float = 0.0,
    ) -> None:
        """Initialize exponential atmosphere.

        Args:
            h0: Reference altitude in meters.
            rho0: Reference density at h0 in kg / m^3.
            scale_height: Scale height H in meters (> 0).
            min_altitude: Minimum altitude clamp in meters.
        """
        if scale_height <= 0.0:
            raise ValueError(f"Scale height must be positive, got {scale_height}")
        if rho0 < 0.0:
            raise ValueError(f"Reference density cannot be negative, got {rho0}")

        self.h0 = float(h0)
        self.rho0 = float(rho0)
        self.scale_height = float(scale_height)
        self.min_altitude = float(min_altitude)

    def density(self, altitude_m: float) -> float:
        """Calculate density at altitude_m [m]."""
        if not np.isfinite(altitude_m):
            raise ValueError(f"Altitude must be finite, got {altitude_m}")
        clamped_h = max(altitude_m, self.min_altitude)
        exponent = -(clamped_h - self.h0) / self.scale_height
        # Prevent numerical underflow/overflow
        if exponent > 700.0:
            return float(self.rho0 * np.exp(700.0))
        if exponent < -700.0:
            return 0.0
        return float(self.rho0 * np.exp(exponent))


class PiecewiseExponentialAtmosphere(BaseAtmosphere):
    """Standard piecewise exponential atmospheric model for Earth's upper atmosphere.

    Uses discrete atmospheric layers with localized scale heights derived from standard
    reference atmospheres (US Standard Atmosphere 1976 / Jacchia model approximations)
    covering altitudes from 0 to 1,000 km.
    """

    # Format: (base_altitude_m, base_density_kg_per_m3, scale_height_m)
    TABLE: tuple[tuple[float, float, float], ...] = (
        (0.0, 1.225, 7249.0),
        (25_000.0, 3.899e-2, 6349.0),
        (50_000.0, 1.027e-3, 7922.0),
        (75_000.0, 3.992e-5, 6382.0),
        (100_000.0, 5.297e-7, 5877.0),
        (110_000.0, 9.661e-8, 7263.0),
        (120_000.0, 2.438e-8, 9473.0),
        (130_000.0, 8.484e-9, 12636.0),
        (140_000.0, 3.845e-9, 16149.0),
        (150_000.0, 2.070e-9, 22523.0),
        (180_000.0, 5.464e-10, 29740.0),
        (200_000.0, 2.789e-10, 37105.0),
        (250_000.0, 7.248e-11, 45546.0),
        (300_000.0, 2.418e-11, 53628.0),
        (350_000.0, 9.518e-12, 53298.0),
        (400_000.0, 3.725e-12, 58515.0),
        (450_000.0, 1.585e-12, 60828.0),
        (500_000.0, 6.967e-13, 63822.0),
        (600_000.0, 1.454e-13, 71835.0),
        (700_000.0, 3.614e-14, 88667.0),
        (800_000.0, 1.170e-14, 124640.0),
        (900_000.0, 5.245e-15, 181050.0),
        (1_000_000.0, 3.019e-15, 268000.0),
    )

    def __init__(self) -> None:
        self._bases = np.array([row[0] for row in self.TABLE], dtype=np.float64)
        self._rhos = np.array([row[1] for row in self.TABLE], dtype=np.float64)
        self._scales = np.array([row[2] for row in self.TABLE], dtype=np.float64)

    def density(self, altitude_m: float) -> float:
        """Compute density via piecewise exponential interpolation."""
        if not np.isfinite(altitude_m):
            raise ValueError(f"Altitude must be finite, got {altitude_m}")
        if altitude_m <= 0.0:
            return float(self._rhos[0])

        idx = np.searchsorted(self._bases, altitude_m, side="right") - 1
        if idx < 0:
            idx = 0
        elif idx >= len(self.TABLE) - 1:
            # Above top table altitude (1000 km), extrapolate using top scale height
            idx = len(self.TABLE) - 1

        h_base = self._bases[idx]
        rho_base = self._rhos[idx]
        scale_h = self._scales[idx]

        exponent = -(altitude_m - h_base) / scale_h
        if exponent < -700.0:
            return 0.0
        return float(rho_base * np.exp(exponent))


def relative_velocity_vector(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    omega_earth: float = OMEGA_EARTH,
    include_earth_rotation: bool = True,
) -> np.ndarray:
    """Calculate the spacecraft velocity relative to the co-rotating atmosphere.

    v_rel = v - (omega_E x r)

    In ECI frame with rotation axis along +z:
        omega_E = [0, 0, omega_earth]
        v_atm = [-omega_earth * y, omega_earth * x, 0]
        v_rel = v - v_atm

    Args:
        r_vec: ECI position vector [x, y, z] in meters.
        v_vec: ECI velocity vector [vx, vy, vz] in m / s.
        omega_earth: Earth rotation rate in rad / s.
        include_earth_rotation: If True, accounts for atmospheric co-rotation.

    Returns:
        np.ndarray: Relative velocity vector [vx_rel, vy_rel, vz_rel] in m / s.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError(f"Vectors must have shape (3,), got r:{r.shape} and v:{v.shape}")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
        raise ValueError("Position and velocity vectors must contain finite values.")

    if not include_earth_rotation or omega_earth == 0.0:
        return v.copy()

    v_atm = np.array([-omega_earth * r[1], omega_earth * r[0], 0.0], dtype=np.float64)
    return v - v_atm


def aerodynamic_drag_acceleration(
    r_vec: np.ndarray | list,
    v_vec: np.ndarray | list,
    cd: float,
    area: float,
    mass: float,
    atmosphere_model: BaseAtmosphere | None = None,
    r_earth: float = R_EARTH,
    omega_earth: float = OMEGA_EARTH,
    include_earth_rotation: bool = True,
) -> np.ndarray:
    """Compute aerodynamic drag acceleration opposing relative motion in the atmosphere.

    a_drag = -0.5 * rho * (cd * area / mass) * |v_rel| * v_rel

    Args:
        r_vec: ECI position vector [x, y, z] in meters.
        v_vec: ECI velocity vector [vx, vy, vz] in m / s.
        cd: Drag coefficient (dimensionless, typically 2.0 to 2.5 for LEO).
        area: Cross-sectional drag area in m^2.
        mass: Spacecraft mass in kg.
        atmosphere_model: Atmosphere instance (defaults to ExponentialAtmosphere).
        r_earth: Earth radius in meters for spherical altitude calculation.
        omega_earth: Earth rotation rate in rad / s.
        include_earth_rotation: Whether to include Earth's rotation in relative velocity.

    Returns:
        np.ndarray: Drag acceleration vector [ax, ay, az] in m / s^2.

    Raises:
        ValueError: If mass <= 0, area < 0, or cd < 0.
    """
    if mass <= 0.0:
        raise ValueError(f"Mass must be strictly positive, got {mass} kg")
    if area < 0.0:
        raise ValueError(f"Cross-sectional area must be non-negative, got {area} m^2")
    if cd < 0.0:
        raise ValueError(f"Drag coefficient must be non-negative, got {cd}")

    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)

    if r.shape != (3,) or v.shape != (3,):
        raise ValueError(f"Vectors must have shape (3,), got r:{r.shape} and v:{v.shape}")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
        raise ValueError("Position and velocity vectors must contain finite values.")

    r_norm = np.linalg.norm(r)
    if r_norm <= 0.0:
        raise ValueError("Position vector norm must be positive.")

    if area == 0.0 or cd == 0.0:
        return np.zeros(3, dtype=np.float64)

    if atmosphere_model is None:
        atmosphere_model = ExponentialAtmosphere()

    altitude = r_norm - r_earth

    rho = atmosphere_model.density(altitude)
    if rho <= 0.0:
        return np.zeros(3, dtype=np.float64)

    v_rel = relative_velocity_vector(
        r, v, omega_earth=omega_earth, include_earth_rotation=include_earth_rotation
    )
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm == 0.0:
        return np.zeros(3, dtype=np.float64)

    # Ballistic factor: (cd * area) / mass
    ballistic_factor = (cd * area) / mass

    # a_drag = - 0.5 * rho * ballistic_factor * |v_rel| * v_rel
    drag_accel = -0.5 * rho * ballistic_factor * v_rel_norm * v_rel
    return drag_accel
