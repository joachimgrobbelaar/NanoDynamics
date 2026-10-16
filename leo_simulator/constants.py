"""Physical and geodetic constants for LEO orbital dynamics simulation.

Values are referenced to WGS-84 / EGM96 standards in SI units (m, kg, s, rad).
"""

from typing import Final

# Earth Standard Gravitational Parameter [m^3 / s^2] (WGS-84)
MU_EARTH: Final[float] = 3.986004418e14

# Earth Equatorial Radius [m] (WGS-84)
R_EARTH: Final[float] = 6378137.0

# Earth Polar Semi-major Axis [m] (WGS-84)
R_EARTH_POLAR: Final[float] = 6356752.3142

# Earth Flattening factor (f = (a - b) / a)
EARTH_FLATTENING: Final[float] = 1.0 / 298.257223563

# Earth Second Zonal Harmonic (J2 oblateness coefficient) (WGS-84 / EGM96)
J2_EARTH: Final[float] = 1.08262668e-3

# Earth Mean Rotation Rate (sidereal angular velocity) [rad / s]
OMEGA_EARTH: Final[float] = 7.2921150e-5

# Moon Standard Gravitational Parameter [m^3 / s^2] (GRAIL)
MU_MOON: Final[float] = 4.902800066e12

# Approximate Moon circular-orbit ephemeris (equatorial plane)
R_MOON_ORBIT: Final[float] = 384_400_000.0  # [m]
T_MOON_ORBIT: Final[float] = 27.32 * 24 * 3600  # [s]
R_MOON: Final[float] = 1_738_000.0  # [m] (LP165P reference radius)

# Sun Standard Gravitational Parameter & 1 AU Ephemeris [SI]
MU_SUN: Final[float] = 1.32712440018e20  # [m^3 / s^2]
R_SUN_ORBIT: Final[float] = 149_597_870_700.0  # [m] (1 AU)
T_SUN_ORBIT: Final[float] = 365.256363 * 86400.0  # [s]

# Lunar LP165P unnormalized gravitational harmonics truncated to degree/order 4x4
# Format: (n, m): (C_nm, S_nm)
LP165P_HARMONICS_4X4: Final[dict[tuple[int, int], tuple[float, float]]] = {
    (2, 0): (-2.0323e-4, 0.0),
    (2, 1): (0.0, 0.0),
    (2, 2): (2.2388e-5, 0.0),
    (3, 0): (-8.459e-6, 0.0),
    (3, 1): (2.848e-5, 5.89e-6),
    (3, 2): (4.84e-6, 1.67e-6),
    (3, 3): (1.71e-6, -2.5e-7),
    (4, 0): (9.64e-6, 0.0),
    (4, 1): (-5.68e-6, 1.58e-6),
    (4, 2): (-1.59e-6, -1.52e-6),
    (4, 3): (-8.2e-8, -7.9e-7),
    (4, 4): (-1.24e-7, 9.1e-8),
}

# Default reference atmospheric parameters for exponential model at LEO
# Reference altitude h0 = 400 km
ATMOSPHERE_DEFAULT_H0: Final[float] = 400_000.0  # [m]
# Reference density at 400 km for moderate solar activity [kg / m^3]
ATMOSPHERE_DEFAULT_RHO0: Final[float] = 2.80e-12  # [kg / m^3]
# Scale height near 400 km [m]
ATMOSPHERE_DEFAULT_SCALE_HEIGHT: Final[float] = 58_200.0  # [m]

# Sea level standard atmospheric parameters (US Standard Atmosphere 1976)
ATMOSPHERE_SEA_LEVEL_RHO: Final[float] = 1.225  # [kg / m^3]
ATMOSPHERE_SEA_LEVEL_SCALE_HEIGHT: Final[float] = 7200.0  # [m]

