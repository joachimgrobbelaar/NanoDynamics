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
MU_MOON: Final[float] = 4.902800066e9

# Approximate Moon circular-orbit ephemeris (equatorial plane)
R_MOON_ORBIT: Final[float] = 384_400_000.0  # [m]
T_MOON_ORBIT: Final[float] = 27.32 * 24 * 3600  # [s]

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
