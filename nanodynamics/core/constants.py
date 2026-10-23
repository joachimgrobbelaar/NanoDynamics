"""
nanodynamics/core/constants.py

Gravitational, planetary, and reference frame constants for astrodynamics.
All values in SI/metric units unless noted.
"""

from __future__ import annotations

import math

# ── Fundamental constants ─────────────────────────────────────────────────────
G = 6.67430e-11                        # Gravitational constant [m³ kg⁻¹ s⁻²]
C_LIGHT = 299_792_458.0                # Speed of light [m/s]
AU_KM = 149_597_870.7                  # 1 Astronomical Unit [km]
AU_M = AU_KM * 1e3                     # 1 Astronomical Unit [m]

# ── Solar constants ───────────────────────────────────────────────────────────
GM_SUN = 1.32712440018e20              # Solar gravitational parameter [m³/s²]
R_SUN_KM = 696_340.0                   # Mean solar radius [km]
L_SUN_W = 3.828e26                     # Solar luminosity [W]

# ── Earth constants ───────────────────────────────────────────────────────────
GM_EARTH = 3.986004418e14              # Earth gravitational parameter [m³/s²]
R_EARTH_KM = 6_371.0                   # Mean Earth radius [km]
R_EARTH_M = R_EARTH_KM * 1e3          # Mean Earth radius [m]
J2_EARTH = 1.08263e-3                  # Earth J2 oblateness coefficient
EARTH_ROT_RAD_S = 7.2921159e-5        # Earth sidereal rotation rate [rad/s]
EARTH_SIDEREAL_DAY_S = 86164.0905     # Sidereal day [s]

# ── Moon constants ────────────────────────────────────────────────────────────
GM_MOON = 4.9048695e12                 # Lunar gravitational parameter [m³/s²]
R_MOON_KM = 1_737.4                   # Mean lunar radius [km]
A_MOON_KM = 384_400.0                 # Mean Earth-Moon distance [km]

# ── Mars constants ────────────────────────────────────────────────────────────
GM_MARS = 4.282837e13                  # Mars gravitational parameter [m³/s²]
R_MARS_KM = 3_389.5                   # Mean Mars radius [km]

# ── Venus constants ───────────────────────────────────────────────────────────
GM_VENUS = 3.24859e14                  # Venus gravitational parameter [m³/s²]
R_VENUS_KM = 6_051.8                   # Mean Venus radius [km]

# ── Mercury constants ────────────────────────────────────────────────────────
GM_MERCURY = 2.2032e13                 # Mercury gravitational parameter [m³/s²]
R_MERCURY_KM = 2_439.7                # Mean Mercury radius [km]

# ── Jupiter constants ────────────────────────────────────────────────────────
GM_JUPITER = 1.26686534e17            # Jupiter gravitational parameter [m³/s²]
R_JUPITER_KM = 69_911.0              # Mean Jupiter radius [km]

# ── Saturn constants ─────────────────────────────────────────────────────────
GM_SATURN = 3.7931187e16              # Saturn gravitational parameter [m³/s²]
R_SATURN_KM = 58_232.0               # Mean Saturn radius [km]

# ── Planetary GM lookup ──────────────────────────────────────────────────────
BODY_GM = {
    "sun": GM_SUN,
    "mercury": GM_MERCURY,
    "venus": GM_VENUS,
    "earth": GM_EARTH,
    "moon": GM_MOON,
    "mars": GM_MARS,
    "jupiter": GM_JUPITER,
    "saturn": GM_SATURN,
}

BODY_RADIUS_KM = {
    "sun": R_SUN_KM,
    "mercury": R_MERCURY_KM,
    "venus": R_VENUS_KM,
    "earth": R_EARTH_KM,
    "moon": R_MOON_KM,
    "mars": R_MARS_KM,
    "jupiter": R_JUPITER_KM,
    "saturn": R_SATURN_KM,
}

# ── Conversion helpers ───────────────────────────────────────────────────────
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
DAY_S = 86400.0                        # Solar day [s]
YEAR_S = 365.25 * DAY_S               # Julian year [s]
JD_J2000 = 2_451_545.0                # Julian Date of J2000.0 epoch
MJD_OFFSET = 2_400_000.5              # JD - MJD offset

# ── Atmospheric reference ────────────────────────────────────────────────────
SEA_LEVEL_DENSITY_KG_M3 = 1.225       # Standard atmosphere density [kg/m³]
SCALE_HEIGHT_KM = 8.5                  # Atmospheric scale height [km]
CD_TYPICAL = 2.2                       # Typical satellite drag coefficient
