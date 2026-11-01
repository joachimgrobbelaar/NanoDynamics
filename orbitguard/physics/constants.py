"""Astrodynamic and planetary constants for OrbitGuard physics engine."""

import math

# Earth Gravitational Parameter [m^3 / s^2]
MU_EARTH: float = 3.986004418e14

# Earth Equatorial Radius (WGS-84) [m]
R_EARTH: float = 6378137.0

# Earth Oblateness Perturbation Coefficient J2 (unnormalized)
J2_EARTH: float = 1.08262668e-3

# Earth Rotation Rate [rad / s]
OMEGA_EARTH: float = 7.2921159e-5

# Standard Sea-Level Gravitational Acceleration [m / s^2]
G0: float = 9.80665

# Atmosphere parameters (Standard Exponential LEO Model)
RHO_0: float = 1.225            # Sea-level atmospheric density [kg / m^3]
SCALE_HEIGHT_LEO: float = 8500.0 # Scale height for lower-upper thermosphere approx [m]
