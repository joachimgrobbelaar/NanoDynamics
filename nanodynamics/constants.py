"""Pinned frames and element-set conventions for the public physics API.

These are the values the SGP4 and display transforms use. They are not
product freshness thresholds and they are not collision-safety criteria.
"""

SGP4_GRAVITY = "WGS72"

# CelesTrak GP JSON omits these mandatory OMM fields. The formats page
# states the omitted values.
# https://celestrak.org/NORAD/documentation/gp-data-formats.php
GP_CENTER = "EARTH"
GP_REFERENCE_FRAME = "TEME"
GP_TIME_SYSTEM = "UTC"
GP_MEAN_ELEMENT_THEORY = "SGP4"
CELESTRAK_GP_DOC = "https://celestrak.org/NORAD/documentation/gp-data-formats.php"

# Earth-fixed display: TEME rotated by GMST. Polar motion and the equation
# of equinoxes are not applied, and UTC stands in for UT1.
DISPLAY_FRAME = "teme_gmst_earth_fixed"
DISPLAY_FRAME_NOTE = (
    "Earth-fixed display frame: TEME position rotated by Greenwich mean "
    "sidereal time. UTC stands in for UT1 (no IERS bulletin). Polar motion "
    "and the equation of equinoxes are omitted, so this is not ITRF."
)

# Frame of leo_simulator.OrbitPropagator. Its J2 axis is the model z axis.
# It is not TEME and not GCRF.
NUMERICAL_FRAME = "nanodynamics_eci"
NUMERICAL_FRAME_NOTE = (
    "Cartesian state for leo_simulator.OrbitPropagator. The integrator's "
    "z axis is the J2 axis of that model. A SGP4/TEME vector is not "
    "transformed into this frame."
)

# IAU 1982 GMST polynomial value at JD 2451545.0 (2000-01-01 12:00 UTC).
GMST_J2000_DEG = 280.46061837
