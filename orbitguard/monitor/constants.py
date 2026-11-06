"""Pinned frames, units, and product settings for OrbitGuard.

Display thresholds below are operator settings for this preview. They are
not 18 SDS, NASA CARA, or collision-safety criteria.
"""

from __future__ import annotations

# SGP4 public GP element sets are propagated with the WGS-72 constant set
# used by the SGP4 theory. Geodetic display then uses the WGS-84 ellipsoid.
# Those are different models; a displayed altitude is derived, not a catalog
# observable.
SGP4_GRAVITY = "WGS72"
SGP4_OPERATION_MODE = "i"  # python-sgp4 improved mode, the twoline2rv default

# CelesTrak JSON omits these mandatory OMM values. The GP formats page states
# the omitted values explicitly.
# https://celestrak.org/NORAD/documentation/gp-data-formats.php
GP_CENTER = "EARTH"
GP_REFERENCE_FRAME = "TEME"
GP_TIME_SYSTEM = "UTC"
GP_MEAN_ELEMENT_THEORY = "SGP4"

# Earth-fixed display: TEME rotated by GMST. Polar motion and the equation of
# equinoxes are not applied, and UTC is used in place of UT1.
DISPLAY_FRAME = "teme_gmst_earth_fixed"
DISPLAY_FRAME_NOTE = (
    "Earth-fixed display frame: TEME position rotated by Greenwich mean "
    "sidereal time. UTC stands in for UT1 (no IERS bulletin). Polar motion "
    "and the equation of equinoxes are omitted, so this is not ITRF."
)

# Frame name of leo_simulator.OrbitPropagator. J2 is aligned with that
# model's z axis. It is not TEME and not GCRF.
NUMERICAL_FRAME = "nanodynamics_eci"
NUMERICAL_FRAME_NOTE = (
    "Cartesian state for leo_simulator.OrbitPropagator. The integrator's "
    "z axis is the J2 axis of that model. Do not place a SGP4/TEME vector "
    "in this frame without a transform this version does not provide."
)

# GMST polynomial constant term at JD 2451545.0 (2000-01-01 12:00).
GMST_J2000_DEG = 280.46061837

# Product display settings, not orbital safety rules.
# CelesTrak states that it checks for new GP data once every 2 hours and
# asks clients not to download a group more often than that.
# https://celestrak.org/NORAD/documentation/gp-data-formats.php
CELESTRAK_MIN_REFETCH_S = 2 * 3600
# How old an element epoch may be before the workspace marks it
# "older than the display setting". CelesTrak's OLDEST table flag uses
# 3.5 days as a data-review aid for active satellites; that flag is also
# not a safety threshold. 72 hours is this preview's own label.
DEFAULT_EPOCH_STALE_S = 72 * 3600
# How long after a successful download the retrieval is marked older than
# the display setting. Independent of element epoch age.
DEFAULT_RETRIEVAL_STALE_S = 6 * 3600
# NOAA index cache. The files update on NOAA's schedule; this only limits
# how often this process requests them.
DEFAULT_ENVIRONMENT_CACHE_S = 30 * 60

# Geometric approach list. Approaches farther than this are omitted from the
# short list. Omission is not a probability and not a clearance.
DEFAULT_GEOMETRIC_SCREEN_KM = 25.0
MAX_SCREEN_MINUTES = 360.0
MIN_SCREEN_STEP_S = 15.0

CELESTRAK_GP_DOC = "https://celestrak.org/NORAD/documentation/gp-data-formats.php"
CELESTRAK_STATIONS_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=JSON"
)
SPACETRACK_DOC = "https://www.space-track.org/documentation"
NOAA_JSON_INDEX = "https://services.swpc.noaa.gov/json/"
NOAA_F107_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
NOAA_KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"

# Allowed public batch in this preview. One group request, not a per-object
# catalog walk, and not the large active/Starlink groups.
ALLOWED_PUBLIC_GROUP = "stations"

USER_AGENT = "OrbitGuard/0.1 (public read-only GP and SWPC; cached)"

F107_UNIT = "sfu (10^-22 W m^-2 Hz^-1)"
KP_UNIT = "1 (dimensionless)"
TEC_UNIT = "TECU (10^16 electrons m^-2)"
DENSITY_UNIT = "kg m^-3"
