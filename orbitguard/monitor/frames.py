"""TEME display transform.

SGP4 position and velocity are TEME. The display frame rotates that
vector by Greenwich mean sidereal time. UTC is used as a stand-in for
UT1. Polar motion and the equation of equinoxes are not applied.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sgp4.api import jday

from leo_simulator.coordinates import OMEGA_EARTH, ecef_to_geodetic
from orbitguard.monitor.constants import DISPLAY_FRAME, DISPLAY_FRAME_NOTE, GMST_J2000_DEG
from orbitguard.monitor.provenance import as_utc


def julian_date(moment: datetime) -> float:
    moment = as_utc(moment)
    assert moment is not None
    jd, fraction = jday(
        moment.year,
        moment.month,
        moment.day,
        moment.hour,
        moment.minute,
        moment.second + moment.microsecond * 1e-6,
    )
    return float(jd) + float(fraction)


def gmst_deg(moment: datetime) -> float:
    """Greenwich mean sidereal time in degrees, IAU 1982 polynomial.

    At 2000-01-01 12:00 UTC the polynomial returns ``GMST_J2000_DEG``.
    """
    jd = julian_date(moment)
    centuries = (jd - 2451545.0) / 36525.0
    gmst = (
        GMST_J2000_DEG
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * centuries * centuries
        - (centuries ** 3) / 38710000.0
    )
    return float(gmst % 360.0)


def rotation_teme_to_display(gmst_degrees: float) -> np.ndarray:
    """Passive rotation taking TEME coordinates into the GMST Earth-fixed frame."""
    theta = np.deg2rad(gmst_degrees)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    return np.array(
        [
            [cos_t, sin_t, 0.0],
            [-sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def teme_to_display(
    r_teme_km: np.ndarray,
    v_teme_km_s: np.ndarray,
    moment: datetime,
) -> dict:
    """Rotate a TEME state and convert the position to WGS-84 geodetic.

    Velocity is rotated and the Earth-rate transport term is removed so the
    vector is an Earth-fixed display velocity. It is still labelled derived.
    """
    r_teme_km = np.asarray(r_teme_km, dtype=np.float64)
    v_teme_km_s = np.asarray(v_teme_km_s, dtype=np.float64)
    gmst = gmst_deg(moment)
    rotation = rotation_teme_to_display(gmst)
    r_display_km = rotation @ r_teme_km
    omega = np.array([0.0, 0.0, OMEGA_EARTH], dtype=np.float64)
    v_display_km_s = rotation @ v_teme_km_s - np.cross(omega, r_display_km)
    lat, lon, alt_km = ecef_to_geodetic(
        r_display_km * 1000.0,
        ellipsoidal=True,
    )
    return {
        "frame": DISPLAY_FRAME,
        "frame_note": DISPLAY_FRAME_NOTE,
        "gmst_deg": gmst,
        "r_km": r_display_km.tolist(),
        "v_km_s": v_display_km_s.tolist(),
        "latitude_deg": float(lat),
        "longitude_deg": float(lon),
        "altitude_km": float(alt_km),
        "geodetic_ellipsoid": "WGS84",
        "status": "derived",
    }
