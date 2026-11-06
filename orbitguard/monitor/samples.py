"""Frozen synthetic mean elements for the labelled simulation sample.

These are not NORAD objects. The integers exist only so SGP4 can be
initialized. States are shown at the element epoch, not extrapolated to
the current time and presented as a live location.
"""

from __future__ import annotations

from typing import Any

SAMPLE_SOURCE = "orbitguard_simulation_sample"

_COMMON = {
    "OBJECT_ID": "2024-901A",
    "EPOCH": "2024-06-01T00:00:00.000000",
    "MEAN_MOTION": 15.5,
    "ECCENTRICITY": 0.001,
    "RA_OF_ASC_NODE": 120.0,
    "ARG_OF_PERICENTER": 45.0,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "ELEMENT_SET_NO": 1,
    "REV_AT_EPOCH": 1,
    "BSTAR": 0.0001,
    "MEAN_MOTION_DOT": 1.0e-5,
    "MEAN_MOTION_DDOT": 0.0,
}


def sample_records() -> list[dict[str, Any]]:
    primary = {
        **_COMMON,
        "OBJECT_NAME": "OG-SAMPLE-LEO-A",
        "INCLINATION": 51.6,
        "MEAN_ANOMALY": 10.0,
        "NORAD_CAT_ID": 90001,
        "public_id": "sample:leo-a",
    }
    secondary = {
        **_COMMON,
        "OBJECT_ID": "2024-901B",
        "OBJECT_NAME": "OG-SAMPLE-LEO-B",
        "INCLINATION": 51.62,
        "MEAN_ANOMALY": 10.05,
        "NORAD_CAT_ID": 90002,
        "public_id": "sample:leo-b",
    }
    return [primary, secondary]
