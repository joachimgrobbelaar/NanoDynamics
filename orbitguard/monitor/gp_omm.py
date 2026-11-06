"""Normalize one CelesTrak GP/OMM JSON object.

The JSON omits CENTER_NAME, REF_FRAME, TIME_SYSTEM, and
MEAN_ELEMENT_THEORY. Those are recorded from the CelesTrak GP formats
page, not invented per object. Public GP records do not include a
calibrated covariance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from typing import Any, Mapping

from orbitguard.monitor.constants import (
    CELESTRAK_GP_DOC,
    GP_CENTER,
    GP_MEAN_ELEMENT_THEORY,
    GP_REFERENCE_FRAME,
    GP_TIME_SYSTEM,
)

REQUIRED_FIELDS = (
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
)

_SECRET_KEYS = {
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "spacetrack_user",
    "spacetrack_password",
    "identity",
    "credential",
}

_OBJECT_ID = re.compile(r"^\d{4}-\d{3}[A-Z]{0,3}$")
_EPOCH_FRACTION = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<frac>\d+))?$"
)


class RecordRejected(ValueError):
    """A raw catalog record failed validation and was not stored."""


def normalize_epoch_text(epoch: str) -> str:
    """Return an ISO epoch with exactly six fractional digits.

    ``sgp4.omm.initialize`` parses ``%Y-%m-%dT%H:%M:%S.%f``.
    """
    text = str(epoch).strip().replace("Z", "")
    match = _EPOCH_FRACTION.match(text)
    if not match:
        raise RecordRejected(f"EPOCH is not UTC ISO-8601: {epoch!r}")
    frac = (match.group("frac") or "")[:6].ljust(6, "0")
    return f"{match.group('head')}.{frac}"


def parse_epoch(epoch: str) -> datetime:
    text = normalize_epoch_text(epoch)
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f").replace(
        tzinfo=__import__("datetime").timezone.utc
    )


def canonical_hash(record: Mapping[str, Any]) -> str:
    payload = {key: record[key] for key in REQUIRED_FIELDS}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _reject_secrets(record: Mapping[str, Any]) -> None:
    found = sorted(key for key in record if key.lower() in _SECRET_KEYS)
    if found:
        raise RecordRejected("Record contains credential-like fields and was rejected: " + ", ".join(found))


def _finite(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RecordRejected(f"{name} is not a number") from exc
    if not math.isfinite(number):
        raise RecordRejected(f"{name} is not finite")
    return number


def validate_omm(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one OMM object and return a copy plus content hash.

    The returned ``record`` keeps the source values. Covariance is recorded
    as absent. No default sigma is added.
    """
    if not isinstance(record, Mapping):
        raise RecordRejected("GP record must be a JSON object")
    _reject_secrets(record)
    missing = [key for key in REQUIRED_FIELDS if key not in record or record[key] is None]
    if missing:
        raise RecordRejected("Missing GP fields: " + ", ".join(missing))

    name = str(record["OBJECT_NAME"]).strip()
    if not name:
        raise RecordRejected("OBJECT_NAME is empty")
    object_id = str(record["OBJECT_ID"]).strip()
    if not _OBJECT_ID.match(object_id):
        raise RecordRejected(f"OBJECT_ID is not a YYYY-NNN designator: {object_id!r}")

    epoch_text = normalize_epoch_text(str(record["EPOCH"]))
    parse_epoch(epoch_text)

    mean_motion = _finite("MEAN_MOTION", record["MEAN_MOTION"])
    eccentricity = _finite("ECCENTRICITY", record["ECCENTRICITY"])
    inclination = _finite("INCLINATION", record["INCLINATION"])
    raan = _finite("RA_OF_ASC_NODE", record["RA_OF_ASC_NODE"])
    argp = _finite("ARG_OF_PERICENTER", record["ARG_OF_PERICENTER"])
    anomaly = _finite("MEAN_ANOMALY", record["MEAN_ANOMALY"])
    bstar = _finite("BSTAR", record["BSTAR"])
    ndot = _finite("MEAN_MOTION_DOT", record["MEAN_MOTION_DOT"])
    nddot = _finite("MEAN_MOTION_DDOT", record["MEAN_MOTION_DDOT"])
    if mean_motion <= 0.0:
        raise RecordRejected("MEAN_MOTION must be positive")
    if not 0.0 <= eccentricity < 1.0:
        raise RecordRejected("ECCENTRICITY must be in [0, 1)")
    if not 0.0 <= inclination <= 180.0:
        raise RecordRejected("INCLINATION must be in [0, 180] degrees")
    for label, angle in (
        ("RA_OF_ASC_NODE", raan),
        ("ARG_OF_PERICENTER", argp),
        ("MEAN_ANOMALY", anomaly),
    ):
        if not 0.0 <= angle < 360.0 and not math.isclose(angle, 360.0):
            # CelesTrak values are in [0, 360). Allow a recorded 0.
            if angle < 0.0 or angle > 360.0:
                raise RecordRejected(f"{label} is outside [0, 360] degrees")

    try:
        norad = int(record["NORAD_CAT_ID"])
    except (TypeError, ValueError) as exc:
        raise RecordRejected("NORAD_CAT_ID is not an integer") from exc
    if norad <= 0:
        raise RecordRejected("NORAD_CAT_ID must be positive")

    cleaned = {
        "OBJECT_NAME": name,
        "OBJECT_ID": object_id,
        "EPOCH": epoch_text,
        "MEAN_MOTION": mean_motion,
        "ECCENTRICITY": eccentricity,
        "INCLINATION": inclination,
        "RA_OF_ASC_NODE": raan,
        "ARG_OF_PERICENTER": argp,
        "MEAN_ANOMALY": anomaly,
        "EPHEMERIS_TYPE": int(record["EPHEMERIS_TYPE"]),
        "CLASSIFICATION_TYPE": str(record["CLASSIFICATION_TYPE"]).strip(),
        "NORAD_CAT_ID": norad,
        "ELEMENT_SET_NO": int(record["ELEMENT_SET_NO"]),
        "REV_AT_EPOCH": int(record["REV_AT_EPOCH"]),
        "BSTAR": bstar,
        "MEAN_MOTION_DOT": ndot,
        "MEAN_MOTION_DDOT": nddot,
    }
    return {
        "record": cleaned,
        "content_hash": canonical_hash(cleaned),
        "dedup_key": {"norad_cat_id": norad, "epoch": epoch_text},
        "element_description": {
            "status": "modelled",
            "center": GP_CENTER,
            "reference_frame": GP_REFERENCE_FRAME,
            "time_system": GP_TIME_SYSTEM,
            "mean_element_theory": GP_MEAN_ELEMENT_THEORY,
            "gravity_constants": "WGS72",
            "source_url": CELESTRAK_GP_DOC,
            "note": (
                "Brouwer mean elements for SGP4. CelesTrak JSON omits "
                "CENTER_NAME, REF_FRAME, TIME_SYSTEM, and MEAN_ELEMENT_THEORY; "
                "the formats page states those omitted values as EARTH, TEME, "
                "UTC, and SGP4. These are not osculating elements and they "
                "are not a calibrated covariance."
            ),
        },
        "covariance": {
            "status": "absent",
            "value": None,
            "note": "Public GP/OMM does not supply a calibrated covariance.",
        },
        "units": {
            "MEAN_MOTION": "rev/day",
            "MEAN_MOTION_DOT": "rev/day^2",
            "MEAN_MOTION_DDOT": "rev/day^3",
            "ECCENTRICITY": "1",
            "INCLINATION": "deg",
            "RA_OF_ASC_NODE": "deg",
            "ARG_OF_PERICENTER": "deg",
            "MEAN_ANOMALY": "deg",
            "BSTAR": "1/Earth radii",
            "EPOCH": "UTC",
        },
    }
