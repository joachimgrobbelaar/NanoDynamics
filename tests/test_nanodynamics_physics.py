"""Public physics API: elements, frames, and the two propagation paths."""

from __future__ import annotations

from datetime import datetime, timezone
import math

import numpy as np
import pytest
from sgp4.api import Satrec, WGS72, jday
from sgp4.conveniences import sat_epoch_datetime

from nanodynamics.constants import NUMERICAL_FRAME
from nanodynamics.elements import RecordRejected, canonical_hash, checksum_matches, decode_alpha5, encode_alpha5, validate_omm
from nanodynamics.frames import gmst_deg, rotation_teme_to_display, teme_to_display
from nanodynamics.links import ORBITGUARD_URL
from nanodynamics.propagation import (
    MeanElementsNotOsculating,
    propagate_osculating_cartesian,
    propagate_sgp4,
    satrec_from_tle,
    two_body_max_position_error_m,
)

_VER_L1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
_VER_L2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
_VER_R_KM = np.array([7022.46529266, -1400.08296755, 0.03995155])

_OMM = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2026-11-04T21:15:15.204960",
    "MEAN_MOTION": 15.49180468,
    "ECCENTRICITY": 0.00072122,
    "INCLINATION": 51.6327,
    "RA_OF_ASC_NODE": 323.8448,
    "ARG_OF_PERICENTER": 348.649,
    "MEAN_ANOMALY": 11.4335,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 25544,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 59356,
    "BSTAR": 0.00010456704,
    "MEAN_MOTION_DOT": 5.352e-05,
    "MEAN_MOTION_DDOT": 0,
}


def test_alpha5_strict_roundtrip():
    assert encode_alpha5(25544) == "25544"
    assert decode_alpha5("25544") == 25544
    assert encode_alpha5(100000) == "A0000"
    assert decode_alpha5("A0000") == 100000
    assert encode_alpha5(180000) == "J0000"
    assert decode_alpha5("J0000") == 180000
    assert encode_alpha5(339999) == "Z9999"
    assert decode_alpha5("Z9999") == 339999
    with pytest.raises(ValueError):
        decode_alpha5("I0000")
    with pytest.raises(ValueError):
        decode_alpha5("O0000")
    with pytest.raises(ValueError):
        decode_alpha5("a0000")
    with pytest.raises(ValueError):
        encode_alpha5(340000)


def test_tle_checksum():
    assert len(_VER_L1) == 69
    assert checksum_matches(_VER_L1)
    assert checksum_matches(_VER_L2[:69])
    broken = _VER_L1[:68] + ("0" if _VER_L1[68] != "0" else "1")
    assert checksum_matches(broken) is False


def test_omm_is_mean_elements_without_covariance():
    validated = validate_omm(_OMM)
    assert validated["element_description"]["reference_frame"] == "TEME"
    assert validated["element_description"]["time_system"] == "UTC"
    assert validated["element_description"]["mean_element_theory"] == "SGP4"
    assert validated["element_description"]["status"] == "modelled"
    assert validated["units"]["MEAN_MOTION"] == "rev/day"
    assert validated["covariance"]["status"] == "absent"
    assert validated["covariance"]["value"] is None
    assert canonical_hash(validated["record"]) == validate_omm(_OMM)["content_hash"]
    with pytest.raises(RecordRejected):
        validate_omm({**_OMM, "password": "should-not-store"})


def test_sgp4_matches_published_verification_vector_at_epoch():
    satellite = satrec_from_tle(_VER_L1, _VER_L2[:69])
    epoch = sat_epoch_datetime(satellite).replace(tzinfo=timezone.utc)
    direct = Satrec.twoline2rv(_VER_L1, _VER_L2[:69], WGS72)
    jd, fraction = jday(
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second + epoch.microsecond * 1e-6,
    )
    code, radius, _velocity = direct.sgp4(jd, fraction)
    assert code == 0
    propagated = propagate_sgp4(satellite, epoch, 0.0, 60.0)
    assert propagated["reference_frame"] == "TEME"
    assert propagated["gravity_constants"] == "WGS72"
    got = np.asarray(propagated["state"]["r_teme_km"])
    assert np.allclose(got, radius, atol=1e-8)
    assert np.allclose(got, _VER_R_KM, atol=1e-6)


def test_mean_elements_are_rejected_by_the_numerical_integrator():
    state = dict(
        position_m=np.array([7000e3, 0.0, 0.0]),
        velocity_m_s=np.array([0.0, 7500.0, 0.0]),
        duration_s=60.0,
        step_s=30.0,
    )
    with pytest.raises(MeanElementsNotOsculating):
        propagate_osculating_cartesian(**state, state_kind="gp", frame=NUMERICAL_FRAME)
    with pytest.raises(MeanElementsNotOsculating):
        propagate_osculating_cartesian(**state, state_kind="osculating_cartesian", frame="TEME")


def test_numerical_two_body_baseline_and_j2_sign():
    assert two_body_max_position_error_m(duration_s=600.0, step_s=60.0) < 20.0
    from leo_simulator.constants import MU_EARTH, R_EARTH
    from leo_simulator.orbit.elements import analytical_j2_raan_rate, rv_to_coe

    radius = R_EARTH + 500_000.0
    speed = math.sqrt(MU_EARTH / radius)
    inclination = math.radians(51.6)
    position = np.array([radius, 0.0, 0.0])
    velocity = speed * np.array([0.0, math.cos(inclination), math.sin(inclination)])
    duration = 3 * 3600.0
    result = propagate_osculating_cartesian(
        position,
        velocity,
        duration,
        120.0,
        state_kind="osculating_cartesian",
        frame=NUMERICAL_FRAME,
        include_j2=True,
        include_drag=False,
    )
    assert result["reference_frame"] == NUMERICAL_FRAME
    assert result["atmosphere"] is None
    start = rv_to_coe(position, velocity)
    end_r = np.asarray(result["samples"][-1]["r_km"]) * 1000.0
    end_v = np.asarray(result["samples"][-1]["v_km_s"]) * 1000.0
    end = rv_to_coe(end_r, end_v)
    delta = (end.raan - start.raan + math.pi) % (2.0 * math.pi) - math.pi
    secular = analytical_j2_raan_rate(radius, start.e, start.i) * duration
    assert delta * secular > 0.0
    assert abs(delta) == pytest.approx(abs(secular), rel=0.5)


def test_gmst_and_display_frame():
    moment = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert gmst_deg(moment) == pytest.approx(280.46061837, abs=1e-6)
    rotated = rotation_teme_to_display(90.0) @ np.array([1.0, 0.0, 0.0])
    assert rotated[0] == pytest.approx(0.0, abs=1e-12)
    assert rotated[1] == pytest.approx(-1.0, abs=1e-12)
    display = teme_to_display(np.array([0.0, 0.0, 7000.0]), np.array([0.0, 0.0, 0.0]), moment)
    assert display["frame"] == "teme_gmst_earth_fixed"
    assert display["r_km"][2] == pytest.approx(7000.0, abs=1e-6)
    assert display["latitude_deg"] == pytest.approx(90.0, abs=0.02)
    assert "not ITRF" in display["frame_note"]


def test_physics_modules_do_not_import_a_web_framework():
    import nanodynamics.elements as elements
    import nanodynamics.frames as frames
    import nanodynamics.propagation as propagation

    for module in (elements, frames, propagation):
        assert "fastapi" not in module.__dict__
        assert "torch" not in module.__dict__
    assert ORBITGUARD_URL == "" or ORBITGUARD_URL.startswith(("http://", "https://"))
