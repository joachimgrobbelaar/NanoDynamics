"""OrbitGuard monitoring: normalization, frames, freshness, and honest risk gaps."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sgp4.api import Satrec, WGS72, jday
from sgp4.conveniences import sat_epoch_datetime

from orbitguard.monitor.api import build_test_router
from orbitguard.monitor.environment import build_environment, parse_f107, parse_kp
from orbitguard.monitor.frames import gmst_deg, rotation_teme_to_display, teme_to_display
from orbitguard.monitor.gp_omm import RecordRejected, canonical_hash, validate_omm
from orbitguard.monitor.ingest import HttpResult
from orbitguard.monitor.propagation import (
    MeanElementsNotOsculating,
    propagate_osculating_cartesian,
    propagate_sgp4,
    satrec_from_tle,
    two_body_max_position_error_m,
)
from orbitguard.monitor.risk import assess_covariance
from orbitguard.monitor.service import MonitorService
from orbitguard.monitor.tle_compat import checksum_matches, decode_alpha5, encode_alpha5
from orbitguard.monitor.constants import NUMERICAL_FRAME

# Vallado / C++ SGP4 verification element and the t = 0 TEME state shipped
# with python-sgp4 (tcppver.out). This checks the library wiring. It is not
# an operational accuracy claim and the element is not a current catalog set.
_VER_L1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
_VER_L2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
_VER_R_KM = np.array([7022.46529266, -1400.08296755, 0.03995155])


def _client(tmp_path, transport=None):
    service = MonitorService(tmp_path, transport=transport or (lambda url: HttpResult(503, b"", url)))
    app = FastAPI()
    app.include_router(build_test_router(service))
    return TestClient(app), service


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


def test_tle_checksum_accepts_verification_line_and_rejects_a_flip():
    assert len(_VER_L1) == 69
    assert checksum_matches(_VER_L1)
    assert checksum_matches(_VER_L2[:69])
    broken = _VER_L1[:68] + ("0" if _VER_L1[68] != "0" else "1")
    assert checksum_matches(broken) is False


def test_omm_normalization_units_frame_and_absent_covariance():
    record = {
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
    validated = validate_omm(record)
    assert validated["element_description"]["reference_frame"] == "TEME"
    assert validated["element_description"]["time_system"] == "UTC"
    assert validated["element_description"]["mean_element_theory"] == "SGP4"
    assert validated["element_description"]["status"] == "modelled"
    assert validated["units"]["MEAN_MOTION"] == "rev/day"
    assert validated["units"]["INCLINATION"] == "deg"
    assert validated["covariance"]["status"] == "absent"
    assert validated["covariance"]["value"] is None
    assert canonical_hash(validated["record"]) == validate_omm(record)["content_hash"]
    with pytest.raises(RecordRejected):
        validate_omm({**record, "password": "should-not-store"})


def test_ingest_is_idempotent_and_hides_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACETRACK_USER", "operator-name")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "s3cr3t-token-value")
    client, service = _client(tmp_path)
    record = {
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
    first = client.post("/api/orbitguard/ingest", json={"records": [record], "source_name": "unit"}).json()
    second = client.post("/api/orbitguard/ingest", json={"records": [record], "source_name": "unit"}).json()
    secret = client.post(
        "/api/orbitguard/ingest",
        json={"records": [{**record, "EPOCH": "2026-11-05T00:00:00", "token": "s3cr3t-token-value"}]},
    ).json()
    assert first["inserted_count"] == 1
    assert second["inserted_count"] == 0
    assert second["duplicate_count"] == 1
    assert secret["rejected_count"] == 1
    blob = json.dumps({"first": first, "second": second, "secret": secret, "status": client.get("/api/orbitguard/status").json()})
    assert "s3cr3t-token-value" not in blob
    assert "operator-name" not in blob
    status = client.get("/api/orbitguard/status").json()
    assert status["space_track"]["configured"] is True
    assert status["space_track"]["fetch_performed"] is False
    assert status["credentials_exposed"] is False
    assert service.status()["collision_probability"] == "not_computable"


def test_public_fetch_is_one_group_and_does_not_retry_403(tmp_path):
    calls = []

    def transport(url: str) -> HttpResult:
        calls.append(url)
        return HttpResult(status=403, body=b"GP data has not updated", url=url)

    client, _service = _client(tmp_path, transport)
    first = client.post("/api/orbitguard/catalog/refresh").json()
    second = client.post("/api/orbitguard/catalog/refresh").json()
    assert first["network_request"] is True
    assert first["http_status"] == 403
    assert first["calls"] == 1
    assert second["network_request"] is False
    assert second["reason"] == "blocked_response_not_retried"
    assert calls == ["https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=JSON"]
    assert "CATNR" not in calls[0]
    assert "s3cr3t" not in json.dumps(first)


def test_epoch_age_is_not_retrieval_age_and_sample_is_labelled(tmp_path):
    client, _service = _client(tmp_path)
    payload = client.get(
        "/api/orbitguard/workspace",
        params={"mode": "simulation_sample", "epoch_stale_s": 3600, "retrieval_stale_s": 10},
    ).json()
    assert payload["mode"] == "simulation_sample"
    assert "Simulation sample" in payload["mode_label"]
    assert payload["objects"]
    for obj in payload["objects"]:
        assert obj["mode"] == "simulation_sample"
        assert obj["synthetic"] is True
        assert obj["norad_cat_id"] is None
        assert "Simulation sample" in obj["mode_label"]
        assert obj["pc"] is None
        assert obj["pc_status"] == "not_computable"
        inclination = next(row for row in obj["provenance"] if row["name"] == "inclination")
        assert inclination["unit"] == "deg"
        assert inclination["reference_frame"] == "TEME"
        assert inclination["status"] == "modelled"
        assert inclination["epoch_age_s"] > 3600
        assert inclination["epoch_freshness"] == "older_than_display_setting"
        assert inclination["retrieval_age_s"] is None
        assert inclination["retrieval_freshness"] == "unknown"
        assert inclination["observation_time"] != payload["now"]
    altitude = next(row for row in payload["objects"][0]["provenance"] if row["name"] == "geodetic_altitude")
    assert altitude["status"] == "derived"
    assert altitude["unit"] == "km"
    assert altitude["value"] is not None
    # State is held at the element epoch, so its observation time is not "now".
    assert altitude["observation_time"].startswith("2024-06-01")


def test_missing_and_invalid_covariance_are_not_zero_or_safe():
    absent = assess_covariance(None, state_epoch=None, covariance_epoch=None, state_frame=None, covariance_frame=None)
    assert absent["status"] == "absent"
    assert absent["pc"] is None
    assert absent["pc_status"] == "not_computable"
    assert absent["nc"] is None
    assert absent["displayed_as_safe"] is False
    invalid = assess_covariance(
        [[1.0, 0.0, 0.0], [0.0, -5.0, 0.0], [0.0, 0.0, 1.0]],
        state_epoch="2024-06-01T00:00:00.000000",
        covariance_epoch="2024-06-01T00:00:00.000000",
        state_frame="TEME",
        covariance_frame="TEME",
    )
    assert invalid["status"] == "invalid"
    assert invalid["pc"] is None
    assert invalid["positive_semidefinite"] is False
    valid = assess_covariance(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        state_epoch="2024-06-01T00:00:00.000000",
        covariance_epoch="2024-06-01T00:00:00.000000",
        state_frame="TEME",
        covariance_frame="TEME",
    )
    assert valid["status"] == "present_not_propagated"
    assert valid["pc"] is None
    assert valid["nc_status"] == "not_implemented"
    assert "expected number" in valid["nc_reason"].lower() or "expected number of collisions" in valid["nc_reason"]
    mismatched = assess_covariance(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        state_epoch="2024-06-01T00:00:00.000000",
        covariance_epoch="2024-06-02T00:00:00.000000",
        state_frame="TEME",
        covariance_frame="TEME",
    )
    assert mismatched["status"] == "invalid"
    assert mismatched["pc"] is None


def test_sample_screen_reports_geometry_without_a_probability(tmp_path):
    client, _service = _client(tmp_path)
    response = client.post(
        "/api/orbitguard/conjunction/screen",
        json={
            "mode": "simulation_sample",
            "primary_id": "sample:leo-a",
            "secondary_id": "sample:leo-b",
            "window_minutes": 30,
            "threshold_km": 50,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode_label"].startswith("Simulation sample")
    assert body["geometric_miss_km"] is not None
    assert body["geometric_miss_km"] > 0.0
    assert body["pc"] is None
    assert body["pc_status"] == "not_computable"
    assert body["nc"] is None
    assert body["nc_status"] == "not_implemented"
    assert body["safety_determination"] == "not_made"
    assert body["risk_tier"] is None
    assert body["covariance"]["status"] == "absent"
    assert "0" != str(body["pc"])


def test_mean_elements_are_rejected_by_the_numerical_integrator():
    with pytest.raises(MeanElementsNotOsculating):
        propagate_osculating_cartesian(
            np.array([7000e3, 0.0, 0.0]),
            np.array([0.0, 7500.0, 0.0]),
            60.0,
            30.0,
            state_kind="gp",
            frame=NUMERICAL_FRAME,
        )
    with pytest.raises(MeanElementsNotOsculating):
        propagate_osculating_cartesian(
            np.array([7000e3, 0.0, 0.0]),
            np.array([0.0, 7500.0, 0.0]),
            60.0,
            30.0,
            state_kind="osculating_cartesian",
            frame="TEME",
        )


def test_sgp4_matches_published_verification_vector_at_epoch():
    satellite = satrec_from_tle(_VER_L1, _VER_L2[:69])
    epoch = sat_epoch_datetime(satellite).replace(tzinfo=timezone.utc)
    direct = Satrec.twoline2rv(_VER_L1, _VER_L2[:69], WGS72)
    jd, fraction = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second + epoch.microsecond * 1e-6)
    code, radius, _velocity = direct.sgp4(jd, fraction)
    assert code == 0
    propagated = propagate_sgp4(satellite, epoch, 0.0, 60.0)
    assert propagated["reference_frame"] == "TEME"
    assert propagated["status"] == "derived"
    got = np.asarray(propagated["state"]["r_teme_km"])
    assert np.allclose(got, radius, atol=1e-8)
    # Independent of our wrapper: the published verification coordinate.
    assert np.allclose(got, _VER_R_KM, atol=1e-6)


def test_numerical_two_body_baseline_and_j2_sign():
    error_m = two_body_max_position_error_m(duration_s=600.0, step_s=60.0)
    assert error_m < 20.0
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
    assert result["mode_label"].startswith("Simulation")
    assert result["reference_frame"] == NUMERICAL_FRAME
    assert result["atmosphere"] is None
    start = rv_to_coe(position, velocity)
    end_r = np.asarray(result["samples"][-1]["r_km"]) * 1000.0
    end_v = np.asarray(result["samples"][-1]["v_km_s"]) * 1000.0
    end = rv_to_coe(end_r, end_v)
    # Unwrap the osculating node. Secular theory is not the osculating element,
    # so the test only requires the same sign and a comparable magnitude.
    delta = (end.raan - start.raan + math.pi) % (2.0 * math.pi) - math.pi
    secular = analytical_j2_raan_rate(radius, start.e, start.i) * duration
    assert delta * secular > 0.0
    assert abs(delta) == pytest.approx(abs(secular), rel=0.5)


def test_gmst_and_display_frame():
    moment = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert gmst_deg(moment) == pytest.approx(280.46061837, abs=1e-6)
    rotation = rotation_teme_to_display(90.0)
    rotated = rotation @ np.array([1.0, 0.0, 0.0])
    assert rotated[0] == pytest.approx(0.0, abs=1e-12)
    assert rotated[1] == pytest.approx(-1.0, abs=1e-12)
    display = teme_to_display(np.array([0.0, 0.0, 7000.0]), np.array([0.0, 0.0, 0.0]), moment)
    assert display["frame"] == "teme_gmst_earth_fixed"
    assert display["status"] == "derived"
    assert display["r_km"][2] == pytest.approx(7000.0, abs=1e-6)
    assert display["latitude_deg"] == pytest.approx(90.0, abs=0.02)
    assert "not ITRF" in display["frame_note"]


def test_environment_units_and_no_filled_density():
    now = datetime(2026, 11, 6, 6, 0, tzinfo=timezone.utc)
    f107 = parse_f107(
        [
            {"time_tag": "2026-11-05T22:00:00", "frequency": 2800, "flux": 122.0, "reporting_schedule": "Afternoon"},
            {"time_tag": "2026-09-26T17:00:00", "frequency": 2800, "flux": 100.0, "reporting_schedule": "Morning"},
        ]
    )
    assert f107["flux"] == 122.0
    assert f107["frequency_mhz"] == 2800
    kp = parse_kp(
        [
            {"time_tag": "2026-11-06T06:05:00", "kp_index": 0, "estimated_kp": 0.33, "kp": "0Z"},
        ]
    )
    payload = build_environment(
        now=now,
        retrieval_time=now,
        f107=f107,
        kp=kp,
        f107_error=None,
        kp_error=None,
    )
    assert payload["f107"]["unit"].startswith("sfu")
    assert payload["f107"]["status"] == "measured"
    assert payload["f107"]["value"] == 122.0
    assert payload["planetary_k"]["unit"].startswith("1")
    assert payload["planetary_k"]["status"] == "derived"
    assert payload["planetary_k"]["value"] == 0.33
    assert payload["tec"]["status"] == "unavailable"
    assert payload["tec"]["value"] is None
    assert payload["thermospheric_density"]["status"] == "unavailable"
    assert payload["thermospheric_density"]["value"] is None
    assert "column" in payload["tec"]["note"].lower() or "integrated" in payload["tec"]["note"].lower()
    empty = build_environment(
        now=now,
        retrieval_time=None,
        f107=None,
        kp=None,
        f107_error="down",
        kp_error="down",
    )
    assert empty["f107"]["value"] is None
    assert empty["planetary_k"]["value"] is None
    assert empty["f107"]["status"] == "unavailable"


def test_sensor_copy_does_not_claim_a_universal_laser_range(tmp_path):
    client, _service = _client(tmp_path)
    sensors = client.get("/api/orbitguard/sensors").json()
    blob = json.dumps(sensors)
    assert "Laser Clearinghouse" in blob
    assert "South African" in blob
    optical = next(item for item in sensors["sensors"] if item["id"] == "optical_track")
    assert "range" in optical["does_not_measure"].lower()
    laser = next(item for item in sensors["sensors"] if item["id"] == "debris_laser_ranging")
    assert "universal" in laser["does_not_measure"].lower()
    assert sensors["feed_status"] == "disconnected"
    roadmap = client.get("/api/orbitguard/roadmap").json()
    assert any("SGP4" in item for item in roadmap["implemented"])
    assert any("Pc" in item or "Nc" in item for item in roadmap["not_implemented"])


def test_simulation_response_is_labelled(tmp_path):
    client, _service = _client(tmp_path)
    radius = 6378137.0 + 500_000.0
    speed = math.sqrt(3.986004418e14 / radius)
    response = client.post(
        "/api/orbitguard/simulation/propagate",
        json={
            "position_m": [radius, 0.0, 0.0],
            "velocity_m_s": [0.0, speed, 0.0],
            "duration_s": 120.0,
            "step_s": 60.0,
            "state_kind": "osculating_cartesian",
            "frame": "nanodynamics_eci",
            "include_j2": False,
            "include_drag": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "osculating_simulation"
    assert "Simulation" in body["mode_label"]
    assert body["reference_frame"] == "nanodynamics_eci"
    rejected = client.post(
        "/api/orbitguard/simulation/propagate",
        json={
            "position_m": [radius, 0.0, 0.0],
            "velocity_m_s": [0.0, speed, 0.0],
            "state_kind": "tle",
            "frame": "nanodynamics_eci",
        },
    )
    assert rejected.status_code == 422


def test_page_marks_sample_mode_and_does_not_claim_a_live_subscription():
    html = open("visualization_3d/static/orbitguard.html", encoding="utf-8").read()
    assert "OrbitGuard" in html
    assert "Simulation sample" in html
    assert "not computable" in html
    assert "Subscribed to CelesTrak" not in html
    assert "Autonomous Conjunction" not in html
    assert "@media" in html
    assert "overflow-x" in html
    assert 'name="viewport"' in html
