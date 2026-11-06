"""Two propagation paths that are not interchangeable.

* GP/TLE/OMM mean elements go through SGP4 and remain TEME until the
  display transform.
* ``leo_simulator.OrbitPropagator`` accepts only a Cartesian state the
  caller declares as osculating, in ``nanodynamics_eci``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sgp4.api import Satrec, WGS72, jday

from leo_simulator.constants import MU_EARTH, R_EARTH
from leo_simulator.propagator import OrbitPropagator
from orbitguard.monitor.constants import (
    NUMERICAL_FRAME,
    NUMERICAL_FRAME_NOTE,
    SGP4_GRAVITY,
)
from orbitguard.monitor.frames import teme_to_display
from orbitguard.monitor.gp_omm import normalize_epoch_text
from orbitguard.monitor.provenance import as_utc


class MeanElementsNotOsculating(ValueError):
    """Raised when mean elements are offered to the numerical integrator."""


def satrec_from_omm(fields: dict[str, Any]) -> Satrec:
    """Initialize SGP4 from validated OMM fields using WGS-72 constants."""
    from sgp4.omm import initialize

    prepared = dict(fields)
    prepared["EPOCH"] = normalize_epoch_text(str(fields["EPOCH"]))
    prepared["OBJECT_ID"] = str(fields["OBJECT_ID"])
    prepared["CLASSIFICATION_TYPE"] = str(fields["CLASSIFICATION_TYPE"])
    satellite = Satrec()
    initialize(satellite, prepared, WGS72)
    return satellite


def satrec_from_tle(line1: str, line2: str) -> Satrec:
    return Satrec.twoline2rv(line1, line2, WGS72)


def _propagate_times(satellite: Satrec, moments: list[datetime]) -> dict[str, Any]:
    positions = []
    velocities = []
    used = []
    errors = []
    for moment in moments:
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
        code, radius_km, velocity_km_s = satellite.sgp4(jd, fraction)
        if code != 0:
            errors.append({"time": moment.isoformat().replace("+00:00", "Z"), "sgp4_error": int(code)})
            continue
        positions.append(np.asarray(radius_km, dtype=np.float64))
        velocities.append(np.asarray(velocity_km_s, dtype=np.float64))
        used.append(moment)
    return {
        "theory": "SGP4",
        "gravity_constants": SGP4_GRAVITY,
        "reference_frame": "TEME",
        "time_system": "UTC",
        "status": "derived" if positions else "unavailable",
        "times": used,
        "r_km": positions,
        "v_km_s": velocities,
        "errors": errors,
    }


def propagate_sgp4(
    satellite: Satrec,
    start: datetime,
    duration_s: float,
    step_s: float,
) -> dict[str, Any]:
    if duration_s < 0.0 or step_s <= 0.0:
        raise ValueError("SGP4 duration must be >= 0 and step must be positive")
    count = int(np.floor(duration_s / step_s)) + 1
    moments = [as_utc(start) + timedelta(seconds=step_s * index) for index in range(count)]
    assert moments[0] is not None
    raw = _propagate_times(satellite, moments)  # type: ignore[arg-type]
    track = []
    for moment, radius, velocity in zip(raw["times"], raw["r_km"], raw["v_km_s"]):
        display = teme_to_display(radius, velocity, moment)
        track.append(
            {
                "time": moment.isoformat().replace("+00:00", "Z"),
                "latitude_deg": display["latitude_deg"],
                "longitude_deg": display["longitude_deg"],
                "altitude_km": display["altitude_km"],
                "r_teme_km": radius.tolist(),
                "v_teme_km_s": velocity.tolist(),
            }
        )
    state = None
    if track:
        state = {
            "time": track[0]["time"],
            "r_teme_km": track[0]["r_teme_km"],
            "v_teme_km_s": track[0]["v_teme_km_s"],
            "latitude_deg": track[0]["latitude_deg"],
            "longitude_deg": track[0]["longitude_deg"],
            "altitude_km": track[0]["altitude_km"],
            "reference_frame": "TEME",
            "display_frame": "teme_gmst_earth_fixed",
            "status": "derived",
            "unit_position": "km",
            "unit_velocity": "km/s",
        }
    return {
        "theory": raw["theory"],
        "gravity_constants": raw["gravity_constants"],
        "reference_frame": "TEME",
        "status": raw["status"],
        "state": state,
        "track": track,
        "errors": raw["errors"],
        "note": (
            "Position is the SGP4 result for these mean elements. It is not "
            "an independent measurement and it is not a covariance."
        ),
    }


def propagate_osculating_cartesian(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    duration_s: float,
    step_s: float,
    *,
    state_kind: str,
    frame: str,
    include_j2: bool = True,
    include_drag: bool = False,
) -> dict[str, Any]:
    """Run the NanoDynamics numerical propagator on an osculating state.

    ``state_kind`` must be ``osculating_cartesian``. Mean-element labels are
    rejected so a TLE is not integrated as if it were osculating.
    """
    if state_kind != "osculating_cartesian":
        raise MeanElementsNotOsculating(
            "The numerical integrator accepts an osculating Cartesian state "
            "only. GP, TLE, and OMM mean elements must be propagated with SGP4."
        )
    if frame != NUMERICAL_FRAME:
        raise MeanElementsNotOsculating(
            "Refusing to integrate a state that is not labelled "
            f"{NUMERICAL_FRAME}. SGP4 TEME states are not transformed into "
            "this frame in this version."
        )
    position_m = np.asarray(position_m, dtype=np.float64)
    velocity_m_s = np.asarray(velocity_m_s, dtype=np.float64)
    if position_m.shape != (3,) or velocity_m_s.shape != (3,):
        raise ValueError("Osculating position and velocity must each have shape (3,)")
    state = np.concatenate([position_m, velocity_m_s])
    propagator = OrbitPropagator(
        include_central_gravity=True,
        include_j2=include_j2,
        include_drag=include_drag,
        include_earth_rotation=include_drag,
        include_moon=False,
        include_srp=False,
        solver_method="DOP853",
    )
    result = propagator.propagate(state, duration_seconds=duration_s, t_start=0.0, dt_eval=step_s)
    radius_km = result.r / 1000.0
    velocity_km_s = result.v / 1000.0
    samples = []
    for index, time_s in enumerate(result.t):
        samples.append(
            {
                "time_offset_s": float(time_s),
                "r_km": radius_km[index].tolist(),
                "v_km_s": velocity_km_s[index].tolist(),
                "altitude_km": float(result.altitudes[index] / 1000.0),
            }
        )
    atmosphere = None
    if include_drag:
        atmosphere = {
            "model": "exponential",
            "status": "modelled",
            "h0_m": propagator.atmosphere.h0,
            "rho0_kg_m3": propagator.atmosphere.rho0,
            "scale_height_m": propagator.atmosphere.scale_height,
            "unit": "kg m^-3",
            "note": (
                "Single-scale-height exponential density inside the numerical "
                "model. It is not NRLMSISE-00, not JB2008, and not a density "
                "observation. It is separate from ionospheric TEC."
            ),
        }
    return {
        "mode": "osculating_simulation",
        "mode_label": "Simulation — osculating numerical propagation, not a catalog state",
        "propagator": "leo_simulator.OrbitPropagator",
        "solver": propagator.solver_method,
        "reference_frame": NUMERICAL_FRAME,
        "frame_note": NUMERICAL_FRAME_NOTE,
        "include_j2": include_j2,
        "include_drag": include_drag,
        "include_moon": False,
        "include_srp": False,
        "mu_m3_s2": MU_EARTH,
        "equatorial_radius_m": R_EARTH,
        "gravity_model_note": "Point mass plus optional J2. Not a geopotential to degree and order 4, despite older site copy.",
        "atmosphere": atmosphere,
        "success": bool(result.success),
        "samples": samples,
        "status": "derived",
    }


def two_body_max_position_error_m(duration_s: float = 900.0, step_s: float = 30.0) -> float:
    """Max |numerical − Keplerian| position error for a circular orbit.

    J2, drag, Moon, and SRP are off. This checks the reused integrator
    against the closed-form two-body solution. It is not an SGP4 test and
    it is not a PINN accuracy claim.
    """
    radius_m = R_EARTH + 500_000.0
    speed = float(np.sqrt(MU_EARTH / radius_m))
    position = np.array([radius_m, 0.0, 0.0])
    velocity = np.array([0.0, speed, 0.0])
    result = propagate_osculating_cartesian(
        position,
        velocity,
        duration_s,
        step_s,
        state_kind="osculating_cartesian",
        frame=NUMERICAL_FRAME,
        include_j2=False,
        include_drag=False,
    )
    mean_motion = speed / radius_m
    worst = 0.0
    for sample in result["samples"]:
        theta = mean_motion * sample["time_offset_s"]
        expected = radius_m * np.array([np.cos(theta), np.sin(theta), 0.0])
        got = np.asarray(sample["r_km"], dtype=np.float64) * 1000.0
        worst = max(worst, float(np.linalg.norm(got - expected)))
    return worst
