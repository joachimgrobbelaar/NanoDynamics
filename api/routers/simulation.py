"""Ephemeris propagation and orbital mechanics simulation endpoints."""

from __future__ import annotations

import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
import numpy as np
from pydantic import BaseModel, Field

from core.dynamics import (
    OrbitalElements,
    coe_to_rv,
    ecef_to_geodetic,
    eci_to_ecef,
    rv_to_coe,
)
from core.propagator import (
    NumericalCowellPropagator,
    SGP4Propagator,
    TrajectoryOutput,
)

router = APIRouter(prefix="/api/simulation", tags=["Simulation & Propagation"])


class StateVectorRequest(BaseModel):
    r0: List[float] = Field(..., description="Initial ECI position vector [x, y, z] in meters")
    v0: List[float] = Field(..., description="Initial ECI velocity vector [vx, vy, vz] in m/s")
    duration_seconds: float = Field(5400.0, description="Propagation duration in seconds")
    step_seconds: float = Field(30.0, description="Time step resolution in seconds")
    mass_kg: float = Field(4.0, description="Satellite wet mass in kg")
    drag_area_m2: float = Field(0.03, description="Cross-sectional aerodynamic drag area in m^2")
    cd: float = Field(2.2, description="Drag coefficient")
    include_j2: bool = Field(True, description="Enable J2 oblateness harmonic")
    include_drag: bool = Field(True, description="Enable thermospheric drag")


class TLEPropagationRequest(BaseModel):
    line1: str = Field(..., description="TLE Line 1")
    line2: str = Field(..., description="TLE Line 2")
    duration_seconds: float = Field(5400.0, description="Propagation duration in seconds")
    step_seconds: float = Field(30.0, description="Time step in seconds")


class OrbitalElementsRequest(BaseModel):
    a: float = Field(..., description="Semi-major axis in meters")
    e: float = Field(..., description="Eccentricity")
    i_deg: float = Field(..., description="Inclination in degrees")
    raan_deg: float = Field(..., description="Right Ascension of Ascending Node in degrees")
    arg_pe_deg: float = Field(..., description="Argument of Periapsis in degrees")
    true_anomaly_deg: float = Field(..., description="True Anomaly in degrees")


@router.post("/propagate")
async def propagate_numerical(request: StateVectorRequest):
    """Propagate an orbit using high-precision Cowell numerical integration with J2 and drag."""
    if len(request.r0) != 3 or len(request.v0) != 3:
        raise HTTPException(status_code=400, detail="State vectors r0 and v0 must have exactly 3 components.")

    try:
        propagator = NumericalCowellPropagator(
            mass_kg=request.mass_kg,
            drag_area_m2=request.drag_area_m2,
            cd=request.cd,
            include_j2=request.include_j2,
            include_drag=request.include_drag,
        )

        traj = propagator.propagate(
            r0=np.array(request.r0, dtype=np.float64),
            v0=np.array(request.v0, dtype=np.float64),
            duration_seconds=request.duration_seconds,
            step_seconds=request.step_seconds,
        )

        # Build trajectory samples
        samples = []
        step_sub = max(1, len(traj.t) // 100)  # Return up to 100 trajectory points for low bandwidth
        for idx in range(0, len(traj.t), step_sub):
            r_eci = traj.r[idx]
            r_ecef = eci_to_ecef(r_eci, float(traj.t[idx]))
            lat, lon, alt = ecef_to_geodetic(r_ecef)
            samples.append({
                "time_s": float(traj.t[idx]),
                "position_eci": r_eci.tolist(),
                "velocity_eci": traj.v[idx].tolist(),
                "altitude_km": float(traj.altitudes[idx] / 1000.0),
                "latitude_deg": round(lat, 4),
                "longitude_deg": round(lon, 4),
            })

        return {
            "success": traj.success,
            "reentry_detected": traj.reentry,
            "total_points": len(traj.t),
            "sampled_points": len(samples),
            "trajectory": samples,
            "initial_elements": traj.elements[0].to_dict() if traj.elements else None,
            "final_elements": traj.elements[-1].to_dict() if traj.elements else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Propagation engine error: {str(exc)}")


@router.post("/propagate-tle")
async def propagate_tle(request: TLEPropagationRequest):
    """Propagate an ephemeris using analytical SGP4 algorithm from Two-Line Elements (TLE)."""
    try:
        sgp = SGP4Propagator(request.line1, request.line2)
        start_dt = datetime.datetime.now(datetime.timezone.utc)
        traj = sgp.propagate(
            start_datetime=start_dt,
            duration_seconds=request.duration_seconds,
            step_seconds=request.step_seconds,
        )

        if not traj.success or len(traj.t) == 0:
            raise HTTPException(status_code=400, detail="Invalid TLE or SGP4 propagation failed.")

        samples = []
        step_sub = max(1, len(traj.t) // 100)
        for idx in range(0, len(traj.t), step_sub):
            r_eci = traj.r[idx]
            r_ecef = eci_to_ecef(r_eci, float(traj.t[idx]))
            lat, lon, alt = ecef_to_geodetic(r_ecef)
            samples.append({
                "time_s": float(traj.t[idx]),
                "position_eci": r_eci.tolist(),
                "velocity_eci": traj.v[idx].tolist(),
                "altitude_km": float(traj.altitudes[idx] / 1000.0),
                "latitude_deg": round(lat, 4),
                "longitude_deg": round(lon, 4),
            })

        return {
            "success": True,
            "total_points": len(traj.t),
            "sampled_points": len(samples),
            "trajectory": samples,
            "initial_elements": traj.elements[0].to_dict() if traj.elements else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SGP4 propagation error: {str(exc)}")


@router.post("/elements-to-rv")
async def convert_elements_to_rv(elements: OrbitalElementsRequest):
    """Convert Classical Orbital Elements to ECI Cartesian state vectors (r, v)."""
    try:
        coe = OrbitalElements(
            a=elements.a,
            e=elements.e,
            i=np.radians(elements.i_deg),
            raan=np.radians(elements.raan_deg),
            arg_pe=np.radians(elements.arg_pe_deg),
            true_anomaly=np.radians(elements.true_anomaly_deg),
        )
        r_vec, v_vec = coe_to_rv(coe)
        return {
            "position_eci_m": r_vec.tolist(),
            "velocity_eci_m_s": v_vec.tolist(),
            "period_minutes": float(coe.period / 60.0),
            "perigee_altitude_km": float(coe.perigee_altitude / 1000.0),
            "apogee_altitude_km": float(coe.apogee_altitude / 1000.0),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Element conversion error: {str(exc)}")


@router.get("/presets")
async def get_orbital_presets():
    """Return standard reference LEO satellite orbits (ISS, SSO, Walker constellation)."""
    return {
        "presets": [
            {
                "name": "International Space Station (ISS)",
                "altitude_km": 420.0,
                "a_m": 6798137.0,
                "e": 0.0003,
                "i_deg": 51.64,
                "raan_deg": 120.0,
                "arg_pe_deg": 45.0,
                "true_anomaly_deg": 0.0,
            },
            {
                "name": "Sun-Synchronous LEO (SSO Earth Observation)",
                "altitude_km": 550.0,
                "a_m": 6928137.0,
                "e": 0.001,
                "i_deg": 97.5,
                "raan_deg": 30.0,
                "arg_pe_deg": 90.0,
                "true_anomaly_deg": 0.0,
            },
            {
                "name": "Starlink Shell 1 (Broadband Constellation)",
                "altitude_km": 550.0,
                "a_m": 6928137.0,
                "e": 0.0001,
                "i_deg": 53.0,
                "raan_deg": 65.0,
                "arg_pe_deg": 0.0,
                "true_anomaly_deg": 0.0,
            },
        ]
    }
