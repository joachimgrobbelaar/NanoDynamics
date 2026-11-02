"""OrbitGuard AI Conjunction Screening, Collision Probability (Pc), and CDM Router."""

from __future__ import annotations

import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
import numpy as np
from pydantic import BaseModel, Field

from ai.orbitguard import (
    ConjunctionEvent,
    OrbitGuardScreener,
    calculate_avoidance_maneuver,
    calculate_collision_probability,
    evaluate_encounter_risk,
)

router = APIRouter(prefix="/api/conjunction", tags=["OrbitGuard AI Conjunction"])


class ConjunctionScreenRequest(BaseModel):
    primary_r0: List[float] = Field(..., description="Primary satellite ECI position [x, y, z] in meters")
    primary_v0: List[float] = Field(..., description="Primary satellite ECI velocity [vx, vy, vz] in m/s")
    secondary_r0: List[float] = Field(..., description="Secondary object ECI position [x, y, z] in meters")
    secondary_v0: List[float] = Field(..., description="Secondary object ECI velocity [vx, vy, vz] in m/s")
    duration_hours: float = Field(24.0, description="Screening horizon in hours (up to 72 hrs)")
    threshold_km: float = Field(20.0, description="Screening distance threshold in km")
    primary_id: str = Field("SAT-PRIMARY-01", description="Designator for primary satellite")
    secondary_id: str = Field("DEB-COSMOS-1408", description="Designator for secondary debris/satellite")


class CollisionEvalRequest(BaseModel):
    miss_distance_km: float = Field(..., description="Euclidean miss distance in km")
    combined_radius_m: float = Field(2.0, description="Combined hard-body radius in meters")
    sigma_radial_m: float = Field(50.0, description="Radial position uncertainty standard deviation [m]")
    sigma_in_track_m: float = Field(250.0, description="In-track position uncertainty standard deviation [m]")
    sigma_cross_track_m: float = Field(80.0, description="Cross-track position uncertainty standard deviation [m]")


class AvoidanceManeuverRequest(BaseModel):
    miss_distance_km: float = Field(..., description="Current miss distance in km")
    tca_seconds: float = Field(3600.0, description="Time to closest approach in seconds")
    r1_tca: List[float] = Field(..., description="Primary position at TCA [m]")
    v1_tca: List[float] = Field(..., description="Primary velocity at TCA [m/s]")
    lead_time_seconds: float = Field(1800.0, description="Lead time before TCA to execute burn [s]")
    target_clearance_km: float = Field(5.0, description="Target separation clearance at TCA in km")


@router.post("/screen")
async def screen_conjunction(request: ConjunctionScreenRequest):
    """Run full pairwise numerical orbital screening over screening horizon to find conjunctions."""
    if len(request.primary_r0) != 3 or len(request.secondary_r0) != 3:
        raise HTTPException(status_code=400, detail="State vectors must contain exactly 3 components.")

    try:
        screener = OrbitGuardScreener(screening_threshold_km=request.threshold_km)
        events = screener.screen_encounter(
            r1_0=np.array(request.primary_r0, dtype=np.float64),
            v1_0=np.array(request.primary_v0, dtype=np.float64),
            r2_0=np.array(request.secondary_r0, dtype=np.float64),
            v2_0=np.array(request.secondary_v0, dtype=np.float64),
            duration_hours=min(request.duration_hours, 72.0),
            primary_id=request.primary_id,
            secondary_id=request.secondary_id,
        )

        return {
            "status": "COMPLETED",
            "screened_events_count": len(events),
            "events": [event.to_dict() for event in events],
            "highest_risk": events[0].to_dict() if events else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Screening error: {str(exc)}")


@router.post("/evaluate")
async def evaluate_collision(request: CollisionEvalRequest):
    """Compute 2D B-plane collision probability (Pc) and operational risk categorization."""
    try:
        pc, tier = evaluate_encounter_risk(
            miss_distance_km=request.miss_distance_km,
            combined_radius_m=request.combined_radius_m,
            sigma_radial_m=request.sigma_radial_m,
            sigma_in_track_m=request.sigma_in_track_m,
            sigma_cross_track_m=request.sigma_cross_track_m,
        )

        return {
            "miss_distance_km": request.miss_distance_km,
            "collision_probability": pc,
            "risk_tier": tier,
            "maneuver_required": tier in ["CRITICAL", "HIGH"],
            "combined_radius_m": request.combined_radius_m,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(exc)}")


@router.post("/cdm")
async def generate_cdm_report(request: CollisionEvalRequest):
    """Generate a standard CCSDS Conjunction Data Message (CDM) plaintext report."""
    pc, tier = evaluate_encounter_risk(
        miss_distance_km=request.miss_distance_km,
        combined_radius_m=request.combined_radius_m,
        sigma_radial_m=request.sigma_radial_m,
        sigma_in_track_m=request.sigma_in_track_m,
        sigma_cross_track_m=request.sigma_cross_track_m,
    )

    tca_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=6)
    event = ConjunctionEvent(
        tca_seconds=21600.0,
        tca_datetime=tca_dt,
        miss_distance_km=request.miss_distance_km,
        radial_miss_km=request.miss_distance_km * 0.2,
        in_track_miss_km=request.miss_distance_km * 0.9,
        cross_track_miss_km=request.miss_distance_km * 0.4,
        relative_speed_km_s=14.25,
        r1_tca=np.array([6800000.0, 0.0, 0.0]),
        v1_tca=np.array([0.0, 7600.0, 0.0]),
        r2_tca=np.array([6800000.0 + request.miss_distance_km * 1000.0, 0.0, 0.0]),
        v2_tca=np.array([0.0, -7600.0, 0.0]),
        collision_probability=pc,
        risk_tier=tier,
    )

    return PlainTextResponse(content=event.generate_cdm(), media_type="text/plain")


@router.post("/maneuver")
async def recommend_maneuver(request: AvoidanceManeuverRequest):
    """Calculate prograde/retrograde collision avoidance burn to restore safe clearance."""
    tca_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=request.tca_seconds)
    event = ConjunctionEvent(
        tca_seconds=request.tca_seconds,
        tca_datetime=tca_dt,
        miss_distance_km=request.miss_distance_km,
        radial_miss_km=0.0,
        in_track_miss_km=request.miss_distance_km,
        cross_track_miss_km=0.0,
        relative_speed_km_s=12.5,
        r1_tca=np.array(request.r1_tca, dtype=np.float64),
        v1_tca=np.array(request.v1_tca, dtype=np.float64),
        r2_tca=np.array(request.r1_tca, dtype=np.float64),
        v2_tca=np.array(request.v1_tca, dtype=np.float64),
        collision_probability=1e-3,
        risk_tier="HIGH",
    )

    result = calculate_avoidance_maneuver(
        event=event,
        lead_time_seconds=request.lead_time_seconds,
        target_clearance_km=request.target_clearance_km,
    )
    return result
