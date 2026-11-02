"""System telemetry and health ping endpoint."""

from __future__ import annotations

import datetime
import os
import platform
import time

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])

START_TIME = time.time()


class SystemTelemetry(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    timestamp: str
    uptime_seconds: float
    platform: str
    python_version: str
    features: dict


@router.get("/health", response_model=SystemTelemetry)
@router.get("/api/health", response_model=SystemTelemetry)
async def get_health():
    """Retrieve service health status, active astrodynamics features, and system telemetry."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    uptime = time.time() - START_TIME

    return SystemTelemetry(
        status="healthy",
        service="nanodynamics-platform",
        version="2.4.0",
        environment=os.getenv("ENVIRONMENT", "production"),
        timestamp=now_utc,
        uptime_seconds=round(uptime, 2),
        platform=platform.platform(),
        python_version=platform.python_version(),
        features={
            "sgp4_tle": True,
            "cowell_rk45_perturbation": True,
            "j2_oblateness": True,
            "atmospheric_drag": True,
            "orbitguard_conjunction_screening": True,
            "b_plane_risk_scoring": True,
            "ccsds_cdm_generation": True,
        },
    )
