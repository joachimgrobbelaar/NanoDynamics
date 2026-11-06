"""HTTP API for the OrbitGuard monitoring workspace."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from orbitguard.monitor.propagation import MeanElementsNotOsculating
from orbitguard.monitor.service import MonitorService

router = APIRouter(prefix="/api/orbitguard", tags=["orbitguard-monitor"])
_SERVICE = MonitorService(cache_dir=Path(__file__).resolve().parent / ".cache")


def get_service() -> MonitorService:
    return _SERVICE


def build_test_router(service: MonitorService) -> APIRouter:
    """Point the process-local router at a test service. Tests are single-threaded."""
    global _SERVICE
    _SERVICE = service
    return router


class IngestBody(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list)
    source_name: str = "operator_batch"
    source_url: Optional[str] = None


class SimulateBody(BaseModel):
    position_m: list[float]
    velocity_m_s: list[float]
    duration_s: float = 600.0
    step_s: float = 60.0
    state_kind: str
    frame: str
    include_j2: bool = True
    include_drag: bool = False


class ScreenBody(BaseModel):
    mode: Literal["simulation_sample", "public_gp", "osculating_simulation"]
    primary_id: Optional[str] = None
    secondary_id: Optional[str] = None
    window_minutes: float = 120.0
    step_s: float = 60.0
    threshold_km: float = 25.0
    covariance: Optional[list[list[float]]] = None
    covariance_epoch: Optional[str] = None
    covariance_frame: Optional[str] = None
    state_kind: Optional[str] = None
    frame: Optional[str] = None
    state_epoch: Optional[str] = None
    include_j2: bool = True
    include_drag: bool = False
    primary_position_m: Optional[list[float]] = None
    primary_velocity_m_s: Optional[list[float]] = None
    secondary_position_m: Optional[list[float]] = None
    secondary_velocity_m_s: Optional[list[float]] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


@router.get("/status")
def status() -> dict[str, Any]:
    return get_service().status(_now())


@router.get("/workspace")
def workspace(
    mode: Literal["awaiting", "simulation_sample", "public_gp"] = "awaiting",
    epoch_stale_s: float = 72 * 3600,
    retrieval_stale_s: float = 6 * 3600,
) -> dict[str, Any]:
    if epoch_stale_s <= 0 or retrieval_stale_s <= 0:
        raise HTTPException(status_code=422, detail="Display thresholds must be positive")
    try:
        return get_service().workspace(
            mode,
            now=_now(),
            epoch_stale_s=epoch_stale_s,
            retrieval_stale_s=retrieval_stale_s,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/catalog/refresh")
def refresh_catalog() -> dict[str, Any]:
    return get_service().refresh_public(_now())


@router.post("/ingest")
def ingest(body: IngestBody) -> dict[str, Any]:
    try:
        return get_service().ingest_batch(
            body.records,
            now=_now(),
            source_name=body.source_name,
            source_url=body.source_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/environment")
def environment() -> dict[str, Any]:
    return get_service().environment(_now())


@router.get("/sensors")
def sensors() -> dict[str, Any]:
    return get_service().sensors()


@router.get("/roadmap")
def roadmap_route() -> dict[str, Any]:
    return get_service().architecture()


@router.get("/operator-notes", response_class=PlainTextResponse)
def operator_notes() -> str:
    path = Path(__file__).resolve().parents[2] / "docs" / "orbitguard.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Operator notes are not in this build")
    return path.read_text(encoding="utf-8")


@router.post("/simulation/propagate")
def simulate(body: SimulateBody) -> dict[str, Any]:
    try:
        return get_service().simulate(body.model_dump())
    except MeanElementsNotOsculating as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/conjunction/screen")
def screen(body: ScreenBody) -> dict[str, Any]:
    try:
        return get_service().screen(body.model_dump(), now=_now())
    except MeanElementsNotOsculating as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
