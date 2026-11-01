"""Conjunction events and screening query endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orbitguard.api.schemas import ConjunctionResponse, ScreeningTriggerRequest
from orbitguard.db.models import ConjunctionScreening
from orbitguard.db.session import get_db
from orbitguard.workers.background_tasks import OrbitGuardAutonomousWorker

router = APIRouter(prefix="/conjunctions", tags=["Conjunctions"])
worker = OrbitGuardAutonomousWorker()


@router.get("", response_model=List[ConjunctionResponse])
async def list_conjunctions(
    tenant_id: str = "00000000-0000-0000-0000-000000000001",
    risk_tier: Optional[str] = Query(None, description="Filter by risk tier (CRITICAL, HIGH, MODERATE, LOW)"),
    db: AsyncSession = Depends(get_db),
):
    """List detected conjunction events within the current screening window."""
    query = select(ConjunctionScreening).where(ConjunctionScreening.tenant_id == tenant_id)
    if risk_tier:
        query = query.where(ConjunctionScreening.risk_tier == risk_tier.upper())
    query = query.order_by(ConjunctionScreening.collision_probability.desc())
    res = await db.execute(query)
    return res.scalars().all()


@router.post("/screen", response_model=List[dict])
async def trigger_screening_sweep(
    req: ScreeningTriggerRequest,
    db: AsyncSession = Depends(get_db),
):
    """Trigger an on-demand physics-informed 72-hour conjunction screening cycle."""
    detected = await worker.run_screening_cycle(
        session=db,
        tenant_id=req.tenant_id,
        duration_hours=req.horizon_hours,
    )
    return detected
