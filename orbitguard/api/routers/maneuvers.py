"""Collision avoidance maneuver recommendations and flight director briefings."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orbitguard.api.schemas import ManeuverResponse
from orbitguard.db.models import ManeuverRecommendation
from orbitguard.db.session import get_db

router = APIRouter(prefix="/maneuvers", tags=["Maneuvers"])


@router.get("", response_model=List[ManeuverResponse])
async def list_recommended_maneuvers(
    satellite_id: str = None,
    db: AsyncSession = Depends(get_db),
):
    """List calculated collision avoidance maneuvers."""
    query = select(ManeuverRecommendation)
    if satellite_id:
        query = query.where(ManeuverRecommendation.satellite_id == satellite_id)
    query = query.order_by(ManeuverRecommendation.planned_burn_epoch.asc())
    res = await db.execute(query)
    return res.scalars().all()


@router.get("/{maneuver_id}/briefing")
async def get_maneuver_briefing(
    maneuver_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve raw Markdown mission control executive brief for a maneuver."""
    res = await db.execute(select(ManeuverRecommendation).where(ManeuverRecommendation.id == maneuver_id))
    maneuver = res.scalar_one_or_none()
    if not maneuver:
        raise HTTPException(status_code=404, detail="Maneuver recommendation not found")
    return {
        "maneuver_id": maneuver.id,
        "satellite_id": maneuver.satellite_id,
        "strategy": maneuver.maneuver_strategy,
        "executive_brief_markdown": maneuver.executive_brief_markdown,
    }


@router.post("/{maneuver_id}/approve")
async def approve_maneuver(
    maneuver_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Flight Director 1-click authorization to uplink and arm thruster CAM sequence."""
    res = await db.execute(select(ManeuverRecommendation).where(ManeuverRecommendation.id == maneuver_id))
    maneuver = res.scalar_one_or_none()
    if not maneuver:
        raise HTTPException(status_code=404, detail="Maneuver recommendation not found")
    maneuver.recommendation_status = "operator_approved"
    await db.commit()
    return {
        "status": "approved",
        "maneuver_id": maneuver.id,
        "uplink_window": maneuver.planned_burn_epoch.isoformat(),
        "command_block": f"CAM_ARM:STRAT={maneuver.maneuver_strategy}:DV={maneuver.dv_total_magnitude_m_s:.4f}:BURN_SEC={maneuver.burn_duration_s:.1f}",
    }
