"""Satellite registry and asset management endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orbitguard.api.schemas import SatelliteCreate, SatelliteResponse
from orbitguard.db.models import Satellite
from orbitguard.db.session import get_db

router = APIRouter(prefix="/satellites", tags=["Satellites"])


@router.get("", response_model=List[SatelliteResponse])
async def list_satellites(
    tenant_id: str = "00000000-0000-0000-0000-000000000001",
    db: AsyncSession = Depends(get_db),
):
    """Retrieve all operational satellites for a tenant."""
    res = await db.execute(select(Satellite).where(Satellite.tenant_id == tenant_id))
    return res.scalars().all()


@router.post("", response_model=SatelliteResponse)
async def register_satellite(
    sat_in: SatelliteCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new nanosatellite/CubeSat under mission control."""
    sat = Satellite(**sat_in.model_dump())
    db.add(sat)
    await db.commit()
    await db.refresh(sat)
    return sat
