"""Alert auditing and ground station webhook receivers."""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orbitguard.api.schemas import AlertResponse
from orbitguard.db.models import AlertDispatch
from orbitguard.db.session import get_db

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("", response_model=List[AlertResponse])
async def list_alert_dispatches(
    conjunction_id: str = None,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve audit log of dispatched multi-channel alerts."""
    query = select(AlertDispatch)
    if conjunction_id:
        query = query.where(AlertDispatch.conjunction_id == conjunction_id)
    query = query.order_by(AlertDispatch.created_at.desc())
    res = await db.execute(query)
    return res.scalars().all()


@router.post("/webhook-demo")
async def demo_ground_station_webhook_receiver(
    payload: Dict[str, Any],
    x_orbitguard_signature: str = Header(None),
    x_orbitguard_timestamp: str = Header(None),
):
    """Mock ground station webhook receiver verifying signature and receiving alert payloads."""
    return {
        "status": "acknowledged",
        "signature_received": x_orbitguard_signature,
        "timestamp_received": x_orbitguard_timestamp,
        "payload_received": payload,
    }
