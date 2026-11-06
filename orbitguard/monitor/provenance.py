"""Quantity records with time, source, unit, frame, and quality status."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def as_utc(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def iso(moment: Optional[datetime]) -> Optional[str]:
    moment = as_utc(moment)
    if moment is None:
        return None
    return moment.isoformat().replace("+00:00", "Z")


def age_seconds(event: Optional[datetime], now: Optional[datetime]) -> Optional[float]:
    """Seconds from ``event`` to ``now``. Positive when ``event`` is earlier."""
    event = as_utc(event)
    now = as_utc(now)
    if event is None or now is None:
        return None
    return (now - event).total_seconds()


def freshness_label(age_s: Optional[float], threshold_s: Optional[float]) -> str:
    """Classify an age against a product display setting.

    The labels are not safety states. A missing age stays unknown. An age
    below zero means the timestamp is ahead of the reference clock.
    """
    if age_s is None or threshold_s is None:
        return "unknown"
    if age_s < 0.0:
        return "ahead_of_reference_clock"
    if age_s > threshold_s:
        return "older_than_display_setting"
    return "within_display_setting"


def quantity(
    *,
    name: str,
    value: Any,
    unit: Optional[str],
    status: str,
    reference_frame: Optional[str],
    time_system: Optional[str],
    observation_time: Optional[datetime],
    retrieval_time: Optional[datetime],
    now: datetime,
    source_name: str,
    source_url: Optional[str],
    epoch_stale_s: float,
    retrieval_stale_s: float,
    note: str,
) -> dict[str, Any]:
    if status not in {"measured", "modelled", "derived", "unavailable"}:
        raise ValueError(f"Unsupported quantity status: {status}")
    if status == "unavailable" and value is not None:
        raise ValueError(f"{name} is unavailable and must not carry a filled value")
    epoch_age = age_seconds(observation_time, now)
    retrieval_age = age_seconds(retrieval_time, now)
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "status": status,
        "reference_frame": reference_frame,
        "time_system": time_system,
        "observation_time": iso(observation_time),
        "retrieval_time": iso(retrieval_time),
        "epoch_age_s": epoch_age,
        "retrieval_age_s": retrieval_age,
        "epoch_freshness": freshness_label(epoch_age, epoch_stale_s),
        "retrieval_freshness": freshness_label(retrieval_age, retrieval_stale_s),
        "source_name": source_name,
        "source_url": source_url,
        "note": note,
    }
