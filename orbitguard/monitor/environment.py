"""Space-weather indices as sourced quantities, not density or plasma.

NOAA planetary K is a dimensionless index. F10.7 is a solar radio flux in
solar flux units. Neither is a local thermospheric density or a local
plasma measurement. TEC is an integrated electron column and is not
available from these two files.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from orbitguard.monitor.constants import (
    DEFAULT_ENVIRONMENT_CACHE_S,
    DEFAULT_EPOCH_STALE_S,
    DEFAULT_RETRIEVAL_STALE_S,
    F107_UNIT,
    KP_UNIT,
    NOAA_F107_URL,
    NOAA_JSON_INDEX,
    NOAA_KP_URL,
    TEC_UNIT,
)
from orbitguard.monitor.ingest import HttpResult, Transport, default_transport
from orbitguard.monitor.provenance import as_utc, quantity

_F107_NOTE = (
    "Solar radio flux at 2800 MHz (10.7 cm) from NOAA SWPC f107_cm_flux.json. "
    "Unit is solar flux units. This is not a thermospheric density and not a "
    "plasma density."
)
_KP_NOTE = (
    "Estimated planetary K from NOAA SWPC planetary_k_index_1m.json. "
    "Kp is dimensionless. On 2026-11-06 the legacy path "
    "planetary_k_index.json returned HTTP 404, so OrbitGuard does not use it. "
    "This index is not a local magnetometer reading, not a density, and not TEC."
)


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def parse_f107(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("F10.7 payload is empty")
    chosen = None
    chosen_time = None
    for item in payload:
        if not isinstance(item, dict) or item.get("flux") is None or not item.get("time_tag"):
            continue
        stamp = _parse_time(str(item["time_tag"]))
        if chosen_time is None or stamp >= chosen_time:
            chosen = item
            chosen_time = stamp
    if chosen is None or chosen_time is None:
        raise ValueError("F10.7 payload has no flux")
    return {
        "flux": float(chosen["flux"]),
        "frequency_mhz": None if chosen.get("frequency") is None else float(chosen["frequency"]),
        "reporting_schedule": chosen.get("reporting_schedule"),
        "observation_time": chosen_time,
    }


def parse_kp(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Kp payload is empty")
    chosen = None
    chosen_time = None
    for item in payload:
        if not isinstance(item, dict) or not item.get("time_tag"):
            continue
        if item.get("estimated_kp") is None and item.get("kp_index") is None:
            continue
        stamp = _parse_time(str(item["time_tag"]))
        if chosen_time is None or stamp >= chosen_time:
            chosen = item
            chosen_time = stamp
    if chosen is None or chosen_time is None:
        raise ValueError("Kp payload has no index")
    return {
        "estimated_kp": None if chosen.get("estimated_kp") is None else float(chosen["estimated_kp"]),
        "kp_index": None if chosen.get("kp_index") is None else float(chosen["kp_index"]),
        "kp_symbol": None if chosen.get("kp") is None else str(chosen["kp"]),
        "observation_time": chosen_time,
    }


def _unavailable(name: str, unit: Optional[str], note: str, now: datetime) -> dict[str, Any]:
    return quantity(
        name=name,
        value=None,
        unit=unit,
        status="unavailable",
        reference_frame=None,
        time_system="UTC",
        observation_time=None,
        retrieval_time=None,
        now=now,
        source_name="none",
        source_url=None,
        epoch_stale_s=DEFAULT_EPOCH_STALE_S,
        retrieval_stale_s=DEFAULT_RETRIEVAL_STALE_S,
        note=note,
    )


def build_environment(
    *,
    now: datetime,
    retrieval_time: Optional[datetime],
    f107: Optional[dict[str, Any]],
    kp: Optional[dict[str, Any]],
    f107_error: Optional[str],
    kp_error: Optional[str],
    epoch_stale_s: float = DEFAULT_EPOCH_STALE_S,
    retrieval_stale_s: float = DEFAULT_RETRIEVAL_STALE_S,
) -> dict[str, Any]:
    """Assemble environment quantities. Missing feeds stay null."""
    now = as_utc(now) or now
    if f107 is None:
        f107_row = _unavailable(
            "f107_solar_flux",
            F107_UNIT,
            f107_error or "NOAA F10.7 file was not retrieved. No climatological substitute is inserted.",
            now,
        )
    else:
        f107_row = quantity(
            name="f107_solar_flux",
            value=f107["flux"],
            unit=F107_UNIT,
            status="measured",
            reference_frame=None,
            time_system="UTC",
            observation_time=f107["observation_time"],
            retrieval_time=retrieval_time,
            now=now,
            source_name="noaa_swpc_f107_cm_flux",
            source_url=NOAA_F107_URL,
            epoch_stale_s=epoch_stale_s,
            retrieval_stale_s=retrieval_stale_s,
            note=_F107_NOTE
            + (
                f" frequency_mhz={f107['frequency_mhz']}; reporting_schedule={f107['reporting_schedule']}."
            ),
        )
    if kp is None:
        kp_row = _unavailable(
            "planetary_k",
            KP_UNIT,
            kp_error or "NOAA planetary K file was not retrieved. No default Kp is inserted.",
            now,
        )
    else:
        kp_row = quantity(
            name="planetary_k",
            value=kp["estimated_kp"] if kp["estimated_kp"] is not None else kp["kp_index"],
            unit=KP_UNIT,
            status="derived",
            reference_frame=None,
            time_system="UTC",
            observation_time=kp["observation_time"],
            retrieval_time=retrieval_time,
            now=now,
            source_name="noaa_swpc_planetary_k_index_1m",
            source_url=NOAA_KP_URL,
            epoch_stale_s=epoch_stale_s,
            retrieval_stale_s=retrieval_stale_s,
            note=_KP_NOTE
            + f" estimated_kp={kp['estimated_kp']}; kp_index={kp['kp_index']}; symbol={kp['kp_symbol']}.",
        )
    tec = _unavailable(
        "total_electron_content",
        TEC_UNIT,
        "No GNSS TEC product is connected. TEC would be an integrated electron column "
        "(TECU = 10^16 electrons m^-2), not a local 3D plasma density. It is not inferred from Kp or F10.7.",
        now,
    )
    density = _unavailable(
        "thermospheric_mass_density",
        "kg m^-3",
        "NRLMSISE-00 and JB2008 are not run in this version. F10.7 and Kp are not converted into a density. "
        "Drag-density uncertainty is kept separate from the ionospheric quantities above.",
        now,
    )
    return {
        "mode_label": "Environment feed — values appear only when the cited file was retrieved",
        "index": NOAA_JSON_INDEX,
        "cache_interval_s": DEFAULT_ENVIRONMENT_CACHE_S,
        "cache_note": "The cache interval is a product setting to avoid repeated downloads. It is not a validity period for the indices.",
        "f107": f107_row,
        "planetary_k": kp_row,
        "tec": tec,
        "thermospheric_density": density,
        "distinctions": [
            "F10.7 is a solar radio flux observation, not thermospheric density.",
            "Planetary K is a dimensionless geomagnetic index, not a local plasma sensor.",
            "TEC is an integrated column. It is unavailable here and would not be a local density even if a TEC file were connected.",
            "A future NRLMSISE-00 or JB2008 density would be modelled, and its uncertainty is not the same thing as TEC.",
        ],
    }


def fetch_environment(
    *,
    now: datetime,
    transport: Transport = default_transport,
    cache: Optional[dict[str, Any]] = None,
    cache_s: float = DEFAULT_ENVIRONMENT_CACHE_S,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (public payload, cache document). Uses cache inside ``cache_s``."""
    now = as_utc(now) or now
    if cache and cache.get("retrieved_at"):
        retrieved = _parse_time(cache["retrieved_at"])
        if (now - retrieved).total_seconds() < cache_s and cache.get("payload"):
            refreshed = build_environment(
                now=now,
                retrieval_time=retrieved,
                f107=_restore_parsed(cache["payload"].get("f107_parsed")),
                kp=_restore_parsed(cache["payload"].get("kp_parsed")),
                f107_error=cache["payload"].get("f107_error"),
                kp_error=cache["payload"].get("kp_error"),
            )
            return refreshed, cache

    f107_parsed = None
    kp_parsed = None
    f107_error = None
    kp_error = None
    for url, label in ((NOAA_F107_URL, "f107"), (NOAA_KP_URL, "kp")):
        result: HttpResult = transport(url)
        if result.status != 200:
            message = f"HTTP {result.status} from {url}. No substitute value was inserted."
            if label == "f107":
                f107_error = message
            else:
                kp_error = message
            continue
        try:
            decoded = json.loads(result.body.decode("utf-8"))
            if label == "f107":
                f107_parsed = parse_f107(decoded)
            else:
                kp_parsed = parse_kp(decoded)
        except (ValueError, json.JSONDecodeError) as exc:
            message = f"{url} could not be parsed ({exc}). No substitute value was inserted."
            if label == "f107":
                f107_error = message
            else:
                kp_error = message
    public = build_environment(
        now=now,
        retrieval_time=now,
        f107=f107_parsed,
        kp=kp_parsed,
        f107_error=f107_error,
        kp_error=kp_error,
    )
    document = {
        "retrieved_at": now.isoformat().replace("+00:00", "Z"),
        "payload": {
            "f107_parsed": _dump_parsed(f107_parsed),
            "kp_parsed": _dump_parsed(kp_parsed),
            "f107_error": f107_error,
            "kp_error": kp_error,
        },
    }
    return public, document


def _dump_parsed(parsed: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if parsed is None:
        return None
    dumped = dict(parsed)
    moment = dumped.get("observation_time")
    if isinstance(moment, datetime):
        dumped["observation_time"] = moment.isoformat().replace("+00:00", "Z")
    return dumped


def _restore_parsed(parsed: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not parsed:
        return None
    restored = dict(parsed)
    if isinstance(restored.get("observation_time"), str):
        restored["observation_time"] = _parse_time(restored["observation_time"])
    return restored
