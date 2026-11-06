"""Passive, idempotent GP ingest.

One CelesTrak group request is cached. HTTP 301, 403, and 404 are not
retried. Space-Track credentials are never returned and this preview
does not log in.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from orbitguard.monitor.constants import (
    ALLOWED_PUBLIC_GROUP,
    CELESTRAK_MIN_REFETCH_S,
    CELESTRAK_STATIONS_URL,
    SPACETRACK_DOC,
    USER_AGENT,
)
from orbitguard.monitor.gp_omm import RecordRejected, validate_omm
from orbitguard.monitor.provenance import iso

Transport = Callable[[str], "HttpResult"]


@dataclass
class HttpResult:
    status: int
    body: bytes
    url: str


def default_transport(url: str, timeout_s: float = 20.0) -> HttpResult:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return HttpResult(status=int(response.status), body=response.read(), url=response.geturl())
    except urllib.error.HTTPError as exc:
        payload = exc.read() if exc.fp is not None else b""
        return HttpResult(status=int(exc.code), body=payload, url=url)


class CatalogStore:
    """Content-addressed store of validated OMM records."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"records": {}, "latest_public": None, "fetches": []}
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.setdefault("records", {})
        data.setdefault("fetches", [])
        data.setdefault("latest_public", None)
        return data

    def save(self, data: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        temporary.replace(self.path)


def ingest_validated_batch(
    store: CatalogStore,
    raw_records: list[Mapping[str, Any]],
    *,
    retrieved_at: datetime,
    source_name: str,
    source_url: Optional[str],
) -> dict[str, Any]:
    """Insert records once. Same hash is a duplicate. Same id+epoch with a new hash conflicts."""
    data = store.load()
    inserted = []
    duplicates = []
    conflicts = []
    rejected = []
    index = {
        (item["dedup_key"]["norad_cat_id"], item["dedup_key"]["epoch"]): content_hash
        for content_hash, item in data["records"].items()
    }
    for raw in raw_records:
        try:
            validated = validate_omm(raw)
        except RecordRejected as exc:
            rejected.append({"reason": str(exc)})
            continue
        content_hash = validated["content_hash"]
        key = (validated["dedup_key"]["norad_cat_id"], validated["dedup_key"]["epoch"])
        if content_hash in data["records"]:
            duplicates.append(content_hash)
            continue
        previous = index.get(key)
        if previous is not None and previous != content_hash:
            conflicts.append(
                {
                    "norad_cat_id": key[0],
                    "epoch": key[1],
                    "kept_hash": previous,
                    "rejected_hash": content_hash,
                }
            )
            continue
        data["records"][content_hash] = {
            **validated,
            "source_name": source_name,
            "source_url": source_url,
            "first_retrieved_at": iso(retrieved_at),
        }
        index[key] = content_hash
        inserted.append(content_hash)
    data["latest_batch"] = {
        "retrieved_at": iso(retrieved_at),
        "source_name": source_name,
        "source_url": source_url,
        "inserted": inserted,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "rejected_count": len(rejected),
    }
    store.save(data)
    return {
        "inserted_count": len(inserted),
        "duplicate_count": len(duplicates),
        "conflict_count": len(conflicts),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "conflicts": conflicts,
        "inserted_hashes": inserted,
        "credentials_exposed": False,
    }


def _excerpt(body: bytes, limit: int = 280) -> str:
    text = body.decode("utf-8", errors="replace").strip().replace("\n", " ")
    return text[:limit]


def fetch_public_stations(
    store: CatalogStore,
    *,
    now: datetime,
    transport: Transport = default_transport,
    sleeper: Callable[[float], None] = time.sleep,
    min_interval_s: float = CELESTRAK_MIN_REFETCH_S,
) -> dict[str, Any]:
    """Download the stations group at most once per ``min_interval_s``.

    The URL is a single GROUP query. Catalog numbers are not requested
    one by one. A 301, 403, or 404 stops the client.
    """
    if ALLOWED_PUBLIC_GROUP != "stations":
        raise RuntimeError("This preview is pinned to the stations group")
    data = store.load()
    last_success = _parse_time(_last_tagged(data, "success"))
    last_block = _parse_time(_last_tagged(data, "blocked"))
    if last_success is not None and (now - last_success).total_seconds() < min_interval_s:
        return {
            "network_request": False,
            "reason": "cached_within_refetch_interval",
            "min_interval_s": min_interval_s,
            "retrieved_at": iso(last_success),
            "credentials_exposed": False,
        }
    if last_block is not None and (now - last_block).total_seconds() < min_interval_s:
        return {
            "network_request": False,
            "reason": "blocked_response_not_retried",
            "min_interval_s": min_interval_s,
            "credentials_exposed": False,
        }

    url = CELESTRAK_STATIONS_URL
    result = transport(url)
    calls = 1
    if result.status in {429, 500, 502, 503}:
        sleeper(2.0)
        result = transport(url)
        calls += 1
    entry = {
        "time": iso(now),
        "url": url,
        "status": result.status,
        "calls": calls,
        "tag": "success" if result.status == 200 else ("blocked" if result.status in {301, 403, 404} else "error"),
        "excerpt": _excerpt(result.body) if result.status != 200 else "",
    }
    data = store.load()
    data["fetches"].append(entry)
    store.save(data)
    if result.status != 200:
        return {
            "network_request": True,
            "http_status": result.status,
            "calls": calls,
            "reason": "stop_no_further_retry" if result.status in {301, 403, 404} else "gave_up_after_one_backoff",
            "excerpt": entry["excerpt"],
            "credentials_exposed": False,
            "url": url,
        }
    try:
        payload = json.loads(result.body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "network_request": True,
            "http_status": 200,
            "reason": f"response was not JSON: {exc}",
            "credentials_exposed": False,
            "url": url,
        }
    if not isinstance(payload, list):
        return {
            "network_request": True,
            "reason": "GP JSON response was not a list",
            "credentials_exposed": False,
            "url": url,
        }
    report = ingest_validated_batch(
        store,
        payload,
        retrieved_at=now,
        source_name="celestrak_gp_json",
        source_url=url,
    )
    data = store.load()
    data["latest_public"] = {
        "group": ALLOWED_PUBLIC_GROUP,
        "retrieved_at": iso(now),
        "source_name": "celestrak_gp_json",
        "source_url": url,
        "hashes": report["inserted_hashes"] + [
            content_hash
            for content_hash, item in data["records"].items()
            if item.get("source_url") == url
            and (item["dedup_key"]["norad_cat_id"], item["dedup_key"]["epoch"])
        ],
    }
    # Keep the hash list to records from this payload only.
    wanted = []
    seen = set()
    for raw in payload:
        try:
            validated = validate_omm(raw)
        except RecordRejected:
            continue
        if validated["content_hash"] not in seen and validated["content_hash"] in data["records"]:
            seen.add(validated["content_hash"])
            wanted.append(validated["content_hash"])
    data["latest_public"]["hashes"] = wanted
    data["fetches"][-1]["tag"] = "success"
    store.save(data)
    report.update(
        {
            "network_request": True,
            "http_status": 200,
            "calls": calls,
            "url": url,
            "group": ALLOWED_PUBLIC_GROUP,
            "retrieved_at": iso(now),
            "credentials_exposed": False,
        }
    )
    return report


def space_track_status() -> dict[str, Any]:
    """Report whether server-side credentials exist, without revealing them."""
    identity = os.environ.get("SPACETRACK_USER", "")
    password = os.environ.get("SPACETRACK_PASSWORD", "")
    return {
        "provider": "space-track",
        "documentation": SPACETRACK_DOC,
        "configured": bool(identity and password),
        "fetch_performed": False,
        "credentials_exposed": False,
        "reason": (
            "Space-Track requires an account. This preview does not log in, "
            "does not download the GP class, and does not return credentials. "
            "Use the public CelesTrak GP JSON batch instead."
        ),
    }


def _parse_time(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _last_tagged(data: dict[str, Any], tag: str) -> Optional[str]:
    for entry in reversed(data.get("fetches", [])):
        if entry.get("tag") == tag:
            return entry.get("time")
    if tag == "success" and data.get("latest_public"):
        return data["latest_public"].get("retrieved_at")
    return None
