"""Assemble the monitoring workspace from ingest, SGP4, and the numerical engine."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from orbitguard.monitor.constants import (
    ALLOWED_PUBLIC_GROUP,
    CELESTRAK_GP_DOC,
    CELESTRAK_STATIONS_URL,
    DEFAULT_EPOCH_STALE_S,
    DEFAULT_GEOMETRIC_SCREEN_KM,
    DEFAULT_RETRIEVAL_STALE_S,
    MAX_SCREEN_MINUTES,
    MIN_SCREEN_STEP_S,
)
from orbitguard.monitor.environment import fetch_environment
from orbitguard.monitor.gp_omm import parse_epoch, validate_omm
from orbitguard.monitor.ingest import (
    CatalogStore,
    Transport,
    default_transport,
    fetch_public_stations,
    ingest_validated_batch,
    space_track_status,
)
from orbitguard.monitor.propagation import (
    MeanElementsNotOsculating,
    propagate_osculating_cartesian,
    propagate_sgp4,
    satrec_from_omm,
)
from orbitguard.monitor.provenance import as_utc, iso, quantity
from orbitguard.monitor.risk import assess_covariance, geometric_screen
from orbitguard.monitor.roadmap import roadmap
from orbitguard.monitor.samples import SAMPLE_SOURCE, sample_records
from orbitguard.monitor.sensors import sensor_catalog

MODES = ("awaiting", "simulation_sample", "public_gp")


class MonitorService:
    def __init__(self, cache_dir: Path, transport: Transport = default_transport):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.store = CatalogStore(self.cache_dir / "catalog.json")
        self.environment_path = self.cache_dir / "environment.json"
        self.transport = transport

    def status(self, now: Optional[datetime] = None) -> dict[str, Any]:
        now = as_utc(now) or datetime.now(timezone.utc)
        data = self.store.load()
        latest = data.get("latest_public") or {}
        return {
            "service": "orbitguard-monitor",
            "version": "0.1.0",
            "now": iso(now),
            "modes": list(MODES),
            "public_group": ALLOWED_PUBLIC_GROUP,
            "public_retrieved_at": latest.get("retrieved_at"),
            "public_object_count": len(latest.get("hashes") or []),
            "space_track": space_track_status(),
            "collision_probability": "not_computable",
            "expected_collision_count": "not_implemented",
            "credentials_exposed": False,
            "commands_spacecraft": False,
        }

    def workspace(
        self,
        mode: str,
        now: Optional[datetime] = None,
        *,
        epoch_stale_s: float = DEFAULT_EPOCH_STALE_S,
        retrieval_stale_s: float = DEFAULT_RETRIEVAL_STALE_S,
    ) -> dict[str, Any]:
        now = as_utc(now) or datetime.now(timezone.utc)
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}")
        settings = {
            "epoch_stale_s": epoch_stale_s,
            "retrieval_stale_s": retrieval_stale_s,
            "note": (
                "These ages are compared with display settings for this workspace. "
                "Crossing a setting does not declare an orbit safe or unsafe."
            ),
        }
        if mode == "awaiting":
            return self._empty(now, "awaiting", "No catalog is loaded.", settings)
        if mode == "simulation_sample":
            objects = []
            for raw in sample_records():
                public_id = raw["public_id"]
                validated = validate_omm(raw)
                objects.append(
                    self._publish_object(
                        validated,
                        public_id=public_id,
                        now=now,
                        retrieval_time=None,
                        source_name=SAMPLE_SOURCE,
                        source_url=None,
                        mode="simulation_sample",
                        display_time=parse_epoch(validated["record"]["EPOCH"]),
                        synthetic=True,
                        epoch_stale_s=epoch_stale_s,
                        retrieval_stale_s=retrieval_stale_s,
                    )
                )
            return {
                "mode": "simulation_sample",
                "mode_label": "Simulation sample — synthetic mean elements, not a live catalog and not current time",
                "now": iso(now),
                "display_settings": settings,
                "objects": objects,
                "credentials_exposed": False,
            }
        data = self.store.load()
        latest = data.get("latest_public")
        if not latest or not latest.get("hashes"):
            return self._empty(
                now,
                "awaiting",
                "Public GP has not been retrieved. Sample mode is a separate, labelled action.",
                settings,
            )
        retrieved = _parse(latest["retrieved_at"])
        objects = []
        for content_hash in latest["hashes"]:
            stored = data["records"].get(content_hash)
            if stored is None:
                continue
            record = stored["record"]
            objects.append(
                self._publish_object(
                    stored,
                    public_id=str(record["NORAD_CAT_ID"]),
                    now=now,
                    retrieval_time=retrieved,
                    source_name=stored.get("source_name", "celestrak_gp_json"),
                    source_url=stored.get("source_url", CELESTRAK_STATIONS_URL),
                    mode="public_gp",
                    display_time=now,
                    synthetic=False,
                    epoch_stale_s=epoch_stale_s,
                    retrieval_stale_s=retrieval_stale_s,
                )
            )
        return {
            "mode": "public_gp",
            "mode_label": "Public GP — CelesTrak OMM JSON mean elements propagated with SGP4. Not a sensor feed.",
            "now": iso(now),
            "retrieved_at": latest.get("retrieved_at"),
            "source_url": latest.get("source_url", CELESTRAK_STATIONS_URL),
            "documentation": CELESTRAK_GP_DOC,
            "display_settings": settings,
            "objects": objects,
            "credentials_exposed": False,
        }

    def refresh_public(self, now: Optional[datetime] = None, *, min_interval_s: float | None = None) -> dict[str, Any]:
        now = as_utc(now) or datetime.now(timezone.utc)
        kwargs: dict[str, Any] = {"now": now, "transport": self.transport}
        if min_interval_s is not None:
            kwargs["min_interval_s"] = min_interval_s
        report = fetch_public_stations(self.store, **kwargs)
        report["credentials_exposed"] = False
        report["group"] = ALLOWED_PUBLIC_GROUP
        return report

    def ingest_batch(self, records: list[dict[str, Any]], now: Optional[datetime] = None, *, source_name: str, source_url: Optional[str]) -> dict[str, Any]:
        now = as_utc(now) or datetime.now(timezone.utc)
        if len(records) > 100:
            raise ValueError("This preview accepts at most 100 records in one batch")
        report = ingest_validated_batch(
            self.store,
            records,
            retrieved_at=now,
            source_name=source_name,
            source_url=source_url,
        )
        if report["inserted_hashes"] or report["duplicate_count"]:
            data = self.store.load()
            # A manually ingested batch becomes the public workspace only when
            # the caller used the CelesTrak refresh. Manual ingest is stored
            # for dedup tests and is not silently treated as the live group.
            data["manual_batches"] = data.get("manual_batches", [])
            data["manual_batches"].append(
                {"retrieved_at": iso(now), "source_name": source_name, "hashes": report["inserted_hashes"]}
            )
            self.store.save(data)
        return report

    def environment(self, now: Optional[datetime] = None) -> dict[str, Any]:
        now = as_utc(now) or datetime.now(timezone.utc)
        cache = None
        if self.environment_path.exists():
            with self.environment_path.open("r", encoding="utf-8") as handle:
                cache = json.load(handle)
        public, document = fetch_environment(now=now, transport=self.transport, cache=cache)
        if document.get("retrieved_at") != (cache or {}).get("retrieved_at"):
            with self.environment_path.open("w", encoding="utf-8") as handle:
                json.dump(document, handle)
        public["credentials_exposed"] = False
        return public

    def sensors(self) -> dict[str, Any]:
        return sensor_catalog()

    def architecture(self) -> dict[str, Any]:
        return roadmap()

    def simulate(self, body: dict[str, Any]) -> dict[str, Any]:
        result = propagate_osculating_cartesian(
            body["position_m"],
            body["velocity_m_s"],
            float(body.get("duration_s", 600.0)),
            float(body.get("step_s", 60.0)),
            state_kind=body.get("state_kind", ""),
            frame=body.get("frame", ""),
            include_j2=bool(body.get("include_j2", True)),
            include_drag=bool(body.get("include_drag", False)),
        )
        result["credentials_exposed"] = False
        return result

    def screen(self, body: dict[str, Any], now: Optional[datetime] = None) -> dict[str, Any]:
        now = as_utc(now) or datetime.now(timezone.utc)
        mode = body["mode"]
        window = min(float(body.get("window_minutes", 120.0)), MAX_SCREEN_MINUTES)
        step = max(float(body.get("step_s", 60.0)), MIN_SCREEN_STEP_S)
        threshold = float(body.get("threshold_km", DEFAULT_GEOMETRIC_SCREEN_KM))
        if mode == "osculating_simulation":
            geometry = self._screen_osculating(body, window * 60.0, step, threshold)
            state_epoch = body.get("state_epoch")
            state_frame = body.get("frame")
        else:
            workspace = self.workspace(mode, now=now)
            if workspace["mode"] != mode:
                return {
                    "mode": workspace["mode"],
                    "mode_label": workspace["mode_label"],
                    "geometric_miss_km": None,
                    "pc": None,
                    "pc_status": "not_computable",
                    "nc": None,
                    "nc_status": "not_implemented",
                    "safety_determination": "not_made",
                    "reason": workspace["mode_label"],
                    "credentials_exposed": False,
                }
            primary = _find(workspace["objects"], body["primary_id"])
            secondary = _find(workspace["objects"], body["secondary_id"])
            if primary is None or secondary is None:
                raise KeyError("Both objects must be in the selected catalog")
            if primary["reference_frame"] != secondary["reference_frame"]:
                raise MeanElementsNotOsculating("Refusing to difference states that are not in the same frame")
            geometry = geometric_screen(
                _trim_track(primary["track"], window * 60.0),
                _trim_track(secondary["track"], window * 60.0),
                frame=primary["reference_frame"],
                threshold_km=threshold,
            )
            geometry["primary_id"] = primary["id"]
            geometry["secondary_id"] = secondary["id"]
            geometry["mode"] = mode
            geometry["mode_label"] = workspace["mode_label"]
            state_epoch = primary["element_epoch"]
            state_frame = primary["reference_frame"]
        geometry["covariance"] = assess_covariance(
            body.get("covariance"),
            state_epoch=state_epoch,
            covariance_epoch=body.get("covariance_epoch"),
            state_frame=state_frame,
            covariance_frame=body.get("covariance_frame"),
        )
        geometry["window_minutes_applied"] = window
        geometry["step_s"] = step
        geometry["credentials_exposed"] = False
        geometry["pc"] = None
        geometry["nc"] = None
        return geometry

    def _screen_osculating(self, body: dict[str, Any], duration_s: float, step_s: float, threshold_km: float) -> dict[str, Any]:
        tracks = []
        for label in ("primary", "secondary"):
            result = propagate_osculating_cartesian(
                body[f"{label}_position_m"],
                body[f"{label}_velocity_m_s"],
                duration_s,
                step_s,
                state_kind=body.get("state_kind", ""),
                frame=body.get("frame", ""),
                include_j2=bool(body.get("include_j2", True)),
                include_drag=bool(body.get("include_drag", False)),
            )
            track = [
                {
                    "time": f"T+{sample['time_offset_s']:.3f}",
                    "r_km": sample["r_km"],
                    "v_km_s": sample["v_km_s"],
                }
                for sample in result["samples"]
            ]
            tracks.append(track)
        geometry = geometric_screen(tracks[0], tracks[1], frame="nanodynamics_eci", threshold_km=threshold_km)
        geometry["mode"] = "osculating_simulation"
        geometry["mode_label"] = "Simulation — osculating numerical states in nanodynamics_eci, not catalog elements"
        geometry["primary_id"] = body.get("primary_id", "osculating-primary")
        geometry["secondary_id"] = body.get("secondary_id", "osculating-secondary")
        return geometry

    def _empty(self, now: datetime, mode: str, reason: str, settings: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": mode,
            "mode_label": reason,
            "now": iso(now),
            "display_settings": settings,
            "objects": [],
            "credentials_exposed": False,
        }

    def _publish_object(
        self,
        stored: dict[str, Any],
        *,
        public_id: str,
        now: datetime,
        retrieval_time: Optional[datetime],
        source_name: str,
        source_url: Optional[str],
        mode: str,
        display_time: datetime,
        synthetic: bool,
        epoch_stale_s: float,
        retrieval_stale_s: float,
    ) -> dict[str, Any]:
        record = stored["record"]
        epoch = parse_epoch(record["EPOCH"])
        period_s = 86400.0 / float(record["MEAN_MOTION"])
        step_s = max(30.0, period_s / 36.0)
        try:
            satellite = satrec_from_omm(record)
            propagated = propagate_sgp4(satellite, display_time, period_s, step_s)
        except Exception as exc:  # noqa: BLE001 - surface the propagator failure, do not invent a state
            propagated = {"status": "unavailable", "state": None, "track": [], "errors": [{"message": str(exc)}]}
        state = propagated.get("state")
        rows = [
            quantity(
                name="inclination",
                value=record["INCLINATION"],
                unit="deg",
                status="modelled",
                reference_frame="TEME",
                time_system="UTC",
                observation_time=epoch,
                retrieval_time=retrieval_time,
                now=now,
                source_name=source_name,
                source_url=source_url,
                epoch_stale_s=epoch_stale_s,
                retrieval_stale_s=retrieval_stale_s,
                note="SGP4 mean inclination from the element set. Not an osculating element.",
            ),
            quantity(
                name="mean_motion",
                value=record["MEAN_MOTION"],
                unit="rev/day",
                status="modelled",
                reference_frame="TEME",
                time_system="UTC",
                observation_time=epoch,
                retrieval_time=retrieval_time,
                now=now,
                source_name=source_name,
                source_url=source_url,
                epoch_stale_s=epoch_stale_s,
                retrieval_stale_s=retrieval_stale_s,
                note="Mean motion of the GP element set.",
            ),
        ]
        if state is None:
            rows.append(
                quantity(
                    name="geodetic_altitude",
                    value=None,
                    unit="km",
                    status="unavailable",
                    reference_frame="teme_gmst_earth_fixed",
                    time_system="UTC",
                    observation_time=epoch,
                    retrieval_time=retrieval_time,
                    now=now,
                    source_name=source_name,
                    source_url=source_url,
                    epoch_stale_s=epoch_stale_s,
                    retrieval_stale_s=retrieval_stale_s,
                    note="SGP4 did not return a state. No position was substituted.",
                )
            )
        else:
            rows.append(
                quantity(
                    name="geodetic_altitude",
                    value=state["altitude_km"],
                    unit="km",
                    status="derived",
                    reference_frame="teme_gmst_earth_fixed",
                    time_system="UTC",
                    observation_time=display_time,
                    retrieval_time=retrieval_time,
                    now=now,
                    source_name=source_name,
                    source_url=source_url,
                    epoch_stale_s=epoch_stale_s,
                    retrieval_stale_s=retrieval_stale_s,
                    note="WGS-84 altitude of the GMST display position. Element epoch and this state time are different clocks.",
                )
            )
        rows.append(
            quantity(
                name="covariance",
                value=None,
                unit=None,
                status="unavailable",
                reference_frame=None,
                time_system="UTC",
                observation_time=None,
                retrieval_time=retrieval_time,
                now=now,
                source_name=source_name,
                source_url=source_url,
                epoch_stale_s=epoch_stale_s,
                retrieval_stale_s=retrieval_stale_s,
                note="Public GP does not include a calibrated covariance. None was invented.",
            )
        )
        label = (
            "Simulation sample — not a catalog object"
            if synthetic
            else "Public GP object — mean elements, SGP4 state"
        )
        return {
            "id": public_id,
            "name": record["OBJECT_NAME"],
            "mode": mode,
            "mode_label": label,
            "synthetic": synthetic,
            "norad_cat_id": None if synthetic else record["NORAD_CAT_ID"],
            "international_designator": record["OBJECT_ID"],
            "element_epoch": record["EPOCH"],
            "state_time": None if state is None else state["time"],
            "reference_frame": "TEME",
            "mean_element_theory": "SGP4",
            "gravity_constants": "WGS72",
            "content_hash": stored["content_hash"],
            "state": state,
            "track": propagated.get("track") or [],
            "propagation_errors": propagated.get("errors") or [],
            "provenance": rows,
            "covariance_status": "absent",
            "pc": None,
            "pc_status": "not_computable",
            "nc": None,
            "nc_status": "not_implemented",
        }


def _parse(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _trim_track(track: list[dict[str, Any]], window_s: float) -> list[dict[str, Any]]:
    if not track:
        return track
    start = _parse(track[0]["time"])
    kept = []
    for row in track:
        if (_parse(row["time"]) - start).total_seconds() <= window_s + 1e-6:
            kept.append(row)
    return kept


def _find(objects: list[dict[str, Any]], object_id: str) -> Optional[dict[str, Any]]:
    for item in objects:
        if item["id"] == object_id:
            return item
    return None
