"""
nanodynamics/modules/space_ops_infrastructure/functions.py

Pillar 5: Space Launch Systems, Ground Networks & Agency Intelligence.

Provides data loaders and analytical functions for:
  - Mission lifecycle intelligence
  - Space agency capabilities
  - Historical launch success/failure statistics
  - SpaceX booster turnaround analysis
  - Rocket Lab Electron kick-stage performance
  - Blue Origin New Shepard sub-orbital stats
  - Launch vehicle payload capacity lookup
  - Starlink ground station gateway handover geometry
  - SatNOGS transmitter frequency search
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from nanodynamics.data_ingestion.loaders import get_dataset
from nanodynamics.core.constants import *  # noqa: F401, F403 – project convention

logger = logging.getLogger("nanodynamics.space_ops_infrastructure")

__all__ = [
    # 1  Mission intelligence
    "load_mission_intelligence",
    "query_mission_lifecycle",
    # 2  Space agencies
    "load_space_agencies",
    "get_agency_capabilities",
    # 3  Historical launch log
    "load_historical_launch_log",
    "calculate_launch_success_rate",
    # 4  SpaceX launches
    "load_spacex_launches",
    "analyze_booster_turnaround_time",
    # 5  Rocket Lab launches
    "load_rocket_lab_launches",
    "get_electron_kickstage_performance",
    # 6  Blue Origin launches
    "load_blue_origin_launches",
    "get_new_shepard_apogee_stats",
    # 7  Launch vehicle specs
    "load_launch_vehicle_specs",
    "compute_payload_capacity",
    # 8  Starlink ground stations
    "load_starlink_ground_stations",
    "calculate_gateway_handover",
    # 9  SatNOGS transmitters
    "load_satnogs_transmitters",
    "find_active_frequencies",
]


# ============================================================================
# Helper utilities
# ============================================================================

def _safe_lower(value: Any) -> str:
    """Normalise a field value to lower-case string for comparison."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _parse_date(value: Any) -> Optional[datetime]:
    """Best-effort ISO / common date string parser."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%d/%m/%Y",
                "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    logger.debug("Could not parse date: %s", s)
    return None


def _to_float(value: Any) -> Optional[float]:
    """Safely cast a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ============================================================================
# 1. Mission Intelligence
# ============================================================================

def load_mission_intelligence(**kwargs) -> Any:
    """Load the space-mission-intelligence dataset.

    Returns
    -------
    Dataset
        A Hugging Face Dataset (or IterableDataset) keyed under
        ``"space_mission_intelligence"`` in the NanoDynamics registry.
    """
    logger.info("Loading space-mission-intelligence dataset …")
    return get_dataset("space_mission_intelligence", **kwargs)


def query_mission_lifecycle(
    ds: Any,
    mission_name: str,
) -> Dict[str, Any]:
    """Search for a mission by name and return lifecycle metadata.

    Parameters
    ----------
    ds : Dataset
        The dataset returned by :func:`load_mission_intelligence`.
    mission_name : str
        Full or partial mission name (case-insensitive substring match).

    Returns
    -------
    dict
        Dictionary with keys:
        ``mission_name``, ``budget``, ``milestones``, ``operator``,
        ``launch_mass_kg``, ``target_regime``.
        Returns an empty dict if no match is found.
    """
    target = mission_name.strip().lower()
    logger.info("Querying mission lifecycle for '%s' …", mission_name)

    # Column name candidates (datasets vary in naming)
    name_cols = ["mission_name", "Mission Name", "name", "mission", "Name"]
    budget_cols = ["budget", "Budget", "cost", "Cost", "budget_usd", "budget_million_usd"]
    milestone_cols = ["milestones", "Milestones", "status", "Status", "mission_status",
                      "Mission Status", "phase", "Phase"]
    operator_cols = ["operator", "Operator", "company", "Company", "agency", "Agency",
                     "organisation", "Organisation", "organization", "Organization"]
    mass_cols = ["launch_mass_kg", "launch_mass", "Launch Mass", "mass_kg", "Mass"]
    regime_cols = ["target_regime", "orbit", "Orbit", "orbit_type", "regime",
                   "target_orbit", "orbit_regime", "Orbit Type"]

    columns = set()
    try:
        columns = set(ds.column_names) if hasattr(ds, "column_names") else set()
    except Exception:
        pass

    def _pick(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in columns:
                return c
        return None

    name_col = _pick(name_cols)

    for row in ds:
        # Determine the name field dynamically
        row_name = ""
        if name_col and name_col in row:
            row_name = _safe_lower(row[name_col])
        else:
            # Fall back: try every candidate
            for c in name_cols:
                if c in row:
                    row_name = _safe_lower(row[c])
                    if row_name:
                        name_col = c
                        break

        if target in row_name:
            budget_col = _pick(budget_cols)
            milestone_col = _pick(milestone_cols)
            operator_col = _pick(operator_cols)
            mass_col = _pick(mass_cols)
            regime_col = _pick(regime_cols)

            result: Dict[str, Any] = {
                "mission_name": row.get(name_col, mission_name) if name_col else mission_name,
                "budget": row.get(budget_col) if budget_col else None,
                "milestones": row.get(milestone_col) if milestone_col else None,
                "operator": row.get(operator_col) if operator_col else None,
                "launch_mass_kg": _to_float(row.get(mass_col)) if mass_col else None,
                "target_regime": row.get(regime_col) if regime_col else None,
            }
            logger.info("Found mission: %s", result["mission_name"])
            return result

    logger.warning("No mission matching '%s' found.", mission_name)
    return {}


# ============================================================================
# 2. Space Agencies
# ============================================================================

def load_space_agencies(**kwargs) -> Any:
    """Load the space-agency-database dataset.

    Returns
    -------
    Dataset
        Hugging Face Dataset keyed ``"space_agencies"``.
    """
    logger.info("Loading space-agency-database dataset …")
    return get_dataset("space_agencies", **kwargs)


def get_agency_capabilities(
    ds: Any,
    agency_code: str,
) -> Dict[str, Any]:
    """Return agency information for a given agency code.

    Parameters
    ----------
    ds : Dataset
        The dataset returned by :func:`load_space_agencies`.
    agency_code : str
        Short code such as ``"NASA"``, ``"ESA"``, ``"CNSA"``, ``"ISRO"``
        (case-insensitive).

    Returns
    -------
    dict
        Keys: ``agency_code``, ``launch_sites``, ``funding``,
        ``dsn_assets``.  Empty dict if not found.
    """
    code = agency_code.strip().upper()
    logger.info("Looking up agency capabilities for '%s' …", code)

    code_cols = ["agency_code", "code", "Code", "Agency Code", "abbreviation",
                 "Abbreviation", "acronym", "Acronym"]
    site_cols = ["launch_sites", "Launch Sites", "sites", "launch_site",
                 "launchpad", "spaceport", "Spaceport"]
    funding_cols = ["funding", "Funding", "budget", "Budget", "annual_budget",
                    "Annual Budget", "funding_usd"]
    dsn_cols = ["dsn_assets", "DSN Assets", "dsn", "DSN", "ground_stations",
                "Ground Stations", "deep_space_network"]

    columns = set()
    try:
        columns = set(ds.column_names) if hasattr(ds, "column_names") else set()
    except Exception:
        pass

    def _pick(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in columns:
                return c
        return None

    for row in ds:
        # Try matching on code columns
        matched = False
        for c in code_cols:
            val = row.get(c)
            if val is not None and str(val).strip().upper() == code:
                matched = True
                break
        if not matched:
            # Try matching on name columns too
            for c in ["name", "Name", "agency_name", "Agency Name"]:
                val = row.get(c)
                if val is not None and code in str(val).strip().upper():
                    matched = True
                    break

        if matched:
            site_col = _pick(site_cols)
            fund_col = _pick(funding_cols)
            dsn_col = _pick(dsn_cols)

            result: Dict[str, Any] = {
                "agency_code": code,
                "launch_sites": row.get(site_col) if site_col else None,
                "funding": row.get(fund_col) if fund_col else None,
                "dsn_assets": row.get(dsn_col) if dsn_col else None,
            }
            logger.info("Found agency capabilities for %s", code)
            return result

    logger.warning("Agency '%s' not found in dataset.", agency_code)
    return {}


# ============================================================================
# 3. Historical Launch Log
# ============================================================================

def load_historical_launch_log(**kwargs) -> Any:
    """Load the historical space launch log dataset.

    Returns
    -------
    Dataset
        Hugging Face Dataset keyed ``"space_launch_log"``.
    """
    logger.info("Loading historical space-launch-log dataset …")
    return get_dataset("space_launch_log", **kwargs)


def calculate_launch_success_rate(
    ds: Any,
    vehicle_family: str,
) -> Dict[str, Any]:
    """Compute launch success/failure ratio and mean-time-between-failures
    (MTBF) for a vehicle family.

    Parameters
    ----------
    ds : Dataset
        Dataset from :func:`load_historical_launch_log`.
    vehicle_family : str
        Rocket family name, e.g. ``"Falcon 9"``, ``"Soyuz"``, ``"Ariane 5"``
        (case-insensitive substring match).

    Returns
    -------
    dict
        Keys: ``vehicle_family``, ``total_launches``, ``successes``,
        ``failures``, ``success_rate``, ``mtbf_days``.
    """
    target = vehicle_family.strip().lower()
    logger.info("Computing launch success rate for '%s' …", vehicle_family)

    vehicle_cols = ["vehicle", "Vehicle", "launch_vehicle", "Launch Vehicle",
                    "rocket", "Rocket", "vehicle_family", "Vehicle Family",
                    "LV Type", "lv_type"]
    status_cols = ["status", "Status", "outcome", "Outcome",
                   "mission_status", "Mission Status", "launch_status",
                   "Launch Status", "StatusMission"]
    date_cols = ["date", "Date", "launch_date", "Launch Date",
                 "launch_datetime", "Launch Datetime", "Datum"]

    success_keywords = {"success", "successful", "s", "1", "operational",
                        "nominal", "partial failure"}
    failure_keywords = {"failure", "failed", "f", "0", "prelaunch failure",
                        "launch failure"}

    successes = 0
    failures = 0
    failure_dates: List[datetime] = []

    for row in ds:
        # Match vehicle family
        matched_vehicle = False
        for c in vehicle_cols:
            val = row.get(c)
            if val is not None and target in str(val).strip().lower():
                matched_vehicle = True
                break
        if not matched_vehicle:
            continue

        # Determine success / failure
        status_val = None
        for c in status_cols:
            if c in row and row[c] is not None:
                status_val = _safe_lower(row[c])
                break

        if status_val is None:
            continue

        if any(kw in status_val for kw in success_keywords):
            successes += 1
        elif any(kw in status_val for kw in failure_keywords):
            failures += 1
            # Record date for MTBF
            for c in date_cols:
                dt = _parse_date(row.get(c))
                if dt is not None:
                    failure_dates.append(dt)
                    break

    total = successes + failures
    success_rate = (successes / total) if total > 0 else 0.0

    # MTBF: average interval between consecutive failures
    mtbf_days: Optional[float] = None
    if len(failure_dates) >= 2:
        failure_dates.sort()
        intervals = [
            (failure_dates[i + 1] - failure_dates[i]).total_seconds() / 86400.0
            for i in range(len(failure_dates) - 1)
        ]
        mtbf_days = float(np.mean(intervals))

    result = {
        "vehicle_family": vehicle_family,
        "total_launches": total,
        "successes": successes,
        "failures": failures,
        "success_rate": round(success_rate, 6),
        "mtbf_days": round(mtbf_days, 2) if mtbf_days is not None else None,
    }
    logger.info("Launch stats for '%s': %d/%d success (%.2f%%)",
                vehicle_family, successes, total, success_rate * 100)
    return result


# ============================================================================
# 4. SpaceX Launches
# ============================================================================

def load_spacex_launches(**kwargs) -> Any:
    """Load the SpaceX launches dataset.

    Returns
    -------
    Dataset
        Hugging Face Dataset keyed ``"spacex_launches"``.
    """
    logger.info("Loading SpaceX launches dataset …")
    return get_dataset("spacex_launches", **kwargs)


def analyze_booster_turnaround_time(
    ds: Any,
) -> Dict[str, Any]:
    """Measure turnaround days between successive flights of the same
    recovered booster serial number.

    Parameters
    ----------
    ds : Dataset
        Dataset from :func:`load_spacex_launches`.

    Returns
    -------
    dict
        Keys:
        - ``booster_serials``: list of unique booster serial IDs analysed.
        - ``turnaround_days``: dict mapping serial → list of inter-flight
          day intervals.
        - ``mean_turnaround_days``: overall mean turnaround across all
          boosters.
        - ``min_turnaround_days``: fastest recorded turnaround.
        - ``max_turnaround_days``: slowest recorded turnaround.
    """
    logger.info("Analysing booster turnaround times …")

    serial_cols = ["booster_serial", "core_serial", "Core Serial",
                   "serial", "Serial", "booster", "Booster",
                   "cores.core_serial", "core"]
    date_cols = ["date_utc", "launch_date_utc", "launch_date", "date",
                 "Date", "Launch Date", "date_local"]

    # Gather (serial, date) pairs
    flights: Dict[str, List[datetime]] = {}

    for row in ds:
        serial_val: Optional[str] = None
        for c in serial_cols:
            v = row.get(c)
            if v is not None and str(v).strip():
                serial_val = str(v).strip()
                break
        if serial_val is None:
            continue

        date_val: Optional[datetime] = None
        for c in date_cols:
            date_val = _parse_date(row.get(c))
            if date_val is not None:
                break
        if date_val is None:
            continue

        flights.setdefault(serial_val, []).append(date_val)

    # Compute turnaround intervals
    turnaround: Dict[str, List[float]] = {}
    all_intervals: List[float] = []

    for serial, dates in flights.items():
        if len(dates) < 2:
            continue
        dates.sort()
        intervals = [
            (dates[i + 1] - dates[i]).total_seconds() / 86400.0
            for i in range(len(dates) - 1)
        ]
        turnaround[serial] = [round(d, 2) for d in intervals]
        all_intervals.extend(intervals)

    result: Dict[str, Any] = {
        "booster_serials": list(turnaround.keys()),
        "turnaround_days": turnaround,
        "mean_turnaround_days": round(float(np.mean(all_intervals)), 2) if all_intervals else None,
        "min_turnaround_days": round(float(np.min(all_intervals)), 2) if all_intervals else None,
        "max_turnaround_days": round(float(np.max(all_intervals)), 2) if all_intervals else None,
    }
    logger.info("Analysed %d boosters with reuse, %d total intervals",
                len(turnaround), len(all_intervals))
    return result


# ============================================================================
# 5. Rocket Lab Launches
# ============================================================================

def load_rocket_lab_launches(**kwargs) -> Any:
    """Load the Rocket Lab launches dataset.

    Returns
    -------
    Dataset
        Hugging Face Dataset keyed ``"rocket_lab_launches"``.
    """
    logger.info("Loading Rocket Lab launches dataset …")
    return get_dataset("rocket_lab_launches", **kwargs)


def get_electron_kickstage_performance(
    ds: Any,
) -> Dict[str, Any]:
    """Evaluate Electron kick-stage payload delivery to ~500 km SSO.

    Scans the dataset for missions utilising the Photon / kick-stage
    upper stage and extracts payload mass and target orbit parameters.

    Parameters
    ----------
    ds : Dataset
        Dataset from :func:`load_rocket_lab_launches`.

    Returns
    -------
    dict
        Keys:
        - ``missions``: number of kick-stage missions found.
        - ``payloads``: list of dicts with ``mission_name``,
          ``payload_mass_kg``, ``target_orbit_km``.
        - ``mean_payload_mass_kg``: mean payload mass to ~500 km SSO.
        - ``max_payload_mass_kg``: maximum payload mass delivered.
    """
    logger.info("Evaluating Electron kick-stage performance to 500 km SSO …")

    name_cols = ["mission_name", "Mission Name", "mission", "Mission",
                 "name", "Name", "flight_name"]
    mass_cols = ["payload_mass_kg", "Payload Mass", "payload_mass",
                 "mass_kg", "payload_kg", "mass", "Mass"]
    orbit_cols = ["orbit_km", "orbit_altitude_km", "altitude_km",
                  "target_orbit_km", "orbit", "Orbit", "apogee_km",
                  "target_altitude", "orbit_altitude"]
    kickstage_cols = ["kick_stage", "Kick Stage", "upper_stage",
                      "Upper Stage", "stage_type", "photon", "Photon",
                      "kickstage"]

    payloads: List[Dict[str, Any]] = []
    masses: List[float] = []

    for row in ds:
        # Determine whether this mission used a kick-stage
        has_kickstage = False
        for c in kickstage_cols:
            v = row.get(c)
            if v is not None:
                sv = _safe_lower(v)
                if sv in ("true", "yes", "1", "photon"):
                    has_kickstage = True
                    break
                if "kick" in sv or "photon" in sv:
                    has_kickstage = True
                    break

        # If no explicit kick-stage column, include all missions and
        # filter by orbit altitude ≈ 500 km SSO
        orbit_val: Optional[float] = None
        for c in orbit_cols:
            orbit_val = _to_float(row.get(c))
            if orbit_val is not None:
                break

        # Accept kick-stage missions, or those at 400-650 km (typical SSO)
        if not has_kickstage:
            if orbit_val is None or not (400.0 <= orbit_val <= 650.0):
                continue

        mission_name_val: Optional[str] = None
        for c in name_cols:
            v = row.get(c)
            if v is not None and str(v).strip():
                mission_name_val = str(v).strip()
                break

        mass_val: Optional[float] = None
        for c in mass_cols:
            mass_val = _to_float(row.get(c))
            if mass_val is not None:
                break

        entry: Dict[str, Any] = {
            "mission_name": mission_name_val,
            "payload_mass_kg": mass_val,
            "target_orbit_km": orbit_val,
        }
        payloads.append(entry)
        if mass_val is not None and mass_val > 0:
            masses.append(mass_val)

    result: Dict[str, Any] = {
        "missions": len(payloads),
        "payloads": payloads,
        "mean_payload_mass_kg": round(float(np.mean(masses)), 3) if masses else None,
        "max_payload_mass_kg": round(float(np.max(masses)), 3) if masses else None,
    }
    logger.info("Found %d kick-stage missions; mean payload %.1f kg",
                len(payloads), result["mean_payload_mass_kg"] or 0.0)
    return result


# ============================================================================
# 6. Blue Origin Launches
# ============================================================================

def load_blue_origin_launches(**kwargs) -> Any:
    """Load the Blue Origin launches dataset.

    Returns
    -------
    Dataset
        Hugging Face Dataset keyed ``"blue_origin_launches"``.
    """
    logger.info("Loading Blue Origin launches dataset …")
    return get_dataset("blue_origin_launches", **kwargs)


def get_new_shepard_apogee_stats(
    ds: Any,
) -> Dict[str, Any]:
    """Compute New Shepard sub-orbital flight statistics.

    Parameters
    ----------
    ds : Dataset
        Dataset from :func:`load_blue_origin_launches`.

    Returns
    -------
    dict
        Keys:
        - ``flights``: total New Shepard flights analysed.
        - ``mean_apogee_km``: mean apogee altitude [km].
        - ``max_apogee_km``: maximum apogee altitude [km].
        - ``min_apogee_km``: minimum apogee altitude [km].
        - ``mean_microgravity_duration_s``: mean micro-g duration [s].
        - ``peak_acceleration_g``: maximum recorded acceleration [g].
    """
    logger.info("Computing New Shepard apogee statistics …")

    vehicle_cols = ["vehicle", "Vehicle", "rocket", "Rocket",
                    "launch_vehicle", "Launch Vehicle"]
    apogee_cols = ["apogee_km", "Apogee", "apogee", "peak_altitude_km",
                   "altitude_km", "max_altitude_km", "Apogee (km)"]
    microg_cols = ["microgravity_duration_s", "microgravity_s",
                   "microgravity_duration", "Microgravity Duration",
                   "micro_g_s", "microg_duration_s",
                   "Microgravity Duration (s)"]
    accel_cols = ["peak_acceleration_g", "max_g", "peak_g",
                  "Peak Acceleration", "acceleration_g", "max_acceleration_g",
                  "Peak G"]

    apogees: List[float] = []
    microg_durations: List[float] = []
    peak_accels: List[float] = []
    flights = 0

    for row in ds:
        # Filter to New Shepard flights
        is_ns = False
        for c in vehicle_cols:
            v = row.get(c)
            if v is not None and "shepard" in _safe_lower(v):
                is_ns = True
                break
        # If no vehicle column or all rows are New Shepard (single-vehicle dataset)
        if not is_ns:
            # Assume single-vehicle dataset if no vehicle column matched
            has_vehicle_col = any(c in row for c in vehicle_cols)
            if has_vehicle_col:
                continue
            # No vehicle column at all → treat every row as New Shepard
            is_ns = True

        flights += 1

        for c in apogee_cols:
            v = _to_float(row.get(c))
            if v is not None:
                apogees.append(v)
                break

        for c in microg_cols:
            v = _to_float(row.get(c))
            if v is not None:
                microg_durations.append(v)
                break

        for c in accel_cols:
            v = _to_float(row.get(c))
            if v is not None:
                peak_accels.append(v)
                break

    result: Dict[str, Any] = {
        "flights": flights,
        "mean_apogee_km": round(float(np.mean(apogees)), 2) if apogees else None,
        "max_apogee_km": round(float(np.max(apogees)), 2) if apogees else None,
        "min_apogee_km": round(float(np.min(apogees)), 2) if apogees else None,
        "mean_microgravity_duration_s": (
            round(float(np.mean(microg_durations)), 2) if microg_durations else None
        ),
        "peak_acceleration_g": (
            round(float(np.max(peak_accels)), 3) if peak_accels else None
        ),
    }
    logger.info("New Shepard: %d flights, mean apogee %.1f km",
                flights, result["mean_apogee_km"] or 0.0)
    return result


# ============================================================================
# 7. Launch Vehicle Specs
# ============================================================================

def load_launch_vehicle_specs(**kwargs) -> Any:
    """Load the launch vehicles specifications dataset.

    Returns
    -------
    Dataset
        Hugging Face Dataset keyed ``"launch_vehicles"``.
    """
    logger.info("Loading launch vehicle specs dataset …")
    return get_dataset("launch_vehicles", **kwargs)


def compute_payload_capacity(
    ds: Any,
    vehicle_name: str,
    target_orbit: str = "LEO",
) -> Dict[str, Any]:
    """Look up (or interpolate) payload capacity for a launch vehicle.

    Parameters
    ----------
    ds : Dataset
        Dataset from :func:`load_launch_vehicle_specs`.
    vehicle_name : str
        Launch vehicle name (case-insensitive substring match),
        e.g. ``"Falcon 9"``, ``"Ariane 6"``, ``"Starship"``.
    target_orbit : str, optional
        Target orbit regime: ``"LEO"``, ``"SSO"``, ``"GTO"``, ``"GEO"``,
        ``"TLI"`` etc.  Defaults to ``"LEO"``.

    Returns
    -------
    dict
        Keys: ``vehicle_name``, ``target_orbit``, ``payload_capacity_kg``,
        ``payload_capacity_source`` (``"exact"`` or ``"interpolated"``).
    """
    target_vehicle = vehicle_name.strip().lower()
    target_regime = target_orbit.strip().upper()
    logger.info("Looking up payload capacity for '%s' → %s …",
                vehicle_name, target_regime)

    name_cols = ["vehicle_name", "Vehicle Name", "name", "Name",
                 "launch_vehicle", "Launch Vehicle", "rocket", "Rocket",
                 "vehicle", "Vehicle"]

    # Map orbit regime to candidate columns / sub-keys
    orbit_col_map: Dict[str, List[str]] = {
        "LEO": ["payload_leo_kg", "leo_kg", "LEO", "Payload LEO (kg)",
                "payload_leo", "leo_payload_kg", "payload_capacity_leo_kg"],
        "SSO": ["payload_sso_kg", "sso_kg", "SSO", "Payload SSO (kg)",
                "payload_sso", "sso_payload_kg"],
        "GTO": ["payload_gto_kg", "gto_kg", "GTO", "Payload GTO (kg)",
                "payload_gto", "gto_payload_kg", "payload_capacity_gto_kg"],
        "GEO": ["payload_geo_kg", "geo_kg", "GEO", "Payload GEO (kg)",
                "payload_geo", "geo_payload_kg"],
        "TLI": ["payload_tli_kg", "tli_kg", "TLI", "Payload TLI (kg)",
                "payload_tli"],
    }

    # Also search a generic "payload_kg" column
    generic_cols = ["payload_kg", "payload_capacity_kg", "Payload (kg)",
                    "payload_mass_kg", "capacity_kg"]

    regime_candidates = orbit_col_map.get(target_regime, [])

    for row in ds:
        # Match vehicle name
        matched = False
        for c in name_cols:
            v = row.get(c)
            if v is not None and target_vehicle in _safe_lower(v):
                matched = True
                actual_name = str(v).strip()
                break
        if not matched:
            continue

        # Try regime-specific column
        for c in regime_candidates:
            val = _to_float(row.get(c))
            if val is not None:
                return {
                    "vehicle_name": actual_name,
                    "target_orbit": target_regime,
                    "payload_capacity_kg": round(val, 2),
                    "payload_capacity_source": "exact",
                }

        # Try generic payload column
        for c in generic_cols:
            val = _to_float(row.get(c))
            if val is not None:
                # Apply approximate derating for non-LEO orbits
                derating: Dict[str, float] = {
                    "LEO": 1.0,
                    "SSO": 0.85,
                    "GTO": 0.45,
                    "GEO": 0.22,
                    "TLI": 0.30,
                }
                factor = derating.get(target_regime, 0.5)
                interpolated = val * factor
                return {
                    "vehicle_name": actual_name,
                    "target_orbit": target_regime,
                    "payload_capacity_kg": round(interpolated, 2),
                    "payload_capacity_source": "interpolated",
                }

        # Vehicle matched but no capacity data available
        logger.warning("Vehicle '%s' found but no payload capacity data for %s",
                       actual_name, target_regime)
        return {
            "vehicle_name": actual_name,
            "target_orbit": target_regime,
            "payload_capacity_kg": None,
            "payload_capacity_source": None,
        }

    logger.warning("Vehicle '%s' not found in dataset.", vehicle_name)
    return {
        "vehicle_name": vehicle_name,
        "target_orbit": target_regime,
        "payload_capacity_kg": None,
        "payload_capacity_source": None,
    }


# ============================================================================
# 8. Starlink Ground Stations
# ============================================================================

def load_starlink_ground_stations(**kwargs) -> Any:
    """Load the Starlink ground stations dataset.

    Returns
    -------
    Dataset
        Hugging Face Dataset keyed ``"starlink_ground_stations"``.
    """
    logger.info("Loading Starlink ground stations dataset …")
    return get_dataset("starlink_ground_stations", **kwargs)


def calculate_gateway_handover(
    lat: float,
    lon: float,
    station_lat: float,
    station_lon: float,
    altitude_km: float,
    min_elev_deg: float = 25.0,
) -> Dict[str, Any]:
    """Determine gateway visibility from a satellite sub-point based on
    elevation angle geometry.

    The elevation angle from the ground station to a satellite at
    altitude *h* is computed as:

    .. math::

        \\gamma = \\arccos\\bigl(\\sin\\phi_1 \\sin\\phi_2
                   + \\cos\\phi_1 \\cos\\phi_2 \\cos(\\Delta\\lambda)\\bigr)

        el = \\arctan\\!\\left(
                \\frac{\\cos\\gamma - \\frac{R}{R+h}}{\\sin\\gamma}
             \\right)

    where *R* is the Earth mean radius, *h* the satellite altitude, and
    γ the central angle (angular separation) between the sub-satellite
    point and the ground station.

    Parameters
    ----------
    lat, lon : float
        Sub-satellite point latitude and longitude [deg].
    station_lat, station_lon : float
        Ground station latitude and longitude [deg].
    altitude_km : float
        Satellite altitude above mean sea level [km].
    min_elev_deg : float, optional
        Minimum elevation angle for a valid handover [deg].
        Default is 25.0°.

    Returns
    -------
    dict
        Keys:
        - ``angular_separation_deg``: central angle γ [deg].
        - ``elevation_deg``: computed elevation angle [deg].
        - ``is_visible``: True if elevation ≥ min_elev_deg.
        - ``slant_range_km``: slant range to satellite [km].
    """
    # Convert to radians
    phi1 = math.radians(lat)
    phi2 = math.radians(station_lat)
    dlambda = math.radians(lon - station_lon)

    # Central angle (angular separation) via spherical law of cosines
    cos_gamma = (math.sin(phi1) * math.sin(phi2)
                 + math.cos(phi1) * math.cos(phi2) * math.cos(dlambda))
    # Clamp for numerical safety
    cos_gamma = max(-1.0, min(1.0, cos_gamma))
    gamma = math.acos(cos_gamma)

    R = R_EARTH_KM  # from constants
    h = altitude_km

    # Elevation angle
    sin_gamma = math.sin(gamma)
    if sin_gamma < 1e-12:
        # Satellite is directly overhead the station
        el_rad = math.pi / 2.0
    else:
        el_rad = math.atan2(cos_gamma - R / (R + h), sin_gamma)

    el_deg = math.degrees(el_rad)

    # Slant range via law of cosines in the triangle
    # d² = R² + (R+h)² - 2·R·(R+h)·cos(gamma - el correction)
    # Simplified from geometric relation:
    slant_range_km = math.sqrt(
        R ** 2 + (R + h) ** 2
        - 2.0 * R * (R + h) * cos_gamma
    )

    is_visible = el_deg >= min_elev_deg

    result = {
        "angular_separation_deg": round(math.degrees(gamma), 6),
        "elevation_deg": round(el_deg, 4),
        "is_visible": is_visible,
        "slant_range_km": round(slant_range_km, 3),
    }
    logger.debug("Gateway handover: γ=%.2f°, el=%.2f°, visible=%s",
                 result["angular_separation_deg"], el_deg, is_visible)
    return result


# ============================================================================
# 9. SatNOGS Transmitters
# ============================================================================

def load_satnogs_transmitters(**kwargs) -> Any:
    """Load the SatNOGS transmitters dataset.

    Returns
    -------
    Dataset
        Hugging Face Dataset keyed ``"satnogs_transmitters"``.
    """
    logger.info("Loading SatNOGS transmitters dataset …")
    return get_dataset("satnogs_transmitters", **kwargs)


def find_active_frequencies(
    ds: Any,
    norad_id: int,
) -> Dict[str, Any]:
    """Extract active downlink frequencies, modulation schemes, and modes
    for a satellite identified by its NORAD catalog number.

    Parameters
    ----------
    ds : Dataset
        Dataset from :func:`load_satnogs_transmitters`.
    norad_id : int
        NORAD catalog ID of the satellite.

    Returns
    -------
    dict
        Keys:
        - ``norad_id``: the queried NORAD ID.
        - ``transmitters``: list of dicts, each with
          ``downlink_freq_hz``, ``modulation``, ``mode``, ``baud_rate``,
          ``status``.
        - ``count``: number of active transmitters found.
    """
    logger.info("Finding active frequencies for NORAD ID %d …", norad_id)

    norad_cols = ["norad_cat_id", "NORAD Cat ID", "norad_id",
                  "norad", "NORAD", "sat_id", "catalog_number",
                  "Norad Cat ID"]
    freq_cols = ["downlink_low", "downlink_freq", "Downlink Low",
                 "downlink_low_hz", "frequency_hz", "frequency",
                 "downlink", "Downlink"]
    mod_cols = ["modulation", "Modulation", "mode_modulation"]
    mode_cols = ["mode", "Mode", "service", "Service", "type", "Type"]
    baud_cols = ["baud", "Baud", "baud_rate", "baudrate", "Baud Rate"]
    status_cols = ["status", "Status", "alive", "Alive", "active", "Active"]

    active_keywords = {"active", "alive", "true", "1", "yes", "operational"}

    transmitters: List[Dict[str, Any]] = []

    for row in ds:
        # Match NORAD ID
        matched = False
        for c in norad_cols:
            v = row.get(c)
            if v is not None:
                try:
                    if int(v) == norad_id:
                        matched = True
                        break
                except (ValueError, TypeError):
                    continue
        if not matched:
            continue

        # Check status — include if status is "active" or if no status column
        status_val: Optional[str] = None
        for c in status_cols:
            v = row.get(c)
            if v is not None:
                status_val = _safe_lower(v)
                break

        if status_val is not None and status_val not in active_keywords:
            continue

        # Extract frequency
        freq: Optional[float] = None
        for c in freq_cols:
            freq = _to_float(row.get(c))
            if freq is not None:
                break

        # Extract modulation
        modulation: Optional[str] = None
        for c in mod_cols:
            v = row.get(c)
            if v is not None and str(v).strip():
                modulation = str(v).strip()
                break

        # Extract mode
        mode: Optional[str] = None
        for c in mode_cols:
            v = row.get(c)
            if v is not None and str(v).strip():
                mode = str(v).strip()
                break

        # Extract baud rate
        baud: Optional[float] = None
        for c in baud_cols:
            baud = _to_float(row.get(c))
            if baud is not None:
                break

        transmitters.append({
            "downlink_freq_hz": freq,
            "modulation": modulation,
            "mode": mode,
            "baud_rate": baud,
            "status": status_val or "active",
        })

    result = {
        "norad_id": norad_id,
        "transmitters": transmitters,
        "count": len(transmitters),
    }
    logger.info("Found %d active transmitter(s) for NORAD %d",
                len(transmitters), norad_id)
    return result
