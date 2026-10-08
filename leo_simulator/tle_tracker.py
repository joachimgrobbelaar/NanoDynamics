"""Real-time LEO Satellite Tracker using CelesTrak OMM (Orbit Mean-Elements Message) feeds.

Fetches orbital elements for active LEO satellites (Space Stations, CubeSats, Starlink, Weather),
solves Kepler's equation to derive instantaneous true anomaly and Cartesian state vectors,
and generates NanoDynamics Satellite parameters.
"""

from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request
import numpy as np

from leo_simulator.constants import MU_EARTH, R_EARTH
from leo_simulator.orbit.elements import OrbitalElements, coe_to_rv

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "live_satellites_cache.json")
CACHE_TTL_SECONDS = 3600 * 24  # 24 hours cache

GROUPS = {
    "stations": {
        "name": "Space Stations & Resupply",
        "description": "International Space Station (ISS), Tiangong (CSS), Crew Dragon, Cygnus, Progress, Soyuz",
        "default_mass": 420000.0,
        "default_area": 1200.0,
        "default_color": "#00ffcc",
        "icon": "space_station",
        "celestrak_group": "stations",
    },
    "visual": {
        "name": "100 Brightest Visual Satellites",
        "description": "Hubble Space Telescope (HST), Envisat, Genesis, Seasat, active visual targets",
        "default_mass": 11000.0,
        "default_area": 25.0,
        "default_color": "#eab308",
        "icon": "satellite",
        "celestrak_group": "visual",
    },
    "resource": {
        "name": "Earth Observation & Weather",
        "description": "Landsat 8/9, Sentinel 1/2/3/5P, Terra, Aqua, Suomi NPP, meteorological satellites",
        "default_mass": 2200.0,
        "default_area": 12.0,
        "default_color": "#22c55e",
        "icon": "satellite",
        "celestrak_group": "resource",
    },
    "science": {
        "name": "Space & Earth Science",
        "description": "Astrophysics, Fermi, Swift, SWARM, GRACE, and scientific orbiters",
        "default_mass": 1500.0,
        "default_area": 6.0,
        "default_color": "#38bdf8",
        "icon": "satellite",
        "celestrak_group": "science",
    },
    "oneweb": {
        "name": "OneWeb Constellation",
        "description": "Global low-latency broadband internet satellite fleet",
        "default_mass": 147.0,
        "default_area": 3.2,
        "default_color": "#ec4899",
        "icon": "satellite",
        "celestrak_group": "oneweb",
    },
    "iridium": {
        "name": "Iridium NEXT Constellation",
        "description": "Global cross-linked low Earth orbit communications network",
        "default_mass": 860.0,
        "default_area": 9.5,
        "default_color": "#a855f7",
        "icon": "satellite",
        "celestrak_group": "iridium-NEXT",
    },
    "planet": {
        "name": "Planet Labs Constellation",
        "description": "Commercial high-resolution Earth imaging PlanetScope Doves and SkySats",
        "default_mass": 5.5,
        "default_area": 0.05,
        "default_color": "#f97316",
        "icon": "satellite",
        "celestrak_group": "planet",
    },
    "spire": {
        "name": "Spire Lemur Constellation",
        "description": "Global maritime, aviation, and GNSS radio occultation weather nanosatellites",
        "default_mass": 4.5,
        "default_area": 0.04,
        "default_color": "#06b6d4",
        "icon": "satellite",
        "celestrak_group": "spire",
    },
    "cubesat": {
        "name": "CubeSats & NanoSats",
        "description": "Educational, scientific, and experimental 1U-12U university nanosatellites",
        "default_mass": 4.0,
        "default_area": 0.03,
        "default_color": "#60a5fa",
        "icon": "satellite",
        "celestrak_group": "cubesat",
    },
}


def solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E (radians)."""
    M = M % (2.0 * np.pi)
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1.0 - e * np.cos(E)
        if abs(f_prime) < 1e-12:
            break
        dE = f / f_prime
        E -= dE
        if abs(dE) < tol:
            break
    return float(E)


def eccentric_to_true_anomaly(E: float, e: float) -> float:
    """Convert eccentric anomaly E to true anomaly nu (radians)."""
    sin_half_nu = np.sqrt(1.0 + e) * np.sin(E / 2.0)
    cos_half_nu = np.sqrt(1.0 - e) * np.cos(E / 2.0)
    nu = 2.0 * np.arctan2(sin_half_nu, cos_half_nu)
    return float(nu % (2.0 * np.pi))


def parse_omm_record(record: Dict[str, Any], group: str = "cubesat") -> Optional[Dict[str, Any]]:
    """Parse a single CelesTrak OMM JSON record into Keplerian elements and physical parameters."""
    try:
        object_name = record.get("OBJECT_NAME", "Unknown Satellite").strip()
        norad_id = int(record.get("NORAD_CAT_ID", 0))
        mean_motion_rev_day = float(record.get("MEAN_MOTION", 0.0))
        eccentricity = float(record.get("ECCENTRICITY", 0.0))
        inclination_deg = float(record.get("INCLINATION", 0.0))
        raan_deg = float(record.get("RA_OF_ASC_NODE", 0.0))
        arg_pe_deg = float(record.get("ARG_OF_PERICENTER", 0.0))
        mean_anomaly_deg = float(record.get("MEAN_ANOMALY", 0.0))
        epoch = str(record.get("EPOCH", ""))

        if mean_motion_rev_day <= 0.0 or eccentricity < 0.0 or eccentricity >= 1.0:
            return None

        # Mean motion in rad/s: n = rev/day * 2pi / 86400
        n_rad_s = mean_motion_rev_day * (2.0 * np.pi / 86400.0)
        # Semi-major axis a = (mu / n^2)^(1/3)
        semi_major_axis_m = (MU_EARTH / (n_rad_s**2)) ** (1.0 / 3.0)
        altitude_km = (semi_major_axis_m - R_EARTH) / 1000.0

        # Discard deep space / decayed / invalid LEO ranges
        if altitude_km < 120.0 or altitude_km > 2000.0:
            return None

        # Solve Kepler's equation for true anomaly
        mean_anomaly_rad = np.radians(mean_anomaly_deg)
        eccentric_anomaly_rad = solve_kepler(mean_anomaly_rad, eccentricity)
        true_anomaly_rad = eccentric_to_true_anomaly(eccentric_anomaly_rad, eccentricity)
        raan_deg = float(round(raan_deg % 360.0, 4))
        arg_pe_deg = float(round(arg_pe_deg % 360.0, 4))
        true_anomaly_deg = float(round(float(np.degrees(true_anomaly_rad)) % 360.0, 4))

        group_meta = GROUPS.get(group, GROUPS["cubesat"])

        # Specific custom overrides for well-known spacecraft
        mass = group_meta["default_mass"]
        area = group_meta["default_area"]
        color = group_meta["default_color"]

        if "ISS" in object_name or norad_id == 25544:
            mass = 450000.0
            area = 1500.0
            color = "#00ffcc"
        elif "TIANGONG" in object_name or "CSS" in object_name or norad_id == 48274:
            mass = 100000.0
            area = 400.0
            color = "#f59e0b"
        elif "HST" in object_name or "HUBBLE" in object_name or norad_id == 20580:
            mass = 11110.0
            area = 30.0
            color = "#38bdf8"
        elif "ENVISAT" in object_name or norad_id == 27386:
            mass = 8211.0
            area = 80.0
            color = "#ef4444"

        return {
            "norad_id": norad_id,
            "name": object_name[:90],
            "group": group,
            "epoch": epoch,
            "semi_major_axis_km": float(semi_major_axis_m / 1000.0),
            "altitude_km": float(round(altitude_km, 2)),
            "eccentricity": float(round(eccentricity, 6)),
            "inclination_deg": float(round(inclination_deg, 4)),
            "raan_deg": raan_deg,
            "arg_periapsis_deg": arg_pe_deg,
            "true_anomaly_deg": true_anomaly_deg,
            "period_minutes": float(round((2.0 * np.pi / n_rad_s) / 60.0, 2)),
            "mass": float(mass),
            "drag_area": float(area),
            "cd": 2.2,
            "color": color,
            "icon": group_meta["icon"],
        }
    except Exception as e:
        logger.debug(f"Failed to parse OMM record: {e}")
        return None


def fetch_celestrak_group(group: str = "stations", timeout: float = 3.5) -> List[Dict[str, Any]]:
    """Fetch live OMM data from CelesTrak API."""
    celestrak_param = GROUPS.get(group, {}).get("celestrak_group", group)
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={celestrak_param}&FORMAT=json"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NanoDynamics-LEO-Tracker/1.0 (Academic Research; Linux)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if response.status == 200:
            data = json.loads(response.read().decode("utf-8"))
            parsed_sats = []
            for item in data:
                sat = parse_omm_record(item, group=group)
                if sat:
                    parsed_sats.append(sat)
            return parsed_sats
    return []


def load_cache() -> Dict[str, Any]:
    """Load local cached satellite records."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"timestamp": 0, "groups": {}}


def save_cache(cache_data: Dict[str, Any]) -> None:
    """Save satellite records to local cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save satellite cache: {e}")


def get_live_catalog(
    group: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """Get active LEO satellites with fast cached fallback."""
    cache_data = load_cache()
    now = time.time()
    all_sats = []

    groups_to_query = [group] if group and group in GROUPS else list(GROUPS.keys())

    for grp in groups_to_query:
        grp_cache = cache_data.get("groups", {}).get(grp, {})
        sats = grp_cache.get("satellites", [])

        if force_refresh or not sats:
            try:
                live_sats = fetch_celestrak_group(grp, timeout=2.0)
                if live_sats:
                    sats = live_sats
                    if "groups" not in cache_data:
                        cache_data["groups"] = {}
                    cache_data["groups"][grp] = {
                        "timestamp": now,
                        "satellites": sats,
                    }
                    save_cache(cache_data)
            except Exception as e:
                logger.info(f"CelesTrak online fetch for {grp} failed ({e}), using cached fallback.")

        all_sats.extend(sats)

    # Filter by search query if provided
    if search:
        q = search.lower().strip()
        all_sats = [
            s for s in all_sats
            if q in s["name"].lower() or q in str(s["norad_id"])
        ]

    return all_sats[:limit]


def get_satellite_by_norad(norad_id: int) -> Optional[Dict[str, Any]]:
    """Look up a satellite by its NORAD ID from all groups."""
    catalog = get_live_catalog(limit=10000)
    for sat in catalog:
        if sat["norad_id"] == norad_id:
            return sat
    return None
