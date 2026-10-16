"""POI Orbit Designer and Specialized Constellation Presets.

Calculates resonant repeating ground track orbits targeted to specific geographic
points of interest (cities/POIs) and provides specialized orbit presets including
GEO, GSO, Molniya, Tundra, and Sun-Synchronous Orbits (SSO).
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from leo_simulator.constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH
from leo_simulator.orbit.elements import OrbitalElements, circular_velocity, coe_to_rv


@dataclass
class PointOfInterest:
    name: str
    country: str
    lat_deg: float
    lon_deg: float
    description: str = ""
    urban_blocks: int = 18
    max_height_km: float = 42.0


# Curated catalog of global major cities and key ground stations
MAJOR_POIS: List[PointOfInterest] = [
    PointOfInterest("Tokyo", "Japan", 35.6762, 139.6503, "Greater Tokyo Area, Kanto", 28, 55.0),
    PointOfInterest("London", "United Kingdom", 51.5074, -0.1278, "Greenwich Prime Meridian", 24, 48.0),
    PointOfInterest("New York", "United States", 40.7128, -74.0060, "North American Atlantic coast", 30, 60.0),
    PointOfInterest("Johannesburg", "South Africa", -26.2041, 28.0473, "Highveld Witwatersrand Megacity", 22, 46.0),
    PointOfInterest("Paris", "France", 48.8566, 2.3522, "Western European hub", 22, 44.0),
    PointOfInterest("Singapore", "Singapore", 1.3521, 103.8198, "Equatorial maritime corridor", 26, 52.0),
    PointOfInterest("Sydney", "Australia", -33.8688, 151.2093, "Southern Hemisphere Pacific", 20, 45.0),
    PointOfInterest("Cairo", "Egypt", 30.0444, 31.2357, "Nile Delta / Mediterranean", 20, 40.0),
    PointOfInterest("Nairobi", "Kenya", -1.2921, 36.8219, "East African Rift / Equator", 18, 38.0),
    PointOfInterest("São Paulo", "Brazil", -23.5505, -46.6333, "South American Atlantic metropolis", 26, 50.0),
    PointOfInterest("Mumbai", "India", 19.0760, 72.8777, "Arabian Sea / Indian subcontinent", 24, 48.0),
    PointOfInterest("San Francisco", "United States", 37.7749, -122.4194, "Silicon Valley / Pacific hub", 22, 46.0),
    PointOfInterest("Dubai", "United Arab Emirates", 25.2048, 55.2708, "Arabian Gulf financial corridor", 24, 64.0),
    PointOfInterest("Cape Town", "South Africa", -33.9249, 18.4241, "Cape of Good Hope", 18, 40.0),
    PointOfInterest("Beijing", "China", 39.9042, 116.4074, "Northern East Asia", 26, 52.0),
    PointOfInterest("Berlin", "Germany", 52.5200, 13.4050, "Central Europe", 20, 42.0),
    PointOfInterest("Honolulu", "United States", 21.3069, -157.8583, "Central Pacific isolated station", 16, 36.0),
    PointOfInterest("Reykjavik", "Iceland", 64.1466, -21.9426, "Sub-Arctic North Atlantic", 14, 32.0),
    PointOfInterest("Buenos Aires", "Argentina", -34.6037, -58.3816, "Río de la Plata", 20, 44.0),
    PointOfInterest("Svalbard", "Norway", 78.2232, 15.6267, "Arctic Satellite Ground Station", 12, 30.0),
    PointOfInterest("Cape Canaveral", "United States", 28.3922, -80.6077, "Kennedy Space Center / Space Launch Delta 45", 16, 38.0),
    PointOfInterest("Houston", "United States", 29.5593, -95.0900, "NASA Johnson Space Center Mission Control", 20, 44.0),
    PointOfInterest("Kourou", "French Guiana", 5.2372, -52.7606, "Guiana Space Centre (ESA Europe Spaceport)", 14, 34.0),
    PointOfInterest("Baikonur", "Kazakhstan", 45.9650, 63.3050, "Baikonur Cosmodrome launch complex", 14, 34.0),
    PointOfInterest("Tanegashima", "Japan", 30.3994, 130.9697, "JAXA Yoshinobu Launch Complex", 14, 34.0),
    PointOfInterest("Los Angeles", "United States", 34.0522, -118.2437, "Southern California aerospace cluster", 24, 48.0),
    PointOfInterest("Rome", "Italy", 41.9028, 12.4964, "Mediterranean / Italian Space Agency", 18, 40.0),
    PointOfInterest("Madrid", "Spain", 40.4168, -3.7038, "NASA Deep Space Network (Robledo)", 18, 42.0),
    PointOfInterest("Seoul", "South Korea", 37.5665, 126.9780, "East Asian technological hub", 26, 52.0),
    PointOfInterest("Mexico City", "Mexico", 19.4326, -99.1332, "Central American metropolis", 22, 46.0),
]


def get_all_pois() -> List[Dict[str, Any]]:
    """Return dictionary representations of all standard POIs."""
    return [
        {
            "name": p.name,
            "country": p.country,
            "lat": p.lat_deg,
            "lon": p.lon_deg,
            "description": p.description,
            "urban_blocks": p.urban_blocks,
            "max_height_km": p.max_height_km,
        }
        for p in MAJOR_POIS
    ]


def find_poi_by_name(name: str) -> Optional[PointOfInterest]:
    """Look up a POI by case-insensitive name match."""
    q = name.strip().lower()
    for p in MAJOR_POIS:
        if p.name.lower() == q or q in p.name.lower():
            return p
    return None


def design_orbit_for_poi(
    lat_deg: float,
    lon_deg: float,
    target_alt_km: float = 550.0,
    ascending: bool = True,
    inclination_deg: Optional[float] = None,
    eccentricity: float = 0.001,
) -> Dict[str, Any]:
    """Compute orbital elements (a, e, i, RAAN, arg_pe, nu) that flies over (lat, lon) at t=0.

    Args:
        lat_deg: Target latitude in degrees [-90, +90].
        lon_deg: Target longitude in degrees [-180, +180].
        target_alt_km: Desired orbital altitude above Earth surface.
        ascending: If True, satellite passes over target on northbound track; else southbound.
        inclination_deg: Optional forced inclination. Must be >= |lat_deg|.
        eccentricity: Orbital eccentricity (default near-circular 0.001).

    Returns:
        Dictionary containing Keplerian elements and initial Cartesian state (r, v).
    """
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    # Minimum inclination must equal or exceed target latitude
    min_inc_deg = abs(lat_deg) + 1.5
    if inclination_deg is not None:
        inc_deg = max(min_inc_deg, float(inclination_deg))
    else:
        # Default inclination comfortably covers latitude with good ground swath
        inc_deg = min(98.0, max(28.5, min_inc_deg))

    inc = math.radians(inc_deg)

    # Argument of latitude u = omega + nu at flyover:
    # sin(lat) = sin(inc) * sin(u)
    sin_u = math.sin(phi) / math.sin(inc)
    sin_u = max(-1.0, min(1.0, sin_u))
    u_base = math.asin(sin_u)

    if ascending:
        u_flyover = u_base if u_base >= 0 else (2.0 * math.pi + u_base)
    else:
        u_flyover = math.pi - u_base

    # Longitude difference delta_lambda from RAAN:
    # tan(lambda - RAAN) = cos(inc) * tan(u)
    cos_u = math.cos(u_flyover)
    sin_delta = math.sin(u_flyover) * math.cos(inc)
    cos_delta = cos_u
    delta_lam = math.atan2(sin_delta, cos_delta)

    # RAAN in [0, 2*pi)
    raan = (lam - delta_lam) % (2.0 * math.pi)

    # For circular/near-circular orbit, choose arg_perigee = 0 -> nu = u
    arg_pe = 0.0
    nu = u_flyover % (2.0 * math.pi)

    a_m = R_EARTH + (target_alt_km * 1000.0)
    oe = OrbitalElements(
        a=a_m,
        e=eccentricity,
        i=inc,
        raan=raan,
        arg_pe=arg_pe,
        nu=nu,
    )

    r_vec, v_vec = coe_to_rv(oe)
    return {
        "elements": {
            "a_km": a_m / 1000.0,
            "alt_km": target_alt_km,
            "eccentricity": eccentricity,
            "inclination_deg": math.degrees(inc),
            "raan_deg": math.degrees(raan),
            "arg_perigee_deg": math.degrees(arg_pe),
            "true_anomaly_deg": math.degrees(nu),
            "period_s": float(oe.period),
        },
        "initial_state": {
            "r": r_vec.tolist(),
            "v": v_vec.tolist(),
        },
        "target_coords": {
            "lat": lat_deg,
            "lon": lon_deg,
        },
    }


def design_orbit_for_poi_cluster(
    poi_names_or_coords: List[Any],
    target_alt_km: float = 550.0,
) -> Dict[str, Any]:
    """Design a resonant repeating ground track orbit covering multiple POIs."""
    coords: List[Tuple[float, float]] = []
    for item in poi_names_or_coords:
        if isinstance(item, str):
            p = find_poi_by_name(item)
            if p:
                coords.append((p.lat_deg, p.lon_deg))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            coords.append((float(item[0]), float(item[1])))

    if not coords:
        # Default to Tokyo if empty
        coords = [(35.6762, 139.6503)]

    # Inclination covers the highest latitude in the cluster
    max_lat = max(abs(c[0]) for c in coords)
    inc_deg = min(98.2, max(35.0, max_lat + 2.5))

    # Target primary (first) POI
    primary = coords[0]
    result = design_orbit_for_poi(
        lat_deg=primary[0],
        lon_deg=primary[1],
        target_alt_km=target_alt_km,
        inclination_deg=inc_deg,
    )
    result["cluster_count"] = len(coords)
    result["cluster_coords"] = coords
    return result


def get_specialized_orbit_presets() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of high-demand specialized astrodynamics orbits."""
    # 1. Geostationary Earth Orbit (GEO)
    geo_r = 42_164_140.0  # meters
    geo_v = circular_velocity(geo_r, MU_EARTH)

    # 2. Molniya Orbit (12-hour, critical inclination 63.435 deg)
    molniya_a = 26_600_000.0  # meters
    molniya_e = 0.72
    molniya_i = math.radians(63.435)

    # 3. Tundra Orbit (24-hour eccentric geosynchronous)
    tundra_a = geo_r
    tundra_e = 0.26
    tundra_i = math.radians(63.435)

    # 4. Sun-Synchronous Orbit (SSO) at 550 km
    sso_alt = 550_000.0
    sso_a = R_EARTH + sso_alt
    sso_i = math.radians(97.6)

    return {
        "GEO": {
            "name": "Geostationary (GEO)",
            "description": "Zero inclination, zero ground track drift, stationary over equator at 35,786 km altitude.",
            "altitude_km": (geo_r - R_EARTH) / 1000.0,
            "eccentricity": 0.0001,
            "inclination_deg": 0.0,
            "raan_deg": 0.0,
            "arg_pe_deg": 0.0,
            "arg_periapsis_deg": 0.0,
            "nu_deg": 0.0,
            "true_anomaly_deg": 0.0,
            "period_s": 86164.1,
        },
        "GSO_INCLINED": {
            "name": "Geosynchronous Inclined (GSO)",
            "description": "24-hour period with 28.5° inclination; traces a daily figure-8 analemma ground track.",
            "altitude_km": (geo_r - R_EARTH) / 1000.0,
            "eccentricity": 0.0001,
            "inclination_deg": 28.5,
            "raan_deg": 45.0,
            "arg_pe_deg": 0.0,
            "arg_periapsis_deg": 0.0,
            "nu_deg": 0.0,
            "true_anomaly_deg": 0.0,
            "period_s": 86164.1,
        },
        "MOLNIYA": {
            "name": "Molniya High-Latitude Dwell",
            "description": "12-hour critical inclination (63.4°) with high eccentricity (e=0.72); dwells 8 hrs over northern targets.",
            "altitude_km": (molniya_a - R_EARTH) / 1000.0,
            "eccentricity": molniya_e,
            "inclination_deg": 63.435,
            "raan_deg": 120.0,
            "arg_pe_deg": 270.0,
            "arg_periapsis_deg": 270.0,
            "nu_deg": 0.0,
            "true_anomaly_deg": 0.0,
            "period_s": 43082.0,
        },
        "TUNDRA": {
            "name": "Tundra 24-Hour Dwell",
            "description": "24-hour eccentric geosynchronous orbit with 63.4° critical inclination providing all-day high-latitude visibility.",
            "altitude_km": (tundra_a - R_EARTH) / 1000.0,
            "eccentricity": tundra_e,
            "inclination_deg": 63.435,
            "raan_deg": 60.0,
            "arg_pe_deg": 270.0,
            "arg_periapsis_deg": 270.0,
            "nu_deg": 0.0,
            "true_anomaly_deg": 0.0,
            "period_s": 86164.1,
        },
        "SSO": {
            "name": "Sun-Synchronous (SSO)",
            "description": "Retrograde 97.6° inclination at 550 km altitude; nodal precession rate exactly matches Earth's annual solar rate.",
            "altitude_km": 550.0,
            "eccentricity": 0.001,
            "inclination_deg": 97.6,
            "raan_deg": 0.0,
            "arg_pe_deg": 0.0,
            "arg_periapsis_deg": 0.0,
            "nu_deg": 0.0,
            "true_anomaly_deg": 0.0,
            "period_s": 5740.0,
        },
        "POLAR_LEO": {
            "name": "Polar Earth Observation (90°)",
            "description": "Pure polar LEO orbit (90.0° inclination) sweeping every latitude for global planetary mapping.",
            "altitude_km": 500.0,
            "eccentricity": 0.001,
            "inclination_deg": 90.0,
            "raan_deg": 30.0,
            "arg_pe_deg": 0.0,
            "arg_periapsis_deg": 0.0,
            "nu_deg": 0.0,
            "true_anomaly_deg": 0.0,
            "period_s": 5680.0,
        },
    }
