"""Coordinate transformation utilities for LEO satellite simulation.

Supports transformations between:
- ECI (Earth-Centered Inertial / GCRF-like)
- ECEF (Earth-Centered Earth-Fixed)
- Geodetic Coordinates (Spherical / WGS84 Latitude, Longitude, Altitude)
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

# WGS84 / Reference Constants
WGS84_A = 6378137.0         # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # Semi-minor axis (meters)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # First eccentricity squared
OMEGA_EARTH = 7.2921150e-5  # Earth rotation rate (rad/s)


def eci_to_ecef(
    r_eci: np.ndarray,
    t_seconds: Union[float, np.ndarray],
    omega_earth: float = OMEGA_EARTH,
    theta_0: float = 0.0,
) -> np.ndarray:
    """Transform Cartesian position vectors from ECI to ECEF frame.
    
    Args:
        r_eci: Shape (3,) or (N, 3) array of ECI position vectors in meters.
        t_seconds: Epoch time in seconds (float or shape (N,) array).
        omega_earth: Angular rotation velocity of the central body in rad/s.
        theta_0: Initial prime meridian angle at t=0 in radians.
        
    Returns:
        r_ecef: Shape (3,) or (N, 3) array in meters.
    """
    r_eci = np.asarray(r_eci, dtype=np.float64)
    is_1d = r_eci.ndim == 1
    if is_1d:
        r_eci = r_eci.reshape(1, 3)
        t_seconds = np.array([t_seconds], dtype=np.float64)
    else:
        t_seconds = np.asarray(t_seconds, dtype=np.float64)

    theta = theta_0 + omega_earth * t_seconds
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x = r_eci[:, 0] * cos_t + r_eci[:, 1] * sin_t
    y = -r_eci[:, 0] * sin_t + r_eci[:, 1] * cos_t
    z = r_eci[:, 2]

    r_ecef = np.column_stack((x, y, z))
    return r_ecef[0] if is_1d else r_ecef


def ecef_to_geodetic(
    r_ecef: np.ndarray,
    radius_m: float = WGS84_A,
    ellipsoidal: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform Cartesian ECEF vectors to Geodetic Latitude, Longitude, and Altitude.
    
    Args:
        r_ecef: Shape (3,) or (N, 3) position vector in meters.
        radius_m: Reference equatorial radius in meters.
        ellipsoidal: If True, uses WGS84 ellipsoidal model (Bowring's algorithm);
                     if False, uses spherical body approximation.
                     
    Returns:
        lat_deg: Geodetic latitude in degrees [-90.0, +90.0]
        lon_deg: Geodetic longitude in degrees [-180.0, +180.0]
        alt_km: Altitude above reference surface in kilometers
    """
    r_ecef = np.asarray(r_ecef, dtype=np.float64)
    is_1d = r_ecef.ndim == 1
    if is_1d:
        r_ecef = r_ecef.reshape(1, 3)

    x = r_ecef[:, 0]
    y = r_ecef[:, 1]
    z = r_ecef[:, 2]

    lon_rad = np.arctan2(y, x)
    lon_deg = np.degrees(lon_rad)
    # Normalize longitude to [-180, 180]
    lon_deg = (lon_deg + 180.0) % 360.0 - 180.0

    p = np.sqrt(x**2 + y**2)

    if not ellipsoidal or radius_m != WGS84_A:
        # Spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        lat_rad = np.arcsin(np.clip(z / np.maximum(r, 1e-12), -1.0, 1.0))
        lat_deg = np.degrees(lat_rad)
        alt_km = (r - radius_m) / 1000.0
    else:
        # Bowring's method for WGS84 ellipsoid
        a = WGS84_A
        b = WGS84_B
        e2 = WGS84_E2
        ep2 = (a**2 - b**2) / (b**2)
        
        theta = np.arctan2(z * a, p * b)
        lat_rad = np.arctan2(
            z + ep2 * b * np.sin(theta)**3,
            p - e2 * a * np.cos(theta)**3
        )
        lat_deg = np.degrees(lat_rad)
        
        n_rad = a / np.sqrt(1.0 - e2 * np.sin(lat_rad)**2)
        alt_m = p / np.cos(lat_rad) - n_rad
        alt_km = alt_m / 1000.0

    if is_1d:
        return float(lat_deg[0]), float(lon_deg[0]), float(alt_km[0])
    return lat_deg, lon_deg, alt_km


def eci_to_geodetic(
    r_eci: np.ndarray,
    t_seconds: Union[float, np.ndarray],
    radius_m: float = WGS84_A,
    omega_earth: float = OMEGA_EARTH,
    ellipsoidal: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct conversion from ECI Cartesian to Geodetic Lat/Lon/Alt."""
    r_ecef = eci_to_ecef(r_eci, t_seconds, omega_earth=omega_earth)
    return ecef_to_geodetic(r_ecef, radius_m=radius_m, ellipsoidal=ellipsoidal)


def extract_trajectory_key_events(
    t_arr: np.ndarray,
    r_eci_arr: np.ndarray,
    radius_m: float = WGS84_A,
    omega_earth: float = OMEGA_EARTH,
    reentry_alt_km: float = 120.0,
) -> Dict[str, Optional[Dict[str, float]]]:
    """Extract initial, re-entry (120 km), and impact (0 km) coordinates and epochs.
    
    Returns:
        dict with keys: 'initial', 'reentry', 'impact' containing dicts with:
        {'time_s': float, 'lat_deg': float, 'lon_deg': float, 'alt_km': float}
    """
    if len(t_arr) == 0:
        return {"initial": None, "reentry": None, "impact": None}

    lat, lon, alt = eci_to_geodetic(r_eci_arr, t_arr, radius_m=radius_m, omega_earth=omega_earth)

    initial_coords = {
        "time_s": float(t_arr[0]),
        "lat_deg": round(float(lat[0]), 6),
        "lon_deg": round(float(lon[0]), 6),
        "alt_km": round(float(alt[0]), 6),
    }

    reentry_coords = None
    impact_coords = None

    # Find first crossing below reentry altitude (e.g. 120 km)
    reentry_indices = np.where(alt <= reentry_alt_km)[0]
    if len(reentry_indices) > 0:
        idx_r = reentry_indices[0]
        reentry_coords = {
            "time_s": float(t_arr[idx_r]),
            "lat_deg": round(float(lat[idx_r]), 6),
            "lon_deg": round(float(lon[idx_r]), 6),
            "alt_km": round(float(alt[idx_r]), 6),
        }

    # Find ground collision / impact (alt <= 0.05 or end of terminal trajectory)
    impact_indices = np.where(alt <= 0.05)[0]
    if len(impact_indices) > 0:
        idx_i = impact_indices[0]
        impact_coords = {
            "time_s": float(t_arr[idx_i]),
            "lat_deg": round(float(lat[idx_i]), 6),
            "lon_deg": round(float(lon[idx_i]), 6),
            "alt_km": max(0.0, round(float(alt[idx_i]), 6)),
        }

    return {
        "initial": initial_coords,
        "reentry": reentry_coords,
        "impact": impact_coords,
    }
