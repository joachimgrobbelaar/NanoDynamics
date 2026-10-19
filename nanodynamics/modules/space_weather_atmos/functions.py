"""
nanodynamics/modules/space_weather_atmos/functions.py

Pillar 3: Space Weather, Earth Orientation, Ionosphere & Atmospheric Dynamics.

Provides data loaders and physics-based analysis functions for geomagnetic
indices, solar flux, atmospheric drag, Earth orientation parameters, ionospheric
models, climate reanalysis, severe weather launch criteria, and optical ground
station link-budget estimation.

All units SI/metric: m, km, s, rad, kg, W/m², K, Hz.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nanodynamics.core.constants import (
    C_LIGHT,
    CD_TYPICAL,
    R_EARTH_KM,
    SCALE_HEIGHT_KM,
    SEA_LEVEL_DENSITY_KG_M3,
)
from nanodynamics.data_ingestion.loaders import get_dataset

logger = logging.getLogger("nanodynamics.space_weather_atmos")

__all__ = [
    # 1 – Geomagnetic indices
    "load_space_weather_indices",
    "get_geomagnetic_indices",
    "is_storm_condition",
    # 2 – CelesTrak solar flux
    "load_celestrak_space_weather",
    "interpolate_solar_flux_f107",
    # 3 – Atmospheric drag
    "load_f107_flux",
    "compute_atmospheric_drag_density",
    # 4 – IERS EOP
    "load_iers_eop",
    "get_eop_parameters",
    # 5 – Ionosphere radar
    "load_ionosphere_radar_data",
    "classify_radar_return",
    # 6 – IRI ionospheric maps / GNSS delay
    "load_iri_iono_maps",
    "get_total_electron_content",
    # 7 – CESM atmospheric model
    "load_cesm_atm",
    "slice_stratospheric_winds",
    # 8 – SCREAM climate data
    "load_scream_climate_data",
    "extract_subdecadal_reentry_densities",
    # 9 – WRF severe weather / launch commit
    "load_wrf_severe_weather",
    "check_launch_commit_criteria",
    # 10 – Optical ground station attenuation
    "load_cloud_temp_data",
    "predict_optical_ground_station_attenuation",
]


# ============================================================================
# Helper utilities
# ============================================================================

def _parse_utc(timestamp_str: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime.

    Supports formats with or without trailing 'Z', fractional seconds, and
    explicit ``+00:00`` offsets.
    """
    ts = timestamp_str.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except ValueError:
        # Fallback for dates without time component
        return datetime.strptime(ts, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _nearest_row(ds, col: str, target_val: float, *, n: int = 1):
    """Return the row(s) whose *col* value is closest to *target_val*.

    Parameters
    ----------
    ds : datasets.Dataset
        HuggingFace dataset (non-streaming).
    col : str
        Column name containing numeric values.
    target_val : float
        Target value to match.
    n : int
        Number of nearest rows to return (default 1).
    """
    values = np.asarray(ds[col], dtype=np.float64)
    idx = np.argsort(np.abs(values - target_val))[:n]
    return ds.select(idx.tolist())


# ============================================================================
# 1. Space-Weather Geomagnetic Indices  (Kp, Ap, Dst)
# ============================================================================

def load_space_weather_indices(**kwargs):
    """Load the space-weather-indices dataset from the registry.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
        The ``space_weather_indices`` dataset.
    """
    logger.info("Loading space-weather-indices dataset …")
    return get_dataset("space_weather_indices", **kwargs)


def get_geomagnetic_indices(
    ds,
    timestamp_str: str,
) -> Dict[str, Optional[float]]:
    """Look up Kp, Ap, and Dst indices for the closest timestamp in *ds*.

    Parameters
    ----------
    ds : datasets.Dataset
        A loaded ``space_weather_indices`` dataset.
    timestamp_str : str
        ISO-8601 UTC timestamp (e.g. ``"2023-09-15T12:00:00Z"``).

    Returns
    -------
    dict
        ``{"kp": float|None, "ap": float|None, "dst": float|None,
           "timestamp_matched": str}``.
    """
    target_dt = _parse_utc(timestamp_str)
    target_epoch = target_dt.timestamp()

    columns = ds.column_names
    # Attempt common column-name variants
    time_col = next(
        (c for c in columns if c.lower() in {"timestamp", "datetime", "date", "time", "epoch"}),
        columns[0],
    )

    best_row: Optional[Dict[str, Any]] = None
    best_delta = float("inf")

    for row in ds:
        raw_t = row[time_col]
        try:
            row_epoch = _parse_utc(str(raw_t)).timestamp()
        except Exception:
            try:
                row_epoch = float(raw_t)
            except (TypeError, ValueError):
                continue
        delta = abs(row_epoch - target_epoch)
        if delta < best_delta:
            best_delta = delta
            best_row = row

    if best_row is None:
        logger.warning("No matching row found for timestamp %s", timestamp_str)
        return {"kp": None, "ap": None, "dst": None, "timestamp_matched": None}

    def _get(row: Dict, *candidates: str) -> Optional[float]:
        for key in candidates:
            for col in row:
                if col.lower() == key.lower():
                    val = row[col]
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return None

    return {
        "kp": _get(best_row, "kp", "kp_index", "Kp"),
        "ap": _get(best_row, "ap", "ap_index", "Ap"),
        "dst": _get(best_row, "dst", "dst_index", "Dst"),
        "timestamp_matched": str(best_row.get(time_col)),
    }


def is_storm_condition(kp_value: float, threshold_kp: float = 5.0) -> bool:
    """Return ``True`` if the given Kp index indicates a geomagnetic storm.

    NOAA classifies Kp ≥ 5 as storm-level (G1+).

    Parameters
    ----------
    kp_value : float
        Measured Kp index (0–9 scale).
    threshold_kp : float
        Minimum Kp to flag as storm (default 5.0, G1 threshold).
    """
    return float(kp_value) >= threshold_kp


# ============================================================================
# 2. CelesTrak Space-Weather / F10.7 Solar Flux Interpolation
# ============================================================================

def load_celestrak_space_weather(**kwargs):
    """Load CelesTrak space-weather dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
    """
    logger.info("Loading CelesTrak space-weather dataset …")
    return get_dataset("celestrak_space_weather", **kwargs)


def interpolate_solar_flux_f107(
    ds,
    datetime_utc_str: str,
) -> Dict[str, float]:
    """Interpolate the daily F10.7 solar radio flux and 81-day centered average.

    Uses linear interpolation between the two bounding daily records.

    Parameters
    ----------
    ds : datasets.Dataset
        CelesTrak space-weather dataset with daily F10.7 values.
    datetime_utc_str : str
        ISO-8601 UTC timestamp to interpolate for.

    Returns
    -------
    dict
        ``{"f107_daily": float, "f107_81day_avg": float}``
    """
    target_dt = _parse_utc(datetime_utc_str)
    target_epoch = target_dt.timestamp()

    columns = ds.column_names
    time_col = next(
        (c for c in columns if c.lower() in {"date", "datetime", "timestamp", "time"}),
        columns[0],
    )

    # Collect epochs and flux values
    epochs: List[float] = []
    f107_daily: List[float] = []
    f107_81: List[float] = []

    f107_col = next((c for c in columns if "f10" in c.lower() and "81" not in c.lower()), None)
    f107_81_col = next((c for c in columns if "f10" in c.lower() and "81" in c.lower()), None)

    if f107_col is None:
        # Fallback: try common names
        f107_col = next((c for c in columns if "flux" in c.lower()), columns[1] if len(columns) > 1 else None)
    if f107_81_col is None:
        f107_81_col = f107_col  # degrade gracefully

    for row in ds:
        try:
            ep = _parse_utc(str(row[time_col])).timestamp()
        except Exception:
            continue
        try:
            f_daily = float(row[f107_col])
        except (TypeError, ValueError, KeyError):
            continue
        try:
            f_81 = float(row[f107_81_col]) if f107_81_col != f107_col else f_daily
        except (TypeError, ValueError, KeyError):
            f_81 = f_daily

        epochs.append(ep)
        f107_daily.append(f_daily)
        f107_81.append(f_81)

    if not epochs:
        logger.warning("No valid F10.7 records found in dataset.")
        return {"f107_daily": 150.0, "f107_81day_avg": 150.0}

    epochs_arr = np.asarray(epochs)
    sort_idx = np.argsort(epochs_arr)
    epochs_arr = epochs_arr[sort_idx]
    f107_daily_arr = np.asarray(f107_daily)[sort_idx]
    f107_81_arr = np.asarray(f107_81)[sort_idx]

    interp_daily = float(np.interp(target_epoch, epochs_arr, f107_daily_arr))
    interp_81 = float(np.interp(target_epoch, epochs_arr, f107_81_arr))

    return {"f107_daily": interp_daily, "f107_81day_avg": interp_81}


# ============================================================================
# 3. Atmospheric Drag Density  (Simplified NRLMSISE-00)
# ============================================================================

def load_f107_flux(**kwargs):
    """Load the F10.7 solar-flux dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
    """
    logger.info("Loading F10.7 solar-flux dataset …")
    return get_dataset("f107_solar_flux", **kwargs)


def compute_atmospheric_drag_density(
    alt_km: float,
    f107: float,
    ap: float,
    *,
    cd: float = CD_TYPICAL,
    area_to_mass: float = 0.01,
    velocity_ms: float = 7_600.0,
    rho0: float = SEA_LEVEL_DENSITY_KG_M3,
    scale_height_km: float = SCALE_HEIGHT_KM,
) -> Dict[str, float]:
    """Compute thermospheric mass density and drag acceleration.

    Uses a simplified NRLMSISE-00-style exponential atmosphere with F10.7
    and Ap corrections:

    .. math::

        \\rho = \\rho_0 \\exp(-h / H)
                \\bigl(1 + 0.0013 (F_{10.7} - 150)\\bigr)
                \\bigl(1 + 0.005 \\, a_p\\bigr)

    Drag acceleration (opposing velocity):

    .. math::

        a_{\\mathrm{drag}} = -\\tfrac{1}{2}\\,\\rho\\,C_D\\,\\frac{A}{m}\\,v^2

    Parameters
    ----------
    alt_km : float
        Geodetic altitude [km].
    f107 : float
        Daily F10.7 solar radio flux [SFU, 10⁻²² W m⁻² Hz⁻¹].
    ap : float
        3-hour Ap geomagnetic index.
    cd : float
        Drag coefficient (default 2.2).
    area_to_mass : float
        Cross-sectional area / mass ratio [m²/kg] (default 0.01).
    velocity_ms : float
        Spacecraft velocity [m/s] (default 7 600 for LEO).
    rho0 : float
        Sea-level reference density [kg/m³].
    scale_height_km : float
        Atmospheric scale height [km].

    Returns
    -------
    dict
        ``{"density_kg_m3": float, "drag_accel_m_s2": float}``
    """
    h = float(alt_km)
    H = float(scale_height_km)

    # Exponential base density: standard tropospheric/stratospheric for h <= 100 km,
    # thermospheric regime (H ~ 50 km) for LEO altitudes (h > 100 km).
    if h > 100.0:
        H_eff = 50.0 + 0.05 * (h - 200.0)
        rho = 2.8e-12 * math.exp(-(h - 400.0) / H_eff)
    else:
        rho = rho0 * math.exp(-h / H)

    # Solar-flux correction (centred on F10.7 = 150 SFU baseline)
    rho *= 1.0 + 0.0013 * (f107 - 150.0)

    # Geomagnetic-activity correction
    rho *= 1.0 + 0.005 * ap

    # Drag acceleration magnitude (negative = decelerating)
    a_drag = -0.5 * rho * cd * area_to_mass * velocity_ms ** 2

    logger.debug(
        "alt=%.1f km  F10.7=%.1f  Ap=%.1f → ρ=%.4e kg/m³  a_drag=%.4e m/s²",
        alt_km, f107, ap, rho, a_drag,
    )

    return {"density_kg_m3": rho, "drag_accel_m_s2": a_drag}


# ============================================================================
# 4. IERS Earth Orientation Parameters
# ============================================================================

def load_iers_eop(**kwargs):
    """Load the IERS Earth Orientation Parameters dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
    """
    logger.info("Loading IERS EOP dataset …")
    return get_dataset("iers_eop", **kwargs)


def get_eop_parameters(
    ds,
    mjd: float,
) -> Dict[str, Optional[float]]:
    """Return Earth Orientation Parameters for a given Modified Julian Date.

    Performs nearest-neighbour lookup on the MJD column and returns:

    - ``xp`` – x pole coordinate [arcsec]
    - ``yp`` – y pole coordinate [arcsec]
    - ``ut1_utc`` – UT1-UTC [s]
    - ``dX`` – celestial pole offset dX [mas]
    - ``dY`` – celestial pole offset dY [mas]

    Parameters
    ----------
    ds : datasets.Dataset
        IERS EOP dataset.
    mjd : float
        Modified Julian Date for the query epoch.

    Returns
    -------
    dict
        EOP values keyed as ``xp, yp, ut1_utc, dX, dY``.
    """
    columns = ds.column_names
    mjd_col = next(
        (c for c in columns if c.lower() in {"mjd", "modified_julian_date", "mjd_utc"}),
        columns[0],
    )

    nearest = _nearest_row(ds, mjd_col, mjd, n=1)
    if len(nearest) == 0:
        logger.warning("No EOP row found near MJD %.2f", mjd)
        return {"xp": None, "yp": None, "ut1_utc": None, "dX": None, "dY": None}

    row = nearest[0]

    def _col(row_dict: Dict, *candidates: str) -> Optional[float]:
        for cand in candidates:
            for key in row_dict:
                if key.lower() == cand.lower():
                    try:
                        return float(row_dict[key])
                    except (TypeError, ValueError):
                        pass
        return None

    return {
        "xp": _col(row, "xp", "x_pole", "PM_x", "pmx"),
        "yp": _col(row, "yp", "y_pole", "PM_y", "pmy"),
        "ut1_utc": _col(row, "ut1_utc", "UT1-UTC", "ut1_minus_utc", "dut1"),
        "dX": _col(row, "dX", "dx", "dPsi", "dpsi"),
        "dY": _col(row, "dY", "dy", "dEps", "deps"),
    }


# ============================================================================
# 5. Ionosphere Radar Data Classification
# ============================================================================

def load_ionosphere_radar_data(**kwargs):
    """Load the ionosphere-radar dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
    """
    logger.info("Loading ionosphere-radar dataset …")
    return get_dataset("ionosphere_radar", **kwargs)


def classify_radar_return(features_dict: Dict[str, float]) -> Dict[str, Any]:
    """Classify an ionospheric radar return as *good* or *bad*.

    Uses simple threshold-based rules on common radar pulse features:

    * ``pulse_mean`` > 0.5 → good signal
    * ``pulse_std``  < 0.3 → stable return → good
    * ``snr``        > 1.0 → signal above noise → good

    If none of the known features are present the classifier defaults to
    scoring by the mean of all provided numeric values (> 0.5 → good).

    Parameters
    ----------
    features_dict : dict
        Feature name → numeric value mapping (e.g. radar attribute columns).

    Returns
    -------
    dict
        ``{"classification": "good"|"bad", "confidence": float,
           "rules_triggered": list[str]}``
    """
    rules_triggered: List[str] = []
    score = 0.0
    n_rules = 0

    # Rule 1 – pulse mean amplitude
    pulse_mean = features_dict.get("pulse_mean")
    if pulse_mean is not None:
        n_rules += 1
        if float(pulse_mean) > 0.5:
            score += 1.0
            rules_triggered.append("pulse_mean > 0.5")

    # Rule 2 – pulse standard deviation (stability)
    pulse_std = features_dict.get("pulse_std")
    if pulse_std is not None:
        n_rules += 1
        if float(pulse_std) < 0.3:
            score += 1.0
            rules_triggered.append("pulse_std < 0.3")

    # Rule 3 – signal-to-noise ratio
    snr = features_dict.get("snr")
    if snr is not None:
        n_rules += 1
        if float(snr) > 1.0:
            score += 1.0
            rules_triggered.append("snr > 1.0")

    # Fallback: mean of all numeric values
    if n_rules == 0:
        numeric_vals = [
            float(v) for v in features_dict.values()
            if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).lstrip("-").isdigit())
        ]
        if numeric_vals:
            avg = np.mean(numeric_vals)
            n_rules = 1
            score = 1.0 if avg > 0.5 else 0.0
            rules_triggered.append(f"fallback_mean={avg:.4f}")

    confidence = score / max(n_rules, 1)
    classification = "good" if confidence >= 0.5 else "bad"

    return {
        "classification": classification,
        "confidence": round(confidence, 4),
        "rules_triggered": rules_triggered,
    }


# ============================================================================
# 6. IRI Ionospheric Maps / GNSS Delay
# ============================================================================

def load_iri_iono_maps(**kwargs):
    """Load the IRI ionospheric-maps dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
    """
    logger.info("Loading IRI ionospheric-maps dataset …")
    return get_dataset("iri_iono_maps", **kwargs)


def get_total_electron_content(
    ds,
    lat: float,
    lon: float,
    utc_time_str: str,
    *,
    gnss_freq_hz: float = 1.57542e9,
) -> Dict[str, float]:
    """Look up vertical TEC and compute the associated GNSS signal delay.

    The ionospheric group delay for a single-frequency GNSS receiver is:

    .. math::

        \\Delta\\tau = \\frac{40.3 \\times \\mathrm{TEC}}{c \\, f^2}

    where *TEC* is in electrons/m², *c* is the speed of light [m/s], and
    *f* is the carrier frequency [Hz].

    Parameters
    ----------
    ds : datasets.Dataset
        IRI ionospheric-maps dataset.
    lat, lon : float
        Geographic latitude / longitude [deg] of the pierce point.
    utc_time_str : str
        ISO-8601 UTC timestamp.
    gnss_freq_hz : float
        GNSS carrier frequency [Hz] (default L1 = 1575.42 MHz).

    Returns
    -------
    dict
        ``{"vtec_tecu": float, "gnss_delay_s": float, "gnss_delay_m": float}``
    """
    target_dt = _parse_utc(utc_time_str)
    columns = ds.column_names

    # Identify coordinate / TEC columns
    lat_col = next((c for c in columns if c.lower() in {"lat", "latitude", "glat"}), None)
    lon_col = next((c for c in columns if c.lower() in {"lon", "longitude", "glon", "long"}), None)
    tec_col = next((c for c in columns if "tec" in c.lower()), None)
    time_col = next(
        (c for c in columns if c.lower() in {"time", "datetime", "timestamp", "date", "utc"}),
        None,
    )

    if lat_col is None or lon_col is None or tec_col is None:
        logger.warning(
            "Could not identify required columns (lat/lon/tec) in dataset "
            "columns: %s.  Returning default TEC=30 TECU.", columns,
        )
        vtec_tecu = 30.0
    else:
        # Filter by time first if possible, then pick spatial nearest
        best_tec: Optional[float] = None
        best_dist = float("inf")

        for row in ds:
            # Optional temporal filter
            if time_col is not None:
                try:
                    row_dt = _parse_utc(str(row[time_col]))
                    if abs((row_dt - target_dt).total_seconds()) > 3600:
                        continue
                except Exception:
                    pass

            try:
                rlat = float(row[lat_col])
                rlon = float(row[lon_col])
            except (TypeError, ValueError):
                continue

            dist = (rlat - lat) ** 2 + (rlon - lon) ** 2
            if dist < best_dist:
                best_dist = dist
                try:
                    best_tec = float(row[tec_col])
                except (TypeError, ValueError):
                    pass

        vtec_tecu = best_tec if best_tec is not None else 30.0

    # TEC in TECU (1 TECU = 1e16 el/m²)
    tec_el_m2 = vtec_tecu * 1.0e16

    # Ionospheric delay: Δτ = 40.3 * TEC / (c * f²)
    delay_s = 40.3 * tec_el_m2 / (C_LIGHT * gnss_freq_hz ** 2)
    delay_m = delay_s * C_LIGHT

    return {
        "vtec_tecu": vtec_tecu,
        "gnss_delay_s": delay_s,
        "gnss_delay_m": delay_m,
    }


# ============================================================================
# 7. CESM Atmospheric Model
# ============================================================================

def load_cesm_atm(**kwargs):
    """Load the CESM-ATM reanalysis dataset (large, streams by default).

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
    """
    logger.info("Loading CESM-ATM dataset (streaming) …")
    return get_dataset("cesm_atm", **kwargs)


def slice_stratospheric_winds(
    ds,
    pressure_level_hpa: float,
) -> Dict[str, Any]:
    """Extract zonal (U) and meridional (V) wind fields at a pressure level.

    Searches the dataset for rows matching the requested pressure level
    (within ±1 hPa) and returns arrays of wind components.

    Parameters
    ----------
    ds : datasets.Dataset
        CESM-ATM dataset.
    pressure_level_hpa : float
        Target pressure level [hPa].

    Returns
    -------
    dict
        ``{"pressure_hpa": float, "U_wind": ndarray, "V_wind": ndarray,
           "n_records": int}``
    """
    columns = ds.column_names
    pres_col = next(
        (c for c in columns if c.lower() in {"pressure", "lev", "level", "plev", "pressure_hpa"}),
        None,
    )
    u_col = next((c for c in columns if c.upper() in {"U", "U_WIND", "UWIND", "U_GRD"}), None)
    v_col = next((c for c in columns if c.upper() in {"V", "V_WIND", "VWIND", "V_GRD"}), None)

    u_vals: List[float] = []
    v_vals: List[float] = []

    tolerance_hpa = 1.0

    for row in ds:
        if pres_col is not None:
            try:
                plev = float(row[pres_col])
            except (TypeError, ValueError):
                continue
            if abs(plev - pressure_level_hpa) > tolerance_hpa:
                continue

        try:
            u_val = float(row[u_col]) if u_col else 0.0
            v_val = float(row[v_col]) if v_col else 0.0
        except (TypeError, ValueError):
            continue

        u_vals.append(u_val)
        v_vals.append(v_val)

    logger.info(
        "Extracted %d records at %.1f hPa pressure level.", len(u_vals), pressure_level_hpa,
    )

    return {
        "pressure_hpa": pressure_level_hpa,
        "U_wind": np.asarray(u_vals),
        "V_wind": np.asarray(v_vals),
        "n_records": len(u_vals),
    }


# ============================================================================
# 8. SCREAM Climate Data – Sub-decadal Density Trends
# ============================================================================

def load_scream_climate_data(**kwargs):
    """Load the NVIDIA STRATA-SCREAM sub-decadal climate dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
    """
    logger.info("Loading STRATA-SCREAM climate dataset (streaming) …")
    return get_dataset("strata_scream", **kwargs)


def extract_subdecadal_reentry_densities(
    ds,
    year_start: int,
    year_end: int,
) -> Dict[str, Any]:
    """Extract atmospheric density trends for reentry corridor modelling.

    Filters the SCREAM dataset for records within ``[year_start, year_end]``
    and collects density/temperature fields relevant to spacecraft reentry
    heating and trajectory analysis.

    Parameters
    ----------
    ds : datasets.Dataset | datasets.IterableDataset
        STRATA-SCREAM dataset.
    year_start, year_end : int
        Inclusive year range.

    Returns
    -------
    dict
        ``{"years": ndarray, "density_profiles": ndarray,
           "temperature_profiles": ndarray, "n_records": int}``
    """
    columns = ds.column_names
    year_col = next((c for c in columns if c.lower() in {"year", "time", "date"}), None)
    dens_col = next(
        (c for c in columns if "dens" in c.lower() or "rho" in c.lower()), None,
    )
    temp_col = next(
        (c for c in columns if "temp" in c.lower() or c.upper() == "T"), None,
    )

    years: List[float] = []
    densities: List[float] = []
    temperatures: List[float] = []

    for row in ds:
        # Extract year
        yr: Optional[float] = None
        if year_col is not None:
            raw = row[year_col]
            try:
                yr = float(raw)
            except (TypeError, ValueError):
                try:
                    yr = float(_parse_utc(str(raw)).year)
                except Exception:
                    continue

        if yr is not None and not (year_start <= yr <= year_end):
            continue

        try:
            d = float(row[dens_col]) if dens_col else float("nan")
        except (TypeError, ValueError):
            d = float("nan")
        try:
            t = float(row[temp_col]) if temp_col else float("nan")
        except (TypeError, ValueError):
            t = float("nan")

        years.append(yr if yr is not None else float("nan"))
        densities.append(d)
        temperatures.append(t)

    logger.info(
        "Extracted %d density records for years %d–%d.", len(years), year_start, year_end,
    )

    return {
        "years": np.asarray(years),
        "density_profiles": np.asarray(densities),
        "temperature_profiles": np.asarray(temperatures),
        "n_records": len(years),
    }


# ============================================================================
# 9. WRF Severe Weather – Launch Commit Criteria
# ============================================================================

def load_wrf_severe_weather(**kwargs):
    """Load the WRF 250 m severe-weather dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
    """
    logger.info("Loading WRF severe-weather dataset (streaming) …")
    return get_dataset("wrf_severe_weather", **kwargs)


# Range safety thresholds (USAF 45th Weather Squadron / USSF inspired)
_LCC_MAX_WIND_MS = 25.0          # Surface / upper-level wind limit [m/s]
_LCC_MAX_LIGHTNING_INDEX = 0.1   # Lightning Potential Index threshold
_LCC_MAX_SHEAR_MS = 20.0        # Convective wind-shear limit [m/s]


def check_launch_commit_criteria(
    wind_speed_ms: float,
    lightning_index: float,
    convective_shear: float,
    *,
    max_wind_ms: float = _LCC_MAX_WIND_MS,
    max_lightning_index: float = _LCC_MAX_LIGHTNING_INDEX,
    max_shear_ms: float = _LCC_MAX_SHEAR_MS,
) -> Dict[str, Any]:
    """Evaluate weather Launch Commit Criteria (LCC) for range safety.

    Based on simplified USAF 45 WS / USSF rules:

    * **Wind**: surface / upper-level wind speed < 25 m/s
    * **Lightning Potential Index (LPI)**: < 0.1
    * **Convective shear**: < 20 m/s

    Parameters
    ----------
    wind_speed_ms : float
        Peak surface or upper-level wind speed [m/s].
    lightning_index : float
        Lightning Potential Index (dimensionless, 0–1 scale).
    convective_shear : float
        Maximum convective wind shear [m/s].
    max_wind_ms, max_lightning_index, max_shear_ms : float
        Override thresholds if non-standard range safety rules apply.

    Returns
    -------
    dict
        ``{"go": bool, "violations": list[str], "details": dict}``
    """
    violations: List[str] = []

    wind_ok = wind_speed_ms < max_wind_ms
    if not wind_ok:
        violations.append(
            f"Wind {wind_speed_ms:.1f} m/s ≥ limit {max_wind_ms:.1f} m/s"
        )

    lpi_ok = lightning_index < max_lightning_index
    if not lpi_ok:
        violations.append(
            f"LPI {lightning_index:.4f} ≥ limit {max_lightning_index:.4f}"
        )

    shear_ok = convective_shear < max_shear_ms
    if not shear_ok:
        violations.append(
            f"Shear {convective_shear:.1f} m/s ≥ limit {max_shear_ms:.1f} m/s"
        )

    go = wind_ok and lpi_ok and shear_ok

    return {
        "go": go,
        "violations": violations,
        "details": {
            "wind_speed_ms": wind_speed_ms,
            "wind_limit_ms": max_wind_ms,
            "wind_ok": wind_ok,
            "lightning_index": lightning_index,
            "lightning_limit": max_lightning_index,
            "lightning_ok": lpi_ok,
            "convective_shear_ms": convective_shear,
            "shear_limit_ms": max_shear_ms,
            "shear_ok": shear_ok,
        },
    }


# ============================================================================
# 10. Optical Ground Station Attenuation
# ============================================================================

def load_cloud_temp_data(**kwargs):
    """Load the cloud-vs-temperature dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
    """
    logger.info("Loading cloud-vs-temperature dataset …")
    return get_dataset("cloud_temp_data", **kwargs)


def predict_optical_ground_station_attenuation(
    cloud_fraction: float,
    temp_k: float,
) -> Dict[str, float]:
    """Estimate free-space optical link attenuation for a ground station.

    A semi-empirical model combining cloud-cover extinction and a
    temperature-dependent atmospheric absorption term:

    .. math::

        A_{\\mathrm{dB}} = 10\\,\\log_{10}\\!\\left(
            \\frac{1}{1 - 0.8\\,f_c}\\right)
            + 0.01\\,(T - 273)

    where *f_c* is the cloud fraction [0, 1] and *T* is the ambient
    temperature [K].

    Parameters
    ----------
    cloud_fraction : float
        Fractional cloud cover at the ground station (0 = clear, 1 = overcast).
        Clamped to [0, 0.999] to prevent singularity.
    temp_k : float
        Ambient temperature at the ground station [K].

    Returns
    -------
    dict
        ``{"attenuation_dB": float, "cloud_component_dB": float,
           "thermal_component_dB": float, "transmittance_linear": float}``
    """
    # Clamp cloud_fraction to avoid log(0)
    fc = max(0.0, min(float(cloud_fraction), 0.999))
    t = float(temp_k)

    # Cloud extinction component
    cloud_dB = 10.0 * math.log10(1.0 / (1.0 - 0.8 * fc))

    # Temperature-dependent atmospheric absorption
    thermal_dB = 0.01 * (t - 273.0)

    total_dB = cloud_dB + thermal_dB

    # Linear transmittance for link-budget calculations
    transmittance = 10.0 ** (-total_dB / 10.0)

    return {
        "attenuation_dB": round(total_dB, 6),
        "cloud_component_dB": round(cloud_dB, 6),
        "thermal_component_dB": round(thermal_dB, 6),
        "transmittance_linear": round(transmittance, 8),
    }
