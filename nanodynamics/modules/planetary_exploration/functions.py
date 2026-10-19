"""
nanodynamics/modules/planetary_exploration/functions.py

Pillar 4 — Deep Space, Planetary Environments & Lunar Missions.

Functions for:
  • MAVEN in-situ ion escape analysis at Mars
  • OpenMars reanalysis profiles for EDL simulation
  • Venus Express cloud albedo extraction
  • Artemis II circumlunar free-return trajectory (patched-conics)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Union

import numpy as np

from nanodynamics.core.constants import *
from nanodynamics.data_ingestion.loaders import get_dataset

logger = logging.getLogger("nanodynamics.planetary_exploration")

__all__ = [
    "load_maven_insitu",
    "calculate_mars_ion_loss_rate",
    "load_openmars_reanalysis",
    "get_martian_entry_profile",
    "load_venus_express_observations",
    "extract_venus_cloud_albedo",
    "load_artemis_ii_mission_profile",
    "simulate_free_return_trajectory",
]

# ── Physical constants for ion-escape modelling ─────────────────────────────
K_BOLTZMANN = 1.380649e-23          # Boltzmann constant [J/K]
M_PROTON = 1.67262192e-27           # Proton mass [kg]
G_MARS_SURFACE = 3.72076            # Mars surface gravity [m/s²]

# Ion masses in kg (atomic mass units × proton mass)
_ION_MASS = {
    "O+": 16.0 * M_PROTON,
    "O2+": 32.0 * M_PROTON,
    "CO2+": 44.0 * M_PROTON,
}

# Characteristic exospheric temperatures per ion species [K]
# (representative values from MAVEN STATIC/NGIMS observations)
_DEFAULT_EXOBASE_TEMP_K = {
    "O+": 300.0,
    "O2+": 250.0,
    "CO2+": 220.0,
}


# ============================================================================
# 1. MAVEN In-Situ Ion Escape Analysis
# ============================================================================

def load_maven_insitu(**kwargs: Any):
    """Load the NASA MAVEN key-parameter in-situ dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
        Hugging Face dataset containing MAVEN ion/neutral density measurements,
        magnetic field, and plasma data keyed by altitude.
    """
    logger.info("Loading MAVEN in-situ key-parameter dataset")
    return get_dataset("maven_insitu", **kwargs)


def calculate_mars_ion_loss_rate(
    ds,
    altitude_km: float,
    *,
    species: Optional[list[str]] = None,
    exobase_altitude_km: float = 200.0,
    cross_section_area_km2: float = 6.0e6,
) -> Dict[str, Any]:
    """Estimate Martian ion loss rate at a given altitude.

    Uses a simplified Jeans escape formulation for each ion species.
    The thermal escape flux (Φ_jeans) for a species of mass *m* at
    temperature *T* and exobase radius *r_exo* is:

        λ = G·M_mars·m / (k_B·T·r_exo)
        v_thermal = sqrt(2·k_B·T / m)
        Φ = n · v_thermal / (2·√π) · (1 + λ) · exp(-λ)   [ions m⁻² s⁻¹]

    Total loss rate Q = Φ · A_cross  [ions/s].

    Parameters
    ----------
    ds : datasets.Dataset
        MAVEN in-situ dataset (from ``load_maven_insitu``).
    altitude_km : float
        Altitude above Mars surface for the density look-up [km].
    species : list[str] | None
        Ion species to evaluate.  Defaults to ``["O+", "O2+", "CO2+"]``.
    exobase_altitude_km : float
        Mars exobase altitude for the Jeans formula [km].  Default 200 km.
    cross_section_area_km2 : float
        Effective cross-sectional area over which the flux escapes [km²].
        Default is ~6 × 10⁶ km² (hemisphere-like area for Mars upper atm).

    Returns
    -------
    dict
        ``species_results`` : per-species dict with density, thermal speed,
            lambda parameter, flux, and total loss rate.
        ``total_loss_rate_ions_per_s`` : aggregate loss across all species.
        ``altitude_km`` : queried altitude.
    """
    if species is None:
        species = ["O+", "O2+", "CO2+"]

    r_mars_m = R_MARS_KM * 1.0e3
    r_exo_m = r_mars_m + exobase_altitude_km * 1.0e3
    cross_section_m2 = cross_section_area_km2 * 1.0e6  # km² → m²

    # ── Extract densities at the requested altitude ──────────────────────
    ion_densities = _extract_ion_densities(ds, altitude_km, species)

    results: Dict[str, Any] = {}
    total_loss = 0.0

    for sp in species:
        n_density = ion_densities.get(sp, 0.0)  # ions/m³
        m_ion = _ION_MASS.get(sp)
        if m_ion is None:
            logger.warning("Unknown ion species '%s'; skipping.", sp)
            continue

        T_exo = _DEFAULT_EXOBASE_TEMP_K.get(sp, 250.0)

        # Jeans escape parameter
        lam = (GM_MARS * m_ion) / (K_BOLTZMANN * T_exo * r_exo_m)

        # Most-probable thermal speed
        v_th = math.sqrt(2.0 * K_BOLTZMANN * T_exo / m_ion)

        # Jeans flux [ions m⁻² s⁻¹]
        phi = (n_density * v_th / (2.0 * math.sqrt(math.pi))) * (1.0 + lam) * math.exp(-lam)

        # Total loss rate [ions/s]
        q_loss = phi * cross_section_m2

        results[sp] = {
            "density_ions_per_m3": n_density,
            "thermal_speed_m_s": v_th,
            "jeans_lambda": lam,
            "flux_ions_per_m2_s": phi,
            "loss_rate_ions_per_s": q_loss,
            "exobase_temp_K": T_exo,
        }
        total_loss += q_loss

    return {
        "species_results": results,
        "total_loss_rate_ions_per_s": total_loss,
        "altitude_km": altitude_km,
        "exobase_altitude_km": exobase_altitude_km,
    }


def _extract_ion_densities(
    ds,
    altitude_km: float,
    species: list[str],
    altitude_tolerance_km: float = 10.0,
) -> Dict[str, float]:
    """Extract mean ion densities near a target altitude from the MAVEN dataset.

    Searches dataset columns for altitude and density fields, averaging
    samples within *altitude_tolerance_km* of the requested altitude.

    Returns a dict mapping species name → number density [ions/m³].
    """
    # Column-name heuristics (MAVEN KP datasets use varied naming)
    alt_candidates = ["altitude", "alt", "ALT", "Altitude", "periapsis_altitude"]
    density_column_map: Dict[str, list[str]] = {
        "O+": ["O+_density", "oplus_density", "O_plus_density"],
        "O2+": ["O2+_density", "o2plus_density", "O2_plus_density"],
        "CO2+": ["CO2+_density", "co2plus_density", "CO2_plus_density"],
    }

    # Materialise the column names available in the dataset
    try:
        col_names = set(ds.column_names) if hasattr(ds, "column_names") else set()
    except Exception:
        col_names = set()

    alt_col = None
    for c in alt_candidates:
        if c in col_names:
            alt_col = c
            break

    densities: Dict[str, list[float]] = {sp: [] for sp in species}

    if alt_col is None:
        logger.warning(
            "Altitude column not found in MAVEN dataset; "
            "returning zero densities.  Available columns: %s",
            col_names,
        )
        return {sp: 0.0 for sp in species}

    # Determine which density columns are present per species
    sp_col: Dict[str, Optional[str]] = {}
    for sp in species:
        sp_col[sp] = None
        for c in density_column_map.get(sp, []):
            if c in col_names:
                sp_col[sp] = c
                break

    # Iterate and accumulate
    try:
        for row in ds:
            alt_val = row.get(alt_col)
            if alt_val is None:
                continue
            try:
                alt_val = float(alt_val)
            except (TypeError, ValueError):
                continue
            if abs(alt_val - altitude_km) > altitude_tolerance_km:
                continue
            for sp in species:
                dc = sp_col.get(sp)
                if dc is not None and row.get(dc) is not None:
                    try:
                        densities[sp].append(float(row[dc]))
                    except (TypeError, ValueError):
                        pass
    except Exception as exc:
        logger.warning("Error iterating MAVEN dataset: %s", exc)

    result: Dict[str, float] = {}
    for sp in species:
        vals = densities[sp]
        if vals:
            # Convert from cm⁻³ (common MAVEN unit) to m⁻³
            result[sp] = float(np.mean(vals)) * 1.0e6
        else:
            logger.info(
                "No density samples for species '%s' near %.1f km; defaulting to 0.",
                sp,
                altitude_km,
            )
            result[sp] = 0.0

    return result


# ============================================================================
# 2. OpenMars Reanalysis — Mars EDL Atmospheric Profiles
# ============================================================================

def load_openmars_reanalysis(**kwargs: Any):
    """Load the ARCO-OpenMars reanalysis dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
        Mars climate reanalysis data (temperature, pressure, dust opacity, etc.)
        gridded in latitude, longitude, altitude, and solar longitude (Ls).
    """
    logger.info("Loading OpenMars reanalysis dataset")
    return get_dataset("openmars_reanalysis", **kwargs)


def get_martian_entry_profile(
    ds,
    lat: float,
    lon: float,
    solar_longitude_ls: float,
    *,
    alt_min_km: float = 0.0,
    alt_max_km: float = 120.0,
    n_levels: int = 121,
) -> Dict[str, Any]:
    """Extract Mars atmospheric profiles for EDL simulation at a given location.

    Interpolates temperature, density, pressure, and dust-opacity profiles
    from the OpenMars reanalysis grid nearest the requested (lat, lon, Ls).

    If the dataset does not contain grid data at the requested point the
    function falls back to an analytic Mars-GRAM-like exponential atmosphere
    so that downstream EDL codes always receive a usable profile.

    Parameters
    ----------
    ds : datasets.Dataset | datasets.IterableDataset
        OpenMars reanalysis dataset.
    lat : float
        Planetocentric latitude [deg].
    lon : float
        East longitude [deg, 0–360].
    solar_longitude_ls : float
        Solar longitude Ls [deg, 0–360] indicating Mars season.
    alt_min_km, alt_max_km : float
        Altitude bounds for the returned profile [km].
    n_levels : int
        Number of altitude grid points.

    Returns
    -------
    dict
        Keys:
            altitude_km : np.ndarray
            temperature_K : np.ndarray
            density_kg_m3 : np.ndarray
            pressure_Pa : np.ndarray
            dust_opacity : np.ndarray
            metadata : dict  (lat, lon, Ls, source)
    """
    alt_grid = np.linspace(alt_min_km, alt_max_km, n_levels)

    # Attempt extraction from the dataset
    profile = _extract_openmars_profile(ds, lat, lon, solar_longitude_ls, alt_grid)
    if profile is not None:
        return profile

    logger.info(
        "Falling back to analytic Mars atmosphere for lat=%.1f, lon=%.1f, Ls=%.1f",
        lat, lon, solar_longitude_ls,
    )
    return _analytic_mars_atmosphere(lat, lon, solar_longitude_ls, alt_grid)


def _extract_openmars_profile(
    ds,
    lat: float,
    lon: float,
    ls: float,
    alt_grid: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Try to interpolate profile from OpenMars columnar data.

    Returns ``None`` if extraction fails (missing columns, empty slice, etc.).
    """
    try:
        col_names = set(ds.column_names) if hasattr(ds, "column_names") else set()
    except Exception:
        col_names = set()

    needed = {"latitude", "longitude", "altitude", "temperature"}
    if not needed.issubset(col_names):
        logger.debug(
            "OpenMars extraction skipped: missing columns %s",
            needed - col_names,
        )
        return None

    # Collect rows nearest to the requested coordinates
    lat_tol, lon_tol, ls_tol = 2.5, 3.75, 5.0
    rows: list[dict] = []
    ls_col = None
    for c in ("solar_longitude", "Ls", "ls", "sol_lon"):
        if c in col_names:
            ls_col = c
            break

    try:
        for row in ds:
            rlat = float(row["latitude"])
            rlon = float(row["longitude"])
            if abs(rlat - lat) > lat_tol or abs(rlon - lon) > lon_tol:
                continue
            if ls_col is not None:
                rls = float(row[ls_col])
                if abs(rls - ls) > ls_tol:
                    continue
            rows.append(row)
    except Exception as exc:
        logger.debug("OpenMars iteration error: %s", exc)
        return None

    if not rows:
        return None

    # Build arrays and interpolate onto alt_grid
    alts = np.array([float(r["altitude"]) for r in rows])
    temps = np.array([float(r["temperature"]) for r in rows])
    sort_idx = np.argsort(alts)
    alts, temps = alts[sort_idx], temps[sort_idx]

    temp_interp = np.interp(alt_grid, alts, temps)

    # Derive density via hydrostatic exponential from temperature profile
    R_specific = 191.84  # CO₂-dominated gas constant [J/(kg·K)]
    p_surface = 636.0  # Pa (mean Mars surface pressure)
    dz = (alt_grid[1] - alt_grid[0]) * 1.0e3 if len(alt_grid) > 1 else 1000.0

    pressure = np.empty_like(alt_grid)
    density = np.empty_like(alt_grid)
    pressure[0] = p_surface
    density[0] = p_surface / (R_specific * temp_interp[0])
    for i in range(1, len(alt_grid)):
        scale_h = R_specific * temp_interp[i] / G_MARS_SURFACE
        pressure[i] = pressure[i - 1] * math.exp(-dz / scale_h)
        density[i] = pressure[i] / (R_specific * temp_interp[i])

    # Dust opacity (if present)
    dust_col = None
    for c in ("dust_opacity", "dust", "cdod", "CDOD"):
        if c in col_names:
            dust_col = c
            break
    if dust_col is not None:
        dust_vals = np.array([float(r.get(dust_col, 0.3)) for r in rows])
        dust_interp = np.interp(alt_grid, alts, dust_vals[sort_idx])
    else:
        # Simple exponential dust falloff from surface
        dust_interp = 0.3 * np.exp(-alt_grid / 10.0)

    return {
        "altitude_km": alt_grid,
        "temperature_K": temp_interp,
        "density_kg_m3": density,
        "pressure_Pa": pressure,
        "dust_opacity": dust_interp,
        "metadata": {
            "lat_deg": lat,
            "lon_deg": lon,
            "solar_longitude_ls": ls,
            "source": "openmars_reanalysis",
        },
    }


def _analytic_mars_atmosphere(
    lat: float,
    lon: float,
    solar_longitude_ls: float,
    alt_grid: np.ndarray,
) -> Dict[str, Any]:
    """Generate an analytic Mars atmospheric profile (Mars-GRAM-like).

    Uses a bi-exponential model:
        T(z) = T_surface − lapse_rate · z   (troposphere, z < 40 km)
        T(z) = T_tropopause                 (isothermal above 40 km)
        ρ(z) = ρ_0 · exp(−z / H(T))
    """
    # Surface temperature variation with latitude and season
    T_equator = 215.0  # K
    delta_T_lat = -40.0 * abs(math.sin(lat * DEG2RAD))  # cooler at poles
    delta_T_season = 15.0 * math.cos((solar_longitude_ls - 250.0) * DEG2RAD)
    T_surface = T_equator + delta_T_lat + delta_T_season

    lapse_rate_K_per_km = 2.5  # K/km (Mars average lapse rate)
    tropo_ceiling_km = 40.0
    T_tropo = T_surface - lapse_rate_K_per_km * tropo_ceiling_km

    R_specific = 191.84  # J/(kg·K), CO₂-dominated
    p_surface = 636.0    # Pa
    rho_surface = p_surface / (R_specific * T_surface)

    temperature = np.where(
        alt_grid < tropo_ceiling_km,
        T_surface - lapse_rate_K_per_km * alt_grid,
        T_tropo,
    )
    temperature = np.maximum(temperature, 100.0)  # floor at 100 K

    # Hydrostatic integration
    dz_m = np.diff(alt_grid, prepend=0.0) * 1.0e3
    scale_heights = R_specific * temperature / G_MARS_SURFACE
    cumulative_z = np.cumsum(dz_m / scale_heights)
    density = rho_surface * np.exp(-cumulative_z)
    pressure = density * R_specific * temperature

    # Dust opacity: exponential falloff with scale height ~10 km
    tau_surface = 0.3 + 0.2 * math.sin(solar_longitude_ls * DEG2RAD)
    dust_opacity = tau_surface * np.exp(-alt_grid / 10.0)

    return {
        "altitude_km": alt_grid,
        "temperature_K": temperature,
        "density_kg_m3": density,
        "pressure_Pa": pressure,
        "dust_opacity": dust_opacity,
        "metadata": {
            "lat_deg": lat,
            "lon_deg": lon,
            "solar_longitude_ls": solar_longitude_ls,
            "source": "analytic_mars_gram_like",
        },
    }


# ============================================================================
# 3. Venus Express — Cloud Albedo Analysis
# ============================================================================

def load_venus_express_observations(**kwargs: Any):
    """Load the ESA Venus Express observations dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
        Venus Express UV/IR spectral observation records.
    """
    logger.info("Loading Venus Express observations dataset")
    return get_dataset("venus_express_obs", **kwargs)


def extract_venus_cloud_albedo(
    ds,
    wavelength_nm: float,
    *,
    wavelength_tolerance_nm: float = 10.0,
) -> Dict[str, Any]:
    """Estimate Venus cloud-top albedo at a given wavelength.

    Searches the Venus Express dataset for spectral measurements near the
    requested wavelength and computes a mean geometric albedo.

    If dataset columns do not contain spectral data, an analytic albedo
    model based on known Venus reflectance spectra is used as a fallback.

    Parameters
    ----------
    ds : datasets.Dataset | datasets.IterableDataset
        Venus Express observations dataset.
    wavelength_nm : float
        Target wavelength [nm].
    wavelength_tolerance_nm : float
        Half-width of the spectral bin for averaging [nm].

    Returns
    -------
    dict
        ``albedo`` : estimated geometric albedo at wavelength.
        ``wavelength_nm`` : queried wavelength.
        ``n_samples`` : number of spectral samples averaged.
        ``source`` : data source label.
    """
    albedo_values: list[float] = []

    # Try extracting from dataset
    try:
        col_names = set(ds.column_names) if hasattr(ds, "column_names") else set()
    except Exception:
        col_names = set()

    wl_col = None
    for c in ("wavelength", "wavelength_nm", "lambda", "wl"):
        if c in col_names:
            wl_col = c
            break

    albedo_col = None
    for c in ("albedo", "reflectance", "geometric_albedo", "I_over_F"):
        if c in col_names:
            albedo_col = c
            break

    if wl_col is not None and albedo_col is not None:
        try:
            for row in ds:
                wl_val = row.get(wl_col)
                a_val = row.get(albedo_col)
                if wl_val is None or a_val is None:
                    continue
                try:
                    wl_val = float(wl_val)
                    a_val = float(a_val)
                except (TypeError, ValueError):
                    continue
                if abs(wl_val - wavelength_nm) <= wavelength_tolerance_nm:
                    albedo_values.append(a_val)
        except Exception as exc:
            logger.warning("Error iterating Venus Express dataset: %s", exc)

    if albedo_values:
        mean_albedo = float(np.mean(albedo_values))
        return {
            "albedo": mean_albedo,
            "wavelength_nm": wavelength_nm,
            "n_samples": len(albedo_values),
            "std_albedo": float(np.std(albedo_values)),
            "source": "venus_express_obs",
        }

    # Fallback: analytic Venus albedo model
    logger.info(
        "No spectral matches in Venus Express dataset at %.1f nm; "
        "using analytic albedo model.",
        wavelength_nm,
    )
    return _analytic_venus_albedo(wavelength_nm)


def _analytic_venus_albedo(wavelength_nm: float) -> Dict[str, Any]:
    """Analytic Venus cloud albedo model.

    Piecewise model based on published Venus reflectance:
      • UV (200–400 nm):  strong absorption by unknown UV absorber;
        albedo ~ 0.30–0.50
      • Visible (400–700 nm):  high Rayleigh + H₂SO₄ cloud scattering;
        albedo ~ 0.65–0.80
      • Near-IR (700–1000 nm):  CO₂ absorption bands reduce albedo;
        albedo ~ 0.40–0.60
      • Thermal IR (> 1000 nm):  very low reflectance;
        albedo ~ 0.05–0.20
    """
    wl = wavelength_nm

    if wl < 200.0:
        albedo = 0.10
    elif wl < 300.0:
        # Deep UV — UV absorber dominates
        albedo = 0.30 + 0.10 * ((wl - 200.0) / 100.0)
    elif wl < 400.0:
        # Near-UV transition
        albedo = 0.40 + 0.25 * ((wl - 300.0) / 100.0)
    elif wl < 550.0:
        # Visible peak
        albedo = 0.65 + 0.15 * math.sin(math.pi * (wl - 400.0) / 300.0)
    elif wl < 700.0:
        # Visible plateau
        albedo = 0.75 - 0.10 * ((wl - 550.0) / 150.0)
    elif wl < 1000.0:
        # Near-IR decline
        albedo = 0.65 - 0.25 * ((wl - 700.0) / 300.0)
    elif wl < 2500.0:
        # Thermal windows
        albedo = 0.40 * math.exp(-(wl - 1000.0) / 800.0)
    else:
        albedo = 0.05

    return {
        "albedo": round(albedo, 4),
        "wavelength_nm": wavelength_nm,
        "n_samples": 0,
        "source": "analytic_venus_model",
    }


# ============================================================================
# 4. Artemis II — Circumlunar Free-Return Trajectory (Patched-Conics)
# ============================================================================

def load_artemis_ii_mission_profile(**kwargs: Any):
    """Load the Artemis II mission profile dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
        Mission timeline, trajectory waypoints, and vehicle parameters
        for Artemis II.
    """
    logger.info("Loading Artemis II mission profile dataset")
    return get_dataset("artemis_ii", **kwargs)


def simulate_free_return_trajectory(
    v_tli_km_s: float = 10.95,
    perilune_alt_km: float = 200.0,
    *,
    parking_orbit_alt_km: float = 185.0,
) -> Dict[str, Any]:
    """Simulate a circumlunar free-return trajectory using patched-conics.

    Computes the full two-body patched-conic solution for a trans-lunar
    injection (TLI) burn from low-Earth orbit, a hyperbolic flyby of the
    Moon, and the resulting trans-Earth coast.

    Patch point is the Moon's sphere of influence (SOI) boundary.

    Parameters
    ----------
    v_tli_km_s : float
        Inertial speed after TLI burn [km/s]. Default 10.95 km/s (typical circumlunar).
    perilune_alt_km : float
        Closest approach altitude above the lunar surface [km].
    parking_orbit_alt_km : float
        Altitude of the initial circular parking orbit around Earth [km].

    Returns
    -------
    dict
        ``v_circular_parking_km_s`` : circular orbital speed in parking orbit.
        ``v_tli_km_s`` : TLI speed used.
        ``delta_v_tli_km_s`` : TLI Δv from parking orbit.
        ``v_escape_km_s`` : Earth escape speed at parking altitude.
        ``v_infinity_km_s`` : hyperbolic excess speed w.r.t. Earth.
        ``energy_earth_km2_s2`` : specific orbital energy (Earth-centred).
        ``sma_transfer_km`` : semi-major axis of transfer ellipse.
        ``perilune_alt_km`` : perilune altitude.
        ``v_perilune_km_s`` : speed at perilune (Moon-centred hyperbola).
        ``flyby_turning_angle_deg`` : hyperbolic turning angle at Moon.
        ``delta_v_tei_km_s`` : Δv for trans-Earth injection (if powered).
        ``outbound_tof_hours`` : approximate outbound flight time.
        ``total_mission_time_hours`` : approximate total flight time (out + return).
        ``moon_soi_radius_km`` : lunar sphere of influence radius.
    """
    # ── Unit conversions ────────────────────────────────────────────────
    GM_E = GM_EARTH * 1.0e-9   # km³/s²
    GM_M = GM_MOON * 1.0e-9    # km³/s²
    R_E = R_EARTH_KM            # km
    R_M = R_MOON_KM             # km
    a_moon = A_MOON_KM           # km (Earth–Moon distance)

    r_parking = R_E + parking_orbit_alt_km  # km
    r_perilune = R_M + perilune_alt_km       # km

    # ── 1. Parking orbit ────────────────────────────────────────────────
    v_circ = math.sqrt(GM_E / r_parking)  # circular speed [km/s]

    # ── 2. TLI burn ─────────────────────────────────────────────────────
    delta_v_tli = v_tli_km_s - v_circ

    # ── 3. Earth-centred departure hyperbola / ellipse ──────────────────
    v_escape = math.sqrt(2.0 * GM_E / r_parking)
    energy_earth = 0.5 * v_tli_km_s**2 - GM_E / r_parking  # km²/s²

    if energy_earth >= 0.0:
        # Hyperbolic departure
        v_inf = math.sqrt(2.0 * energy_earth)
        sma_transfer = -GM_E / (2.0 * energy_earth)  # negative for hyperbola
    else:
        # Elliptic transfer (typical for free-return)
        v_inf = 0.0
        sma_transfer = -GM_E / (2.0 * energy_earth)  # positive for ellipse

    # ── 4. Lunar sphere of influence ────────────────────────────────────
    # r_SOI ≈ a_moon · (M_moon / M_earth)^(2/5)
    mass_ratio = GM_M / GM_E  # mass ratio (GM ratio = mass ratio)
    r_soi = a_moon * mass_ratio ** 0.4

    # ── 5. Approach speed at Moon SOI boundary (Earth-centred) ──────────
    # v at Moon's distance from Earth on the transfer orbit:
    v_at_moon_sq = max(0.0, 2.0 * (energy_earth + GM_E / a_moon))
    v_at_moon = math.sqrt(v_at_moon_sq)

    # Moon's orbital speed
    v_moon = math.sqrt(GM_E / a_moon)

    # Relative approach speed at SOI entry (simplified: co-planar,
    # approach along velocity vector)
    v_approach = abs(v_at_moon - v_moon)

    # ── 6. Moon-centred hyperbolic flyby ────────────────────────────────
    # Energy of Moon-centred hyperbola
    energy_moon = 0.5 * v_approach**2  # v_inf² / 2
    sma_moon = GM_M / (2.0 * energy_moon)  # positive (Moon hyperbola a)

    # Speed at perilune
    v_perilune = math.sqrt(v_approach**2 + 2.0 * GM_M / r_perilune)

    # Eccentricity of Moon hyperbola
    e_moon = 1.0 + r_perilune / sma_moon

    # Turning angle (total deflection)
    if e_moon > 1.0:
        turning_angle_rad = 2.0 * math.asin(1.0 / e_moon)
    else:
        turning_angle_rad = math.pi  # degenerate case
    turning_angle_deg = turning_angle_rad * RAD2DEG

    # ── 7. Trans-Earth injection Δv ─────────────────────────────────────
    # For a free-return, TEI Δv ≈ 0 (gravity alone bends trajectory back).
    # If a powered flyby is needed (e.g. to lower return perigee), estimate:
    #   Δv_TEI = |v_perilune − v_perilune_free_return|
    # For a pure free-return this is zero by definition; we report a small
    # correction budget for mid-course manoeuvres.
    v_perilune_escape = math.sqrt(2.0 * GM_M / r_perilune)
    delta_v_tei = max(0.0, v_perilune_escape - v_perilune)  # typically ≈ 0

    # ── 8. Flight-time estimates ────────────────────────────────────────
    if sma_transfer > 0:
        # Elliptic half-period as outbound TOF approximation
        period_s = 2.0 * math.pi * math.sqrt(sma_transfer**3 / GM_E)
        outbound_tof_s = period_s / 2.0  # half-orbit to Moon vicinity
    else:
        # Hyperbolic — use straight-line kinematic estimate
        outbound_tof_s = a_moon / v_tli_km_s  # very rough

    # Refine with a simple distance / mean-speed heuristic
    mean_speed_out = 0.5 * (v_tli_km_s + v_at_moon)
    outbound_tof_refined_s = a_moon / mean_speed_out

    # Use the more physical estimate
    outbound_hours = min(outbound_tof_s, outbound_tof_refined_s) / 3600.0
    # Clamp to reasonable range (50–120 h for Apollo / Artemis class)
    outbound_hours = float(np.clip(outbound_hours, 50.0, 120.0))
    total_hours = outbound_hours * 2.0 + 2.0  # ≈ symmetric + flyby dwell

    # ── Assemble results ────────────────────────────────────────────────
    return {
        "v_circular_parking_km_s": round(v_circ, 4),
        "v_tli_km_s": round(v_tli_km_s, 4),
        "delta_v_tli_km_s": round(delta_v_tli, 4),
        "v_escape_km_s": round(v_escape, 4),
        "v_infinity_km_s": round(v_inf, 4),
        "energy_earth_km2_s2": round(energy_earth, 6),
        "sma_transfer_km": round(sma_transfer, 2),
        "v_at_moon_km_s": round(v_at_moon, 4),
        "v_approach_moon_km_s": round(v_approach, 4),
        "perilune_alt_km": round(perilune_alt_km, 2),
        "v_perilune_km_s": round(v_perilune, 4),
        "e_flyby": round(e_moon, 6),
        "flyby_turning_angle_deg": round(turning_angle_deg, 2),
        "delta_v_tei_km_s": round(delta_v_tei, 4),
        "outbound_tof_hours": round(outbound_hours, 2),
        "total_mission_time_hours": round(total_hours, 2),
        "moon_soi_radius_km": round(r_soi, 2),
    }
