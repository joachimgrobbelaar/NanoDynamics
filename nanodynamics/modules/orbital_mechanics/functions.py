"""
nanodynamics/modules/orbital_mechanics/functions.py

Pillar 2: Orbital Mechanics, Flight Dynamics, Constellations & Space
Situational Awareness (SSA).

Implements propagation, conjunction screening, fragmentation analysis,
constellation geometry, transfer orbit planning, and orbital stability
diagnostics.  All physics uses SI / metric units (m, km, s, rad, kg, W/m²).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from nanodynamics.core.constants import (
    AU_KM,
    AU_M,
    CD_TYPICAL,
    DAY_S,
    DEG2RAD,
    EARTH_ROT_RAD_S,
    EARTH_SIDEREAL_DAY_S,
    GM_EARTH,
    GM_SUN,
    J2_EARTH,
    R_EARTH_KM,
    R_EARTH_M,
    RAD2DEG,
    SCALE_HEIGHT_KM,
    SEA_LEVEL_DENSITY_KG_M3,
    YEAR_S,
)
from nanodynamics.data_ingestion.loaders import get_dataset

logger = logging.getLogger("nanodynamics.orbital_mechanics")

# Standard gravitational acceleration [m/s²]
G0 = 9.80665

# ── Convenience: Earth GM in km³/s² for TLE / SGP4 work ─────────────────────
_GM_EARTH_KM3 = GM_EARTH * 1e-9  # km³/s²

# Minutes-per-day constant used by TLE mean-motion convention
_MIN_PER_DAY = 1440.0

__all__ = [
    # 1  – TLE / SGP4
    "load_starlink_tles",
    "propagate_sgp4",
    "compute_tle_age",
    # 2  – SATCAT
    "load_satcat_catalog",
    "query_active_payloads_by_regime",
    "get_satellite_decay_date",
    # 3  – Conjunctions
    "load_satellite_conjunctions",
    "calculate_collision_probability",
    "suggest_avoidance_delta_v",
    # 4  – Fragmentation
    "load_fragmentation_events",
    "generate_gabbard_diagram_data",
    "model_fragment_cloud_dispersion",
    # 5  – Reentry
    "load_reentry_events",
    "predict_surviving_mass_fraction",
    # 6  – Constellation census
    "load_constellation_census",
    "get_walker_delta_parameters",
    # 7  – Starlink fleet
    "load_starlink_fleet",
    "compute_shell_distribution",
    # 8  – OneWeb fleet
    "load_oneweb_fleet",
    "compute_polar_pass_intervals",
    # 9  – Kuiper fleet
    "load_kuiper_fleet",
    "simulate_deployment_timeline",
    # 10 – Globalstar fleet
    "load_globalstar_fleet",
    "compute_bent_pipe_coverage",
    # 11 – UCS satellite database
    "load_ucs_satellite_database",
    "filter_by_propulsion_and_power",
    # 12 – LEOMSAT
    "load_leomsat_dataset",
    "evaluate_multispectral_orbit_suitability",
    # 13 – HET low-thrust transfer
    "load_het_transfer_data",
    "compute_low_thrust_transfer",
    # 14 – Hohmann transfer
    "load_hektor_hohmann_data",
    "calculate_hohmann_transfer_dv",
    # 15 – Rendezvous (CW)
    "load_orbit_planner_benchmarks",
    "plan_constrained_rendezvous",
    # 16 – OrbitBench
    "load_orbitbench",
    "benchmark_orbit_solver",
    # 17 – OrbitWren
    "load_orbitwren_tracker",
    "profile_flight_software_workload",
    # 18 – Orbital chaos / Lyapunov
    "load_orbital_chaos_dataset",
    "compute_lyapunov_exponent",
]


# ============================================================================
# 1. TLE Parsing & Simplified SGP4 Propagation
# ============================================================================

def load_starlink_tles(**kwargs) -> Any:
    """Load the latest Starlink TLE dataset from Hugging Face.

    Returns
    -------
    Dataset
        A Hugging Face ``Dataset`` (or ``IterableDataset``) containing
        columns such as ``TLE_LINE1``, ``TLE_LINE2``, ``OBJECT_NAME``,
        etc.
    """
    logger.info("Loading Starlink TLE dataset …")
    return get_dataset("starlink_tle_latest", **kwargs)


def _parse_tle_epoch(line1: str) -> Tuple[int, float]:
    """Extract epoch year and fractional day from TLE line 1.

    Parameters
    ----------
    line1 : str
        Standard 69-character TLE line 1.

    Returns
    -------
    epoch_year : int
        Four-digit year.
    epoch_day : float
        Fractional day of year (1-based).
    """
    epoch_str = line1[18:32].strip()
    yr2 = int(epoch_str[:2])
    epoch_year = yr2 + (1900 if yr2 >= 57 else 2000)
    epoch_day = float(epoch_str[2:])
    return epoch_year, epoch_day


def _epoch_to_datetime(epoch_year: int, epoch_day: float) -> datetime:
    """Convert TLE epoch (year + fractional day) to a UTC datetime."""
    jan1 = datetime(epoch_year, 1, 1, tzinfo=timezone.utc)
    return jan1 + __import__("datetime").timedelta(days=epoch_day - 1.0)


def propagate_sgp4(
    tle_line1: str,
    tle_line2: str,
    target_datetime: datetime,
) -> Dict[str, NDArray[np.float64]]:
    """Simplified SGP4-style propagator: Keplerian + secular J2 + drag.

    This is an *analytical* propagator that extracts classical elements
    from the TLE, applies secular J2 rates (RAAN & arg-of-perigee drift)
    and exponential mean-motion decay from the first-derivative drag
    term (n-dot), then evaluates position and velocity in the
    True-Equator-Mean-Equinox (TEME) frame.

    Parameters
    ----------
    tle_line1 : str
        TLE line 1 (69 characters).
    tle_line2 : str
        TLE line 2 (69 characters).
    target_datetime : datetime
        Target UTC epoch to propagate to.

    Returns
    -------
    dict
        ``position_km``  – (3,) ndarray [km],
        ``velocity_km_s`` – (3,) ndarray [km/s],
        ``epoch_dt``      – TLE epoch as ``datetime``,
        ``elapsed_min``   – propagation interval in minutes.
    """
    # ── Parse TLE line 1 ─────────────────────────────────────────────────
    epoch_year, epoch_day = _parse_tle_epoch(tle_line1)
    # First derivative of mean motion (rev/day²) ÷ 2
    ndot_str = tle_line1[33:43].strip()
    ndot_half = float(ndot_str)  # rev/day² / 2
    # B* drag term (pseudo-ballistic coefficient)
    bstar_mantissa = tle_line1[53:59].strip()
    bstar_exp = tle_line1[59:61].strip()
    try:
        bstar = float(f"0.{bstar_mantissa}") * 10.0 ** float(bstar_exp)
        if tle_line1[53] == "-":
            bstar = -bstar
    except (ValueError, IndexError):
        bstar = 0.0

    # ── Parse TLE line 2 ─────────────────────────────────────────────────
    inc_deg = float(tle_line2[8:16].strip())
    raan_deg = float(tle_line2[17:25].strip())
    ecc = float(f"0.{tle_line2[26:33].strip()}")
    argp_deg = float(tle_line2[34:42].strip())
    ma_deg = float(tle_line2[43:51].strip())
    n_revday = float(tle_line2[52:63].strip())  # mean motion [rev/day]

    # Derived quantities
    n_rad_s = n_revday * 2.0 * math.pi / DAY_S  # [rad/s]
    a_km = (_GM_EARTH_KM3 / (n_rad_s ** 2)) ** (1.0 / 3.0)  # semi-major axis [km]
    p_km = a_km * (1.0 - ecc ** 2)  # semi-latus rectum [km]

    # ── Elapsed time ─────────────────────────────────────────────────────
    epoch_dt = _epoch_to_datetime(epoch_year, epoch_day)
    if target_datetime.tzinfo is None:
        target_datetime = target_datetime.replace(tzinfo=timezone.utc)
    dt_s = (target_datetime - epoch_dt).total_seconds()
    dt_min = dt_s / 60.0

    # ── Secular J2 rates ─────────────────────────────────────────────────
    cos_i = math.cos(math.radians(inc_deg))
    sin_i = math.sin(math.radians(inc_deg))
    re_over_p = R_EARTH_KM / p_km

    # RAAN drift [rad/s]
    raan_dot = -1.5 * n_rad_s * J2_EARTH * (re_over_p ** 2) * cos_i
    # Argument-of-perigee drift [rad/s]
    argp_dot = 0.75 * n_rad_s * J2_EARTH * (re_over_p ** 2) * (5.0 * cos_i ** 2 - 1.0)

    # Drag-induced mean-motion change: n(t) ≈ n₀ + ndot*Δt
    ndot_rad_s2 = ndot_half * 2.0 * (2.0 * math.pi) / (DAY_S ** 2)  # to rad/s²
    n_t = n_rad_s + ndot_rad_s2 * dt_s

    # ── Mean → Eccentric → True anomaly ──────────────────────────────────
    M_rad = math.radians(ma_deg) + n_t * dt_s
    M_rad = M_rad % (2.0 * math.pi)

    # Kepler equation – Newton-Raphson
    E = M_rad
    for _ in range(25):
        dE = (E - ecc * math.sin(E) - M_rad) / (1.0 - ecc * math.cos(E))
        E -= dE
        if abs(dE) < 1e-12:
            break

    # True anomaly
    sin_nu = (math.sqrt(1.0 - ecc ** 2) * math.sin(E)) / (1.0 - ecc * math.cos(E))
    cos_nu = (math.cos(E) - ecc) / (1.0 - ecc * math.cos(E))
    nu = math.atan2(sin_nu, cos_nu)

    # ── Perifocal position & velocity ────────────────────────────────────
    r_mag = a_km * (1.0 - ecc * math.cos(E))
    r_pf = np.array([r_mag * cos_nu, r_mag * sin_nu, 0.0])

    h = math.sqrt(_GM_EARTH_KM3 * p_km)
    v_pf = (_GM_EARTH_KM3 / h) * np.array([-math.sin(nu), ecc + math.cos(nu), 0.0])

    # ── Perifocal → TEME rotation ────────────────────────────────────────
    raan_rad = math.radians(raan_deg) + raan_dot * dt_s
    argp_rad = math.radians(argp_deg) + argp_dot * dt_s
    inc_rad = math.radians(inc_deg)

    cO, sO = math.cos(raan_rad), math.sin(raan_rad)
    cw, sw = math.cos(argp_rad), math.sin(argp_rad)
    ci, si = math.cos(inc_rad), math.sin(inc_rad)

    R_pf_to_eci = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                  ci      ],
    ])

    pos_km = R_pf_to_eci @ r_pf
    vel_km_s = R_pf_to_eci @ v_pf

    return {
        "position_km": pos_km,
        "velocity_km_s": vel_km_s,
        "epoch_dt": epoch_dt,
        "elapsed_min": dt_min,
    }


def compute_tle_age(
    epoch_year: int,
    epoch_day: float,
    current_datetime: datetime,
) -> float:
    """Return TLE age in fractional days.

    Parameters
    ----------
    epoch_year : int
        Four-digit TLE epoch year.
    epoch_day : float
        Fractional day of year from the TLE epoch.
    current_datetime : datetime
        Reference time (UTC).

    Returns
    -------
    float
        Age of the TLE in days (positive = TLE is older).
    """
    epoch_dt = _epoch_to_datetime(epoch_year, epoch_day)
    if current_datetime.tzinfo is None:
        current_datetime = current_datetime.replace(tzinfo=timezone.utc)
    return (current_datetime - epoch_dt).total_seconds() / DAY_S


# ============================================================================
# 2. SATCAT Catalog Queries
# ============================================================================

def load_satcat_catalog(**kwargs) -> Any:
    """Load the Space-Track SATCAT catalogue.

    Returns
    -------
    Dataset
        A Hugging Face ``Dataset`` with columns like ``NORAD_CAT_ID``,
        ``OBJECT_TYPE``, ``OPS_STATUS_CODE``, ``ORBIT_CENTER``,
        ``PERIOD_MIN``, ``APOGEE``, ``PERIGEE``, ``DECAY_DATE``, etc.
    """
    logger.info("Loading SATCAT catalog …")
    return get_dataset("satcat_catalog", **kwargs)


def query_active_payloads_by_regime(ds: Any, regime: str = "LEO") -> Any:
    """Filter SATCAT for active payloads in the given orbit regime.

    The orbit regime is inferred from the ``PERIOD_MIN`` column:

    * **LEO** : period < 128 min  (altitude ≲ 2 000 km)
    * **MEO** : 128 ≤ period < 1400 min
    * **GEO** : 1400 ≤ period ≤ 1500 min  (≈ 35 786 km)
    * **HEO** : period > 1500 min

    Parameters
    ----------
    ds : Dataset
        SATCAT dataset from :func:`load_satcat_catalog`.
    regime : str
        One of ``"LEO"``, ``"MEO"``, ``"GEO"``, ``"HEO"`` (case-insensitive).

    Returns
    -------
    Dataset
        Filtered subset.
    """
    regime = regime.upper()
    period_bounds: Dict[str, Tuple[float, float]] = {
        "LEO": (0.0, 128.0),
        "MEO": (128.0, 1400.0),
        "GEO": (1400.0, 1500.0),
        "HEO": (1500.0, 1e9),
    }
    if regime not in period_bounds:
        raise ValueError(f"Unknown regime '{regime}'. Choose from {list(period_bounds.keys())}.")

    lo, hi = period_bounds[regime]

    def _filter(row: Dict[str, Any]) -> bool:
        p = row.get("PERIOD_MIN") or row.get("PERIOD") or row.get("period_min")
        if p is None:
            return False
        try:
            p = float(p)
        except (TypeError, ValueError):
            return False
        otype = str(row.get("OBJECT_TYPE", "")).upper()
        status = str(row.get("OPS_STATUS_CODE", "")).upper()
        return lo <= p < hi and "PAY" in otype and status in ("", "+", "P", "B", "S", "X")

    return ds.filter(_filter)


def get_satellite_decay_date(ds: Any, norad_id: int) -> Optional[str]:
    """Look up the decay date for a satellite by its NORAD catalogue ID.

    Parameters
    ----------
    ds : Dataset
        SATCAT dataset.
    norad_id : int
        NORAD catalogue identifier.

    Returns
    -------
    str or None
        ISO-style decay date string, or ``None`` if not decayed.
    """
    for row in ds:
        nid = row.get("NORAD_CAT_ID") or row.get("norad_cat_id")
        if nid is not None and int(nid) == norad_id:
            decay = row.get("DECAY_DATE") or row.get("decay_date")
            return str(decay) if decay else None
    logger.warning("NORAD ID %d not found in dataset.", norad_id)
    return None


# ============================================================================
# 3. Conjunction Assessment & Collision Avoidance
# ============================================================================

def load_satellite_conjunctions(**kwargs) -> Any:
    """Load satellite conjunction (close-approach) events dataset."""
    logger.info("Loading satellite conjunction events …")
    return get_dataset("satellite_conjunctions", **kwargs)


def calculate_collision_probability(
    miss_distance_m: float,
    sigma_a_m: float,
    sigma_b_m: float,
    cross_section_m2: float = 10.0,
) -> float:
    """2-D collision probability via the Foster / Alfriend approximation.

    The combined covariance in the encounter plane is assumed to be a
    bivariate Gaussian with principal axes σ_a and σ_b. The collision
    probability is:

        Pc = (A_cross / (2 π σ_a σ_b)) · exp(−d² / (2 σ²))

    where σ² = σ_a² + σ_b² (isotropic simplification of the radial
    approach) and *A_cross* is the combined hard-body cross-section.

    Parameters
    ----------
    miss_distance_m : float
        Predicted miss distance at TCA [m].
    sigma_a_m : float
        1-σ uncertainty in the radial direction [m].
    sigma_b_m : float
        1-σ uncertainty in the in-track / cross-track direction [m].
    cross_section_m2 : float, optional
        Combined hard-body cross-section area [m²]. Default 10 m².

    Returns
    -------
    float
        Estimated collision probability (dimensionless, 0–1).

    Notes
    -----
    For operational CDMs the full 2-D integral (Chan/Patera) should be
    used; this Foster approximation is accurate when the combined
    covariance is much larger than the hard-body radius.
    """
    if sigma_a_m <= 0 or sigma_b_m <= 0:
        raise ValueError("Covariance axes must be positive.")
    sigma_sq = sigma_a_m ** 2 + sigma_b_m ** 2
    pc = (cross_section_m2 / (2.0 * math.pi * sigma_a_m * sigma_b_m)) * math.exp(
        -(miss_distance_m ** 2) / (2.0 * sigma_sq)
    )
    return min(pc, 1.0)


def suggest_avoidance_delta_v(
    miss_distance_m: float,
    lead_time_s: float,
    mass_kg: float,
    safe_distance_m: float = 1000.0,
) -> Dict[str, float]:
    """Estimate minimum Δv to achieve a safe miss distance.

    Uses a simple along-track impulse model: an impulse applied at time
    *t_lead* before TCA shifts the along-track position by
    Δx ≈ 2 · Δv · t_lead  (first-order CW approximation for near-
    circular orbits).

    Parameters
    ----------
    miss_distance_m : float
        Current predicted miss distance [m].
    lead_time_s : float
        Warning lead time before TCA [s].
    mass_kg : float
        Spacecraft mass [kg] (for fuel-fraction estimate).
    safe_distance_m : float, optional
        Target safe miss distance [m]. Default 1 000 m.

    Returns
    -------
    dict
        ``delta_v_m_s`` – required along-track impulse [m/s],
        ``fuel_fraction`` – approximate propellant fraction (Isp 300 s).
    """
    if lead_time_s <= 0:
        raise ValueError("Lead time must be positive.")
    gap = max(safe_distance_m - miss_distance_m, 0.0)
    # Along-track: Δx ≈ 2·Δv·Δt (linearised CW for near-circular LEO)
    delta_v = gap / (2.0 * lead_time_s)
    isp = 300.0  # representative bipropellant Isp [s]
    fuel_frac = 1.0 - math.exp(-delta_v / (G0 * isp))
    return {
        "delta_v_m_s": delta_v,
        "fuel_fraction": fuel_frac,
        "additional_miss_m": 2.0 * delta_v * lead_time_s,
    }


# ============================================================================
# 4. Fragmentation Events & Gabbard Diagrams
# ============================================================================

def load_fragmentation_events(**kwargs) -> Any:
    """Load the orbital fragmentation events dataset."""
    logger.info("Loading fragmentation events …")
    return get_dataset("orbital_fragmentations", **kwargs)


def generate_gabbard_diagram_data(
    ds: Any,
    event_id: Union[str, int],
) -> Dict[str, NDArray[np.float64]]:
    """Extract apogee / perigee vs. period data for a Gabbard diagram.

    A Gabbard diagram plots fragment apogee and perigee altitudes against
    orbital period, revealing the characteristic "butterfly" pattern of a
    breakup.

    Parameters
    ----------
    ds : Dataset
        Fragmentation events or debris catalogue dataset.
    event_id : str or int
        Identifier for the breakup event.

    Returns
    -------
    dict
        ``period_min``  – (N,) orbital periods [min],
        ``apogee_km``   – (N,) apogee altitudes [km],
        ``perigee_km``  – (N,) perigee altitudes [km].
    """
    periods: List[float] = []
    apogees: List[float] = []
    perigees: List[float] = []

    eid_str = str(event_id)
    for row in ds:
        row_eid = str(
            row.get("EVENT_ID", row.get("event_id", row.get("COSPAR_FRAGMENT", "")))
        )
        if eid_str not in row_eid:
            continue
        p = row.get("PERIOD") or row.get("PERIOD_MIN") or row.get("period_min")
        apo = row.get("APOGEE") or row.get("apogee_km")
        per = row.get("PERIGEE") or row.get("perigee_km")
        if p is not None and apo is not None and per is not None:
            try:
                periods.append(float(p))
                apogees.append(float(apo))
                perigees.append(float(per))
            except (ValueError, TypeError):
                continue

    return {
        "period_min": np.asarray(periods, dtype=np.float64),
        "apogee_km": np.asarray(apogees, dtype=np.float64),
        "perigee_km": np.asarray(perigees, dtype=np.float64),
    }


def model_fragment_cloud_dispersion(
    delta_v_spread_ms: float,
    period_s: float,
    dt_days: float,
) -> Dict[str, float]:
    """Model the dispersion of a debris cloud after a fragmentation event.

    After a breakup at time *t₀* every fragment receives a random
    velocity perturbation δv.  Over time the cloud spreads due to
    differential periods:

    * **Along-track spreading**: ΔL ≈ 3π · δv · N_orbits  (Gauss VOP)
    * **RAAN spreading**: ΔΩ ≈ (∂Ω/∂a) · Δa · dt  (J2 differential)

    Parameters
    ----------
    delta_v_spread_ms : float
        1-σ spread in fragment ejection velocity [m/s].
    period_s : float
        Parent orbit period [s].
    dt_days : float
        Elapsed time since breakup [days].

    Returns
    -------
    dict
        ``along_track_spread_km`` – 1-σ cloud extent in along-track [km],
        ``raan_spread_deg``       – differential RAAN spread [°].
    """
    n = 2.0 * math.pi / period_s  # mean motion [rad/s]
    a = (_GM_EARTH_KM3 / (n ** 2)) ** (1.0 / 3.0)  # semi-major axis [km]

    dt_s = dt_days * DAY_S
    n_orbits = dt_s / period_s

    # Along-track spreading (from Gauss VOP, tangential impulse)
    # Δa ≈ 2·δv / n  (km, converting δv to km/s)
    dv_km = delta_v_spread_ms * 1e-3
    da_km = 2.0 * dv_km / n
    # Along-track drift rate: d(along)/dt ≈ -1.5 n Δa
    along_track_km = abs(1.5 * n * da_km * dt_s)

    # Differential RAAN due to Δa (J2 secular)
    # dΩ/da ≈ (3/2) n J2 (R_E/a)^2 cos(i) / a  – take |cos(i)|≈0.5
    dOmega_da = 1.5 * n * J2_EARTH * (R_EARTH_KM / a) ** 2 * 0.5 / a  # rad/s/km
    raan_spread_rad = abs(dOmega_da * da_km * dt_s)

    return {
        "along_track_spread_km": along_track_km,
        "raan_spread_deg": math.degrees(raan_spread_rad),
    }


# ============================================================================
# 5. Reentry & Ablation
# ============================================================================

def load_reentry_events(**kwargs) -> Any:
    """Load the reentry events dataset."""
    logger.info("Loading reentry events …")
    return get_dataset("reentry_events", **kwargs)


# Heuristic heat-of-ablation values [J/kg] for common spacecraft materials
_ABLATION_HEAT: Dict[str, float] = {
    "aluminum": 1.1e7,
    "titanium": 2.0e7,
    "steel": 1.5e7,
    "carbon_composite": 3.0e7,
    "copper": 0.9e7,
}

# Approximate fraction of kinetic energy deposited as heating
_HEATING_EFFICIENCY = 0.10


def predict_surviving_mass_fraction(
    dry_mass_kg: float,
    material: str = "aluminum",
    entry_velocity_km_s: float = 7.8,
) -> Dict[str, float]:
    """Heuristic ablation model for atmospheric re-entry survival.

    Compares the specific kinetic energy available for heating against
    the material's heat of ablation to estimate the fraction of mass
    that survives to ground impact.

    Parameters
    ----------
    dry_mass_kg : float
        Dry mass of the re-entering object [kg].
    material : str
        Material type (``"aluminum"``, ``"titanium"``, ``"steel"``,
        ``"carbon_composite"``, ``"copper"``).
    entry_velocity_km_s : float
        Entry interface velocity [km/s]. Default 7.8 km/s (typical LEO).

    Returns
    -------
    dict
        ``surviving_mass_fraction`` – fraction 0–1,
        ``surviving_mass_kg``       – estimated surviving mass [kg],
        ``ablation_energy_J``       – energy required for full ablation [J].
    """
    if material not in _ABLATION_HEAT:
        raise ValueError(
            f"Unknown material '{material}'. Choose from {list(_ABLATION_HEAT.keys())}."
        )

    q_ablation = _ABLATION_HEAT[material]  # J/kg
    v_entry = entry_velocity_km_s * 1e3     # m/s
    ke_specific = 0.5 * v_entry ** 2        # J/kg (specific KE)
    q_heating = _HEATING_EFFICIENCY * ke_specific  # J/kg deposited

    # Fraction ablated = heating / ablation energy, clamped to [0, 1]
    fraction_ablated = min(q_heating / q_ablation, 1.0)
    surviving_fraction = 1.0 - fraction_ablated
    surviving_mass = dry_mass_kg * surviving_fraction

    return {
        "surviving_mass_fraction": surviving_fraction,
        "surviving_mass_kg": surviving_mass,
        "ablation_energy_J": q_ablation * dry_mass_kg,
    }


# ============================================================================
# 6. Constellation Census & Walker Parameters
# ============================================================================

def load_constellation_census(**kwargs) -> Any:
    """Load the constellation census dataset."""
    logger.info("Loading constellation census …")
    return get_dataset("constellation_census", **kwargs)


def get_walker_delta_parameters(
    ds: Any,
    constellation_name: str,
) -> Dict[str, Any]:
    """Extract Walker-delta i:t/p/f parameters for a named constellation.

    Walker-delta notation describes a symmetric constellation:
    * *i* – inclination [°]
    * *t* – total number of satellites
    * *p* – number of orbital planes
    * *f* – phasing parameter (0 ≤ f < p)

    Parameters
    ----------
    ds : Dataset
        Constellation census dataset.
    constellation_name : str
        Case-insensitive constellation name (e.g. ``"Starlink"``).

    Returns
    -------
    dict
        ``inclination_deg``, ``total_sats``, ``planes``, ``phasing``,
        ``altitude_km``.
    """
    name_lower = constellation_name.lower()
    for row in ds:
        rname = str(
            row.get("CONSTELLATION") or row.get("constellation") or row.get("name", "")
        ).lower()
        if name_lower in rname:
            inc = row.get("INCLINATION") or row.get("inclination_deg") or row.get("inclination")
            total = row.get("TOTAL_SATS") or row.get("total_satellites") or row.get("num_satellites")
            planes = row.get("PLANES") or row.get("num_planes") or row.get("planes")
            phasing = row.get("PHASING") or row.get("phasing_parameter") or row.get("f")
            alt = row.get("ALTITUDE_KM") or row.get("altitude_km") or row.get("altitude")
            return {
                "inclination_deg": float(inc) if inc is not None else None,
                "total_sats": int(total) if total is not None else None,
                "planes": int(planes) if planes is not None else None,
                "phasing": int(phasing) if phasing is not None else 0,
                "altitude_km": float(alt) if alt is not None else None,
            }

    logger.warning("Constellation '%s' not found in census.", constellation_name)
    return {
        "inclination_deg": None,
        "total_sats": None,
        "planes": None,
        "phasing": None,
        "altitude_km": None,
    }


# ============================================================================
# 7. Starlink Fleet
# ============================================================================

def load_starlink_fleet(**kwargs) -> Any:
    """Load the Starlink fleet dataset."""
    logger.info("Loading Starlink fleet data …")
    return get_dataset("starlink_fleet", **kwargs)


def compute_shell_distribution(ds: Any) -> Dict[str, int]:
    """Count Starlink satellites per altitude shell.

    Shells are binned to the nearest 10 km (e.g. ``"540"`` for the
    540-km shell).

    Parameters
    ----------
    ds : Dataset
        Starlink fleet dataset.

    Returns
    -------
    dict
        Mapping of shell label (string) to count of satellites.
    """
    shells: Dict[str, int] = {}
    for row in ds:
        alt = row.get("ALTITUDE_KM") or row.get("altitude_km") or row.get("APOGEE") or row.get("apogee")
        if alt is None:
            continue
        try:
            alt_val = float(alt)
        except (ValueError, TypeError):
            continue
        # Round to nearest 10 km to define shell bucket
        shell_label = str(int(round(alt_val / 10.0) * 10))
        shells[shell_label] = shells.get(shell_label, 0) + 1

    # Sort by altitude
    return dict(sorted(shells.items(), key=lambda kv: int(kv[0])))


# ============================================================================
# 8. OneWeb Fleet & Polar Pass Intervals
# ============================================================================

def load_oneweb_fleet(**kwargs) -> Any:
    """Load the OneWeb fleet dataset."""
    logger.info("Loading OneWeb fleet data …")
    return get_dataset("oneweb_fleet", **kwargs)


def compute_polar_pass_intervals(
    inclination_deg: float,
    altitude_km: float,
    latitude_deg: float,
) -> Dict[str, float]:
    """Estimate pass interval for a satellite over a given latitude.

    Uses the ground-track geometry for a circular orbit at the specified
    inclination and altitude to compute how often the satellite's
    sub-satellite point crosses the given latitude.

    Parameters
    ----------
    inclination_deg : float
        Orbital inclination [°].
    altitude_km : float
        Orbital altitude above mean sea level [km].
    latitude_deg : float
        Observer latitude [°] (−90 to +90).

    Returns
    -------
    dict
        ``orbital_period_min`` – orbital period [min],
        ``pass_interval_min``  – average interval between passes [min],
        ``passes_per_day``     – number of passes per day,
        ``max_elevation_deg``  – approximate maximum elevation angle [°].

    Raises
    ------
    ValueError
        If the latitude is not reachable by the orbit (|lat| > inc).
    """
    inc_rad = math.radians(inclination_deg)
    lat_rad = math.radians(abs(latitude_deg))

    if abs(latitude_deg) > inclination_deg:
        raise ValueError(
            f"|latitude| ({abs(latitude_deg):.1f}°) exceeds inclination "
            f"({inclination_deg:.1f}°); no passes occur."
        )

    r = R_EARTH_KM + altitude_km
    T_s = 2.0 * math.pi * math.sqrt((r ** 3) / _GM_EARTH_KM3)  # period [s]
    T_min = T_s / 60.0

    # Number of ascending/descending latitude crossings per orbit: 2
    # Longitude shift per orbit due to Earth rotation
    delta_lon_per_orbit = EARTH_ROT_RAD_S * T_s  # rad

    # Node-to-node angular coverage at the given latitude
    # cos(u) = sin(lat)/sin(i) gives the argument of latitude at crossing
    cos_u = math.sin(lat_rad) / math.sin(inc_rad) if math.sin(inc_rad) > 1e-9 else 0.0
    cos_u = max(min(cos_u, 1.0), -1.0)

    # Approximate number of orbits between successive passes over the
    # same latitude band (limited swath width)
    swath_half_angle = math.asin(min(R_EARTH_KM / r, 1.0))  # nadir cone
    effective_strip = 2.0 * swath_half_angle  # rad
    n_orbits_between = max(2.0 * math.pi / (delta_lon_per_orbit + effective_strip), 1.0)

    pass_interval_min = n_orbits_between * T_min
    passes_per_day = _MIN_PER_DAY / pass_interval_min

    # Maximum elevation for a zenith pass
    rho = math.asin(R_EARTH_KM / r)  # Earth angular radius from satellite
    max_elev = 90.0 - math.degrees(rho)

    return {
        "orbital_period_min": T_min,
        "pass_interval_min": pass_interval_min,
        "passes_per_day": passes_per_day,
        "max_elevation_deg": max_elev,
    }


# ============================================================================
# 9. Kuiper Fleet & Deployment
# ============================================================================

def load_kuiper_fleet(**kwargs) -> Any:
    """Load the Kuiper fleet dataset."""
    logger.info("Loading Kuiper fleet data …")
    return get_dataset("kuiper_fleet", **kwargs)


def simulate_deployment_timeline(
    ds: Any,
    target_date_str: str,
) -> Dict[str, Any]:
    """Summarise deployment status relative to a target date.

    Parameters
    ----------
    ds : Dataset
        Kuiper fleet dataset with launch/deployment date columns.
    target_date_str : str
        ISO-format date string (``"YYYY-MM-DD"``).

    Returns
    -------
    dict
        ``total_planned``, ``deployed_by_target``, ``remaining``,
        ``completion_pct``.
    """
    target_dt = datetime.fromisoformat(target_date_str).replace(tzinfo=timezone.utc)
    total = 0
    deployed = 0

    for row in ds:
        total += 1
        launch_str = (
            row.get("LAUNCH_DATE") or row.get("launch_date") or row.get("deploy_date")
        )
        if launch_str is None:
            continue
        try:
            launch_dt = datetime.fromisoformat(str(launch_str).replace("Z", "+00:00"))
            if launch_dt.tzinfo is None:
                launch_dt = launch_dt.replace(tzinfo=timezone.utc)
            if launch_dt <= target_dt:
                deployed += 1
        except (ValueError, TypeError):
            continue

    remaining = total - deployed
    pct = (deployed / total * 100.0) if total > 0 else 0.0

    return {
        "total_planned": total,
        "deployed_by_target": deployed,
        "remaining": remaining,
        "completion_pct": round(pct, 2),
    }


# ============================================================================
# 10. Globalstar Fleet & Bent-Pipe Coverage
# ============================================================================

def load_globalstar_fleet(**kwargs) -> Any:
    """Load the Globalstar fleet dataset."""
    logger.info("Loading Globalstar fleet data …")
    return get_dataset("globalstar_fleet", **kwargs)


def compute_bent_pipe_coverage(
    lat: float,
    lon: float,
    altitude_km: float,
    min_elev_deg: float = 10.0,
) -> Dict[str, float]:
    """Compute the ground footprint radius for a bent-pipe transponder.

    The footprint is defined by the minimum elevation angle at which a
    ground terminal can communicate with the satellite.

    Parameters
    ----------
    lat : float
        Sub-satellite latitude [°] (informational).
    lon : float
        Sub-satellite longitude [°] (informational).
    altitude_km : float
        Satellite altitude [km].
    min_elev_deg : float, optional
        Minimum ground elevation angle [°]. Default 10°.

    Returns
    -------
    dict
        ``footprint_radius_km``  – great-circle footprint radius [km],
        ``footprint_area_km2``   – circular footprint area [km²],
        ``earth_central_angle_deg`` – half-cone angle at Earth centre [°],
        ``slant_range_km``       – maximum slant range to edge [km].
    """
    elev_rad = math.radians(min_elev_deg)
    r_e = R_EARTH_KM
    r_s = r_e + altitude_km

    # Earth central angle from sub-satellite point to footprint edge
    # sin(η) / sin(ε + η) = R_E / r_s  where ε = elevation, η = nadir angle
    # Alternatively: η = arccos(R_E cos(ε) / r_s) - ε
    cos_eta = r_e * math.cos(elev_rad) / r_s
    eta = math.acos(min(cos_eta, 1.0)) - elev_rad  # nadir angle [rad]

    # Earth central angle λ = π/2 - ε - η  (complementary geometry)
    earth_central_rad = math.pi / 2.0 - elev_rad - eta
    if earth_central_rad < 0:
        earth_central_rad = 0.0

    footprint_radius_km = earth_central_rad * r_e  # arc length on surface
    footprint_area_km2 = math.pi * footprint_radius_km ** 2

    # Slant range (law of sines)
    slant_range_km = r_s * math.sin(earth_central_rad) / math.cos(elev_rad)

    return {
        "footprint_radius_km": footprint_radius_km,
        "footprint_area_km2": footprint_area_km2,
        "earth_central_angle_deg": math.degrees(earth_central_rad),
        "slant_range_km": abs(slant_range_km),
    }


# ============================================================================
# 11. UCS Satellite Database
# ============================================================================

def load_ucs_satellite_database(**kwargs) -> Any:
    """Load the Union of Concerned Scientists satellite database."""
    logger.info("Loading UCS satellite database …")
    return get_dataset("ucs_satellites", **kwargs)


def filter_by_propulsion_and_power(
    ds: Any,
    prop_type: str = "electric",
    min_power_w: float = 0.0,
) -> Any:
    """Filter UCS dataset by propulsion type and minimum power.

    Parameters
    ----------
    ds : Dataset
        UCS satellite database.
    prop_type : str
        Propulsion type substring to match (case-insensitive).
        E.g. ``"electric"``, ``"chemical"``, ``"cold gas"``.
    min_power_w : float
        Minimum on-board power [W].

    Returns
    -------
    Dataset
        Filtered subset.
    """
    prop_lower = prop_type.lower()

    def _filter(row: Dict[str, Any]) -> bool:
        propulsion = str(
            row.get("PROPULSION") or row.get("propulsion") or row.get("Propulsion_Type", "")
        ).lower()
        power_val = row.get("POWER_W") or row.get("power_watts") or row.get("Power_watts") or 0
        try:
            power_float = float(power_val)
        except (ValueError, TypeError):
            power_float = 0.0
        return prop_lower in propulsion and power_float >= min_power_w

    return ds.filter(_filter)


# ============================================================================
# 12. LEOMSAT – Multispectral Orbit Suitability
# ============================================================================

def load_leomsat_dataset(**kwargs) -> Any:
    """Load the LEOMSAT dataset."""
    logger.info("Loading LEOMSAT dataset …")
    return get_dataset("leomsat", **kwargs)


def evaluate_multispectral_orbit_suitability(
    altitude_km: float,
    inclination_deg: float,
    swath_km: float,
    focal_length_m: float = 1.0,
    pixel_pitch_um: float = 10.0,
) -> Dict[str, float]:
    """Evaluate orbit suitability for multispectral Earth observation.

    Computes ground sample distance (GSD) and approximate revisit time
    based on orbit geometry.

    Parameters
    ----------
    altitude_km : float
        Orbital altitude [km].
    inclination_deg : float
        Orbital inclination [°].
    swath_km : float
        Instrument swath width on ground [km].
    focal_length_m : float, optional
        Telescope focal length [m]. Default 1.0 m.
    pixel_pitch_um : float, optional
        Detector pixel pitch [µm]. Default 10.0 µm.

    Returns
    -------
    dict
        ``gsd_m``           – ground sample distance [m],
        ``revisit_days``    – approximate nadir revisit time [days],
        ``coverage_rate_km2_per_day`` – daily ground coverage rate [km²/day],
        ``sun_sync``        – True if inclination ≈ sun-sync for this altitude.
    """
    h_m = altitude_km * 1e3
    pixel_pitch_m = pixel_pitch_um * 1e-6

    # GSD = (pixel pitch / focal length) × altitude
    gsd_m = (pixel_pitch_m / focal_length_m) * h_m

    # Orbital period
    r = R_EARTH_KM + altitude_km
    T_s = 2.0 * math.pi * math.sqrt((r ** 3) / _GM_EARTH_KM3)
    orbits_per_day = DAY_S / T_s

    # Ground track spacing at equator
    ground_track_spacing_km = 2.0 * math.pi * R_EARTH_KM / orbits_per_day

    # Revisit time ≈ ground-track spacing / swath width (orbits)
    if swath_km > 0:
        revisit_orbits = max(ground_track_spacing_km / swath_km, 1.0)
        revisit_days = revisit_orbits / orbits_per_day
    else:
        revisit_days = float("inf")

    # Daily coverage rate
    v_ground = 2.0 * math.pi * R_EARTH_KM / (T_s / 60.0)  # km/min
    coverage_km2_day = v_ground * swath_km * _MIN_PER_DAY * 0.5  # ~50% daylight

    # Sun-synchronous inclination check (J2 secular: Ω_dot = -360°/yr)
    n = 2.0 * math.pi / T_s
    # Required cos(i) for sun-sync: cos(i) = -2 Ω_dot_ss / (3 n J2 (Re/a)^2)
    omega_dot_ss = 2.0 * math.pi / YEAR_S  # rad/s (~0.9856°/day)
    cos_i_ss = -omega_dot_ss / (1.5 * n * J2_EARTH * (R_EARTH_KM / r) ** 2)
    cos_i_ss = max(min(cos_i_ss, 1.0), -1.0)
    i_ss_deg = math.degrees(math.acos(cos_i_ss))
    is_sun_sync = abs(inclination_deg - i_ss_deg) < 2.0

    return {
        "gsd_m": gsd_m,
        "revisit_days": revisit_days,
        "coverage_rate_km2_per_day": coverage_km2_day,
        "sun_sync": is_sun_sync,
    }


# ============================================================================
# 13. HET Low-Thrust Transfer (Edelbaum)
# ============================================================================

def load_het_transfer_data(**kwargs) -> Any:
    """Load HET (Hall Effect Thruster) transfer efficiency dataset."""
    logger.info("Loading HET transfer data …")
    return get_dataset("het_transfer_efficiency", **kwargs)


def compute_low_thrust_transfer(
    isp_s: float,
    thrust_n: float,
    mass_initial_kg: float,
    r1_km: float,
    r2_km: float,
    gm: float = GM_EARTH,
) -> Dict[str, float]:
    """Edelbaum low-thrust coplanar transfer Δv and fuel budget.

    For a coplanar circle-to-circle low-thrust spiral the Edelbaum
    approximation gives:

        Δv = |v_circ1 − v_circ2|

    and the Tsiolkovsky equation determines propellant mass.

    Parameters
    ----------
    isp_s : float
        Specific impulse of the thruster [s].
    thrust_n : float
        Continuous thrust level [N].
    mass_initial_kg : float
        Initial spacecraft mass (wet) [kg].
    r1_km : float
        Initial circular orbit radius [km].
    r2_km : float
        Target circular orbit radius [km].
    gm : float, optional
        Gravitational parameter [m³/s²]. Default Earth.

    Returns
    -------
    dict
        ``delta_v_m_s``      – required Δv [m/s],
        ``fuel_mass_kg``     – propellant consumed [kg],
        ``mass_final_kg``    – final mass [kg],
        ``transfer_time_s``  – approximate transfer time [s],
        ``transfer_time_days`` – transfer time [days].
    """
    r1_m = r1_km * 1e3
    r2_m = r2_km * 1e3

    v1 = math.sqrt(gm / r1_m)
    v2 = math.sqrt(gm / r2_m)
    dv = abs(v1 - v2)

    # Tsiolkovsky rocket equation
    ve = G0 * isp_s
    mass_ratio = math.exp(dv / ve)
    mass_final = mass_initial_kg / mass_ratio
    fuel_mass = mass_initial_kg - mass_final

    # Transfer time (continuous thrust)
    # T ≈ m_initial * Δv / F  (constant-thrust approximation)
    if thrust_n > 0:
        t_transfer = mass_initial_kg * dv / thrust_n  # rough estimate
    else:
        t_transfer = float("inf")

    return {
        "delta_v_m_s": dv,
        "fuel_mass_kg": fuel_mass,
        "mass_final_kg": mass_final,
        "transfer_time_s": t_transfer,
        "transfer_time_days": t_transfer / DAY_S,
    }


# ============================================================================
# 14. Hohmann Transfer
# ============================================================================

def load_hektor_hohmann_data(**kwargs) -> Any:
    """Load the 624-Hektor Hohmann transfer dataset."""
    logger.info("Loading Hektor Hohmann transfer data …")
    return get_dataset("hektor_hohmann_transfer", **kwargs)


def calculate_hohmann_transfer_dv(
    r1_km: float,
    r2_km: float,
    gm: float = GM_SUN,
) -> Dict[str, float]:
    """Classic two-impulse Hohmann transfer between coplanar circular orbits.

    Parameters
    ----------
    r1_km : float
        Radius of the departure circular orbit [km].
    r2_km : float
        Radius of the arrival circular orbit [km].
    gm : float, optional
        Gravitational parameter [m³/s²]. Default solar.

    Returns
    -------
    dict
        ``dv1_m_s``      – departure impulse [m/s],
        ``dv2_m_s``      – arrival impulse [m/s],
        ``dv_total_m_s`` – total Δv [m/s],
        ``transfer_time_s``   – half-period of transfer ellipse [s],
        ``transfer_time_days``– transfer time [days].
    """
    r1 = r1_km * 1e3  # m
    r2 = r2_km * 1e3  # m

    # Circular velocities
    v_circ1 = math.sqrt(gm / r1)
    v_circ2 = math.sqrt(gm / r2)

    # Transfer ellipse semi-major axis
    a_t = (r1 + r2) / 2.0

    # Vis-viva at departure and arrival points on the transfer ellipse
    v_t1 = math.sqrt(gm * (2.0 / r1 - 1.0 / a_t))
    v_t2 = math.sqrt(gm * (2.0 / r2 - 1.0 / a_t))

    dv1 = abs(v_t1 - v_circ1)
    dv2 = abs(v_circ2 - v_t2)

    # Transfer time = half the period of the transfer ellipse
    t_transfer = math.pi * math.sqrt(a_t ** 3 / gm)

    return {
        "dv1_m_s": dv1,
        "dv2_m_s": dv2,
        "dv_total_m_s": dv1 + dv2,
        "transfer_time_s": t_transfer,
        "transfer_time_days": t_transfer / DAY_S,
    }


# ============================================================================
# 15. Constrained Rendezvous – Clohessy-Wiltshire (Hill's) Equations
# ============================================================================

def load_orbit_planner_benchmarks(**kwargs) -> Any:
    """Load orbit planner benchmark cases."""
    logger.info("Loading orbit planner benchmarks …")
    return get_dataset("orbit_planner", **kwargs)


def plan_constrained_rendezvous(
    dx: float,
    dy: float,
    dz: float,
    dvx: float,
    dvy: float,
    dvz: float,
    n_mean_motion: float,
    dt_s: float,
) -> Dict[str, NDArray[np.float64]]:
    """Clohessy-Wiltshire (CW) / Hill state transition matrix.

    Propagates relative motion in the LVLH frame (x = radial,
    y = along-track, z = cross-track) assuming a near-circular
    reference orbit.

    The CW state transition matrix Φ(t) maps the initial relative
    state [δr, δv]₀ to the state at time *t*.

    Parameters
    ----------
    dx, dy, dz : float
        Initial relative position offsets [m] (radial, along-track,
        cross-track).
    dvx, dvy, dvz : float
        Initial relative velocity offsets [m/s].
    n_mean_motion : float
        Mean motion of the reference orbit [rad/s].
    dt_s : float
        Propagation time [s].

    Returns
    -------
    dict
        ``position_m``  – (3,) final relative position [m],
        ``velocity_m_s`` – (3,) final relative velocity [m/s],
        ``stm``         – (6, 6) CW state transition matrix.
    """
    n = n_mean_motion
    t = dt_s
    nt = n * t
    c = math.cos(nt)
    s = math.sin(nt)

    # CW state transition matrix Φ(t)
    # rows: [x, y, z, vx, vy, vz]
    stm = np.zeros((6, 6), dtype=np.float64)

    # Position – position block
    stm[0, 0] = 4.0 - 3.0 * c
    stm[0, 1] = 0.0
    stm[0, 2] = 0.0
    stm[1, 0] = 6.0 * (s - nt)
    stm[1, 1] = 1.0
    stm[1, 2] = 0.0
    stm[2, 0] = 0.0
    stm[2, 1] = 0.0
    stm[2, 2] = c

    # Position – velocity block
    stm[0, 3] = s / n
    stm[0, 4] = 2.0 * (1.0 - c) / n
    stm[0, 5] = 0.0
    stm[1, 3] = 2.0 * (c - 1.0) / n
    stm[1, 4] = (4.0 * s - 3.0 * nt) / n
    stm[1, 5] = 0.0
    stm[2, 3] = 0.0
    stm[2, 4] = 0.0
    stm[2, 5] = s / n

    # Velocity – position block
    stm[3, 0] = 3.0 * n * s
    stm[3, 1] = 0.0
    stm[3, 2] = 0.0
    stm[4, 0] = 6.0 * n * (c - 1.0)
    stm[4, 1] = 0.0
    stm[4, 2] = 0.0
    stm[5, 0] = 0.0
    stm[5, 1] = 0.0
    stm[5, 2] = -n * s

    # Velocity – velocity block
    stm[3, 3] = c
    stm[3, 4] = 2.0 * s
    stm[3, 5] = 0.0
    stm[4, 3] = -2.0 * s
    stm[4, 4] = 4.0 * c - 3.0
    stm[4, 5] = 0.0
    stm[5, 3] = 0.0
    stm[5, 4] = 0.0
    stm[5, 5] = c

    state0 = np.array([dx, dy, dz, dvx, dvy, dvz], dtype=np.float64)
    state_f = stm @ state0

    return {
        "position_m": state_f[:3],
        "velocity_m_s": state_f[3:],
        "stm": stm,
    }


# ============================================================================
# 16. OrbitBench – Solver Benchmarking
# ============================================================================

def load_orbitbench(**kwargs) -> Any:
    """Load the OrbitBench dataset."""
    logger.info("Loading OrbitBench dataset …")
    return get_dataset("orbitbench", **kwargs)


def benchmark_orbit_solver(
    solver_fn: Callable[..., Dict[str, Any]],
    test_cases: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score an orbit-determination solver against test cases.

    Each test case must contain ``"inputs"`` (dict passed to
    *solver_fn*) and ``"expected"`` (dict with ``"position_km"`` and/or
    ``"velocity_km_s"`` arrays).

    Parameters
    ----------
    solver_fn : Callable
        A solver function that accepts keyword arguments from
        ``case["inputs"]`` and returns a dict with ``"position_km"``
        and/or ``"velocity_km_s"`` arrays.
    test_cases : Sequence[dict]
        List of test case dicts with ``"inputs"`` and ``"expected"``
        keys.

    Returns
    -------
    dict
        ``n_cases``            – number of test cases,
        ``mean_pos_error_km``  – mean position error [km],
        ``mean_vel_error_km_s``– mean velocity error [km/s],
        ``max_pos_error_km``   – worst-case position error [km],
        ``pass_rate``          – fraction within 1 km position tolerance.
    """
    pos_errors: List[float] = []
    vel_errors: List[float] = []
    passed = 0

    for case in test_cases:
        try:
            result = solver_fn(**case["inputs"])
        except Exception as exc:
            logger.warning("Solver raised %s on case %s", exc, case.get("id", "?"))
            continue

        expected = case["expected"]

        if "position_km" in expected and "position_km" in result:
            dp = np.linalg.norm(
                np.asarray(result["position_km"]) - np.asarray(expected["position_km"])
            )
            pos_errors.append(float(dp))
            if dp < 1.0:
                passed += 1

        if "velocity_km_s" in expected and "velocity_km_s" in result:
            dv = np.linalg.norm(
                np.asarray(result["velocity_km_s"]) - np.asarray(expected["velocity_km_s"])
            )
            vel_errors.append(float(dv))

    n = len(test_cases)
    return {
        "n_cases": n,
        "mean_pos_error_km": float(np.mean(pos_errors)) if pos_errors else float("nan"),
        "mean_vel_error_km_s": float(np.mean(vel_errors)) if vel_errors else float("nan"),
        "max_pos_error_km": float(np.max(pos_errors)) if pos_errors else float("nan"),
        "pass_rate": passed / n if n > 0 else 0.0,
    }


# ============================================================================
# 17. OrbitWren – Flight Software Workload Profiling
# ============================================================================

def load_orbitwren_tracker(**kwargs) -> Any:
    """Load the OrbitWren orbital-compute tracker dataset."""
    logger.info("Loading OrbitWren compute tracker …")
    return get_dataset("orbitwren_compute_tracker", **kwargs)


def profile_flight_software_workload(
    flops_budget: float,
    ops_per_determination: float,
    determination_rate_hz: float = 1.0,
    margin_pct: float = 20.0,
) -> Dict[str, float]:
    """Estimate flight software computational workload vs. budget.

    Parameters
    ----------
    flops_budget : float
        Available on-board FLOPS [FLOP/s].
    ops_per_determination : float
        Floating-point operations per single orbit-determination cycle.
    determination_rate_hz : float, optional
        How often OD runs per second [Hz]. Default 1.0 Hz.
    margin_pct : float, optional
        Required design margin [%]. Default 20 %.

    Returns
    -------
    dict
        ``utilisation_pct``      – processor utilisation [%],
        ``margin_available_pct`` – remaining margin after OD [%],
        ``within_budget``        – boolean,
        ``max_rate_hz``          – maximum sustainable OD rate [Hz].
    """
    required_flops = ops_per_determination * determination_rate_hz
    utilisation = required_flops / flops_budget * 100.0 if flops_budget > 0 else float("inf")
    margin_avail = 100.0 - utilisation
    within = margin_avail >= margin_pct
    max_rate = flops_budget / ops_per_determination if ops_per_determination > 0 else float("inf")

    return {
        "utilisation_pct": utilisation,
        "margin_available_pct": margin_avail,
        "within_budget": within,
        "max_rate_hz": max_rate,
    }


# ============================================================================
# 18. Orbital Chaos – Lyapunov Exponent
# ============================================================================

def load_orbital_chaos_dataset(**kwargs) -> Any:
    """Load the orbital chaos (NASA SSC) dataset."""
    logger.info("Loading orbital chaos dataset …")
    return get_dataset("orbital_chaos_ssc", **kwargs)


def compute_lyapunov_exponent(
    trajectory: NDArray[np.float64],
    dt: float,
) -> Dict[str, float]:
    """Estimate the maximum Lyapunov exponent from trajectory data.

    Uses the Rosenstein (1993) algorithm: for each point on the
    trajectory, find its nearest neighbour in phase space, track
    the divergence over time, and average the logarithmic growth
    rate.

    Parameters
    ----------
    trajectory : ndarray, shape (N, D)
        Phase-space trajectory with *N* time steps and *D* dimensions
        (e.g. D = 6 for [x, y, z, vx, vy, vz]).
    dt : float
        Time step between consecutive samples [s].

    Returns
    -------
    dict
        ``lambda_max``       – maximum Lyapunov exponent [1/s],
        ``lyapunov_time_s``  – e-folding time 1/λ [s],
        ``is_chaotic``       – True if λ > 0 within numerical noise.
    """
    trajectory = np.atleast_2d(trajectory)
    N, D = trajectory.shape
    if N < 10:
        raise ValueError("Trajectory too short; need at least 10 samples.")

    # Mean separation threshold to avoid temporal neighbours
    mean_period_samples = max(N // 10, 2)

    # Find nearest neighbours excluding temporal neighbours
    divergences: List[float] = []
    max_evolve = min(N // 4, 500)

    for i in range(N - max_evolve):
        # Distance from point i to all other points
        diffs = trajectory[i] - trajectory
        dists = np.linalg.norm(diffs, axis=1)
        # Exclude self and temporal neighbours
        dists[max(0, i - mean_period_samples): min(N, i + mean_period_samples + 1)] = np.inf

        j = int(np.argmin(dists))
        d0 = dists[j]
        if d0 < 1e-15:
            continue

        # Track divergence
        log_divs: List[float] = []
        for k in range(1, max_evolve):
            if i + k >= N or j + k >= N:
                break
            dk = np.linalg.norm(trajectory[i + k] - trajectory[j + k])
            if dk > 1e-15:
                log_divs.append(math.log(dk / d0))

        if log_divs:
            # Average log divergence at the end of the tracking window
            divergences.append(log_divs[-1] / (len(log_divs) * dt))

    if not divergences:
        return {
            "lambda_max": 0.0,
            "lyapunov_time_s": float("inf"),
            "is_chaotic": False,
        }

    lambda_max = float(np.median(divergences))
    lyap_time = 1.0 / abs(lambda_max) if abs(lambda_max) > 1e-15 else float("inf")

    return {
        "lambda_max": lambda_max,
        "lyapunov_time_s": lyap_time,
        "is_chaotic": lambda_max > 1e-10,
    }
