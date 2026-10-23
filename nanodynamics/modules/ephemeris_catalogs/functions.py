"""
nanodynamics/modules/ephemeris_catalogs/functions.py

Pillar 1: Solar System Bodies, Asteroids & Ephemeris Catalogs.

Provides dataset loaders and analytical functions for small-body databases,
asteroid taxonomy, comet dynamics, planetary metadata, natural satellites,
Gaia SSO epoch astrometry, solar surface meshes, globular cluster potentials,
and hypervelocity-star galactic orbit tracing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nanodynamics.core.constants import *
from nanodynamics.data_ingestion.loaders import get_dataset

logger = logging.getLogger("nanodynamics.ephemeris_catalogs")

__all__ = [
    # ── Loaders ──────────────────────────────────────────────────────────
    "load_jpl_small_bodies",
    "load_sdss_asteroid_taxonomy",
    "load_comet_catalog",
    "load_solar_system_metadata",
    "load_solar_system_moons",
    "load_gaia_dr3_sso",
    "load_solar_3d_mesh",
    "load_globular_clusters",
    "load_hypervelocity_stars",
    # ── JPL Small Bodies ─────────────────────────────────────────────────
    "filter_neo_hazards",
    "propagate_keplerian_orbit",
    # ── SDSS Asteroid Taxonomy ───────────────────────────────────────────
    "classify_spectral_type",
    "estimate_albedo_density",
    # ── Comet Catalog ────────────────────────────────────────────────────
    "compute_comet_outgassing_acceleration",
    "predict_tail_vector",
    # ── Solar System Metadata ────────────────────────────────────────────
    "get_planetary_constants",
    # ── Solar System Moons ───────────────────────────────────────────────
    "compute_hill_sphere",
    # ── Gaia DR3 SSO ─────────────────────────────────────────────────────
    "extract_astrometric_residuals",
    # ── Solar 3D ─────────────────────────────────────────────────────────
    "generate_solar_surface_grid",
    # ── Globular Clusters ────────────────────────────────────────────────
    "compute_cluster_potential_field",
    # ── Hypervelocity Stars ──────────────────────────────────────────────
    "trace_galactic_origin",
]


# ============================================================================
# 1.  JPL Small Body Database
# ============================================================================

def load_jpl_small_bodies(**kwargs):
    """Load the JPL Small Body Database via the dataset registry.

    Returns
    -------
    Dataset or IterableDataset
        Hugging Face dataset of JPL small-body orbital elements and physical
        parameters.
    """
    logger.info("Loading JPL Small Body Database …")
    return get_dataset("jpl_small_bodies", **kwargs)


def filter_neo_hazards(ds, min_diameter_km: float, max_moid_au: float):
    """Filter Near-Earth Objects / Potentially Hazardous Asteroids.

    Selects objects whose estimated diameter ≥ *min_diameter_km* AND whose
    minimum-orbit intersection distance (MOID) ≤ *max_moid_au*.

    Parameters
    ----------
    ds : Dataset
        JPL small-body dataset (expects columns ``diameter`` and ``moid_au``
        or ``moid``).
    min_diameter_km : float
        Minimum object diameter threshold [km].
    max_moid_au : float
        Maximum MOID threshold [AU].

    Returns
    -------
    Dataset
        Filtered subset of *ds*.
    """
    logger.info(
        "Filtering NEO hazards: diameter >= %.3f km, MOID <= %.4f AU",
        min_diameter_km,
        max_moid_au,
    )

    # Determine the MOID column name (datasets vary)
    moid_col = "moid_au" if "moid_au" in ds.column_names else "moid"
    diam_col = "diameter"

    def _keep(row):
        diam = row.get(diam_col)
        moid = row.get(moid_col)
        if diam is None or moid is None:
            return False
        try:
            return float(diam) >= min_diameter_km and float(moid) <= max_moid_au
        except (TypeError, ValueError):
            return False

    filtered = ds.filter(_keep)
    logger.info("Retained %d objects after NEO hazard filter.", len(filtered))
    return filtered


def propagate_keplerian_orbit(
    a_au: float,
    e: float,
    i_deg: float,
    om_deg: float,
    w_deg: float,
    ma_deg: float,
    epoch_jd: float,
    target_jd: float,
    gm: float = GM_SUN,
) -> np.ndarray:
    """Propagate a two-body Keplerian orbit from osculating elements.

    Solves Kepler's equation (M → E) via Newton–Raphson iteration, then
    converts from perifocal to the ecliptic/ECI frame.

    Parameters
    ----------
    a_au : float
        Semi-major axis [AU].
    e : float
        Eccentricity (0 ≤ e < 1 for elliptic orbits).
    i_deg : float
        Inclination [deg].
    om_deg : float
        Longitude of ascending node Ω [deg].
    w_deg : float
        Argument of perihelion ω [deg].
    ma_deg : float
        Mean anomaly at *epoch_jd* [deg].
    epoch_jd : float
        Julian Date of the element epoch.
    target_jd : float
        Julian Date at which to evaluate the state.
    gm : float, optional
        Gravitational parameter μ [m³ s⁻²].  Defaults to ``GM_SUN``.

    Returns
    -------
    np.ndarray, shape (6,)
        Cartesian state ``[x, y, z, vx, vy, vz]`` in [m] and [m/s].
    """
    # ── unit conversions ──────────────────────────────────────────────────
    a = a_au * AU_M                          # semi-major axis [m]
    i = np.radians(i_deg)
    om = np.radians(om_deg)
    w = np.radians(w_deg)
    M0 = np.radians(ma_deg)

    # ── propagate mean anomaly ────────────────────────────────────────────
    n = np.sqrt(gm / a**3)                  # mean motion [rad/s]
    dt = (target_jd - epoch_jd) * DAY_S     # elapsed time [s]
    M = (M0 + n * dt) % (2.0 * np.pi)

    # ── solve Kepler's equation  M = E - e sin E  (Newton–Raphson) ────────
    E = M + e * np.sin(M)                    # initial guess
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < 1e-14:
            break

    # ── true anomaly ──────────────────────────────────────────────────────
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )

    # ── distance and velocity magnitude in perifocal frame ────────────────
    r_mag = a * (1.0 - e * np.cos(E))
    p = a * (1.0 - e**2)                    # semi-latus rectum

    # Perifocal position & velocity
    r_pf = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
    v_pf = np.sqrt(gm / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # ── rotation matrix  perifocal → ECI ──────────────────────────────────
    cos_om, sin_om = np.cos(om), np.sin(om)
    cos_w, sin_w = np.cos(w), np.sin(w)
    cos_i, sin_i = np.cos(i), np.sin(i)

    R = np.array([
        [
            cos_om * cos_w - sin_om * sin_w * cos_i,
            -cos_om * sin_w - sin_om * cos_w * cos_i,
            sin_om * sin_i,
        ],
        [
            sin_om * cos_w + cos_om * sin_w * cos_i,
            -sin_om * sin_w + cos_om * cos_w * cos_i,
            -cos_om * sin_i,
        ],
        [
            sin_w * sin_i,
            cos_w * sin_i,
            cos_i,
        ],
    ])

    r_eci = R @ r_pf
    v_eci = R @ v_pf

    state = np.concatenate([r_eci, v_eci])
    logger.debug(
        "Keplerian propagation: a=%.4f AU, e=%.6f → |r|=%.3e m, |v|=%.3e m/s",
        a_au,
        e,
        np.linalg.norm(r_eci),
        np.linalg.norm(v_eci),
    )
    return state


# ============================================================================
# 2.  SDSS Asteroid Taxonomy
# ============================================================================

def load_sdss_asteroid_taxonomy(**kwargs):
    """Load the SDSS Moving-Object Catalog asteroid taxonomy dataset.

    Returns
    -------
    Dataset or IterableDataset
        SDSS photometric color data for taxonomic classification.
    """
    logger.info("Loading SDSS asteroid taxonomy …")
    return get_dataset("sdss_asteroid_taxonomy", **kwargs)


# Rough decision-boundary colour indices derived from Bus–DeMeo / Tholen
# taxonomy mapped to SDSS ugriz bands (Ivezić et al. 2001; Parker et al. 2008).
_SPECTRAL_RULES: List[Tuple[str, callable]] = []


def classify_spectral_type(
    u_minus_g: float,
    g_minus_r: float,
    r_minus_i: float,
) -> str:
    """Classify asteroid spectral type from SDSS color indices.

    Uses simplified decision boundaries inspired by Bus–DeMeo / Tholen
    taxonomic mapping to the SDSS ugriz photometric system (Ivezić et al.
    2001).

    Parameters
    ----------
    u_minus_g : float
        (u − g) colour index [mag].
    g_minus_r : float
        (g − r) colour index [mag].
    r_minus_i : float
        (r − i) colour index [mag].

    Returns
    -------
    str
        Taxonomic class label: ``"C"``, ``"S"``, ``"V"``, or ``"X"``.
    """
    # V-types: very steep spectral slope in g−r and deep absorption in r−i
    if g_minus_r > 0.55 and r_minus_i > 0.15:
        return "V"

    # S-types: moderate reddening in g−r, slight red in r−i
    if g_minus_r > 0.45 and r_minus_i > 0.0:
        return "S"

    # C-types: near-solar colours – relatively flat / slightly blue
    if g_minus_r < 0.45 and r_minus_i < 0.10:
        return "C"

    # X-types: spectrally degenerate / featureless moderate slope
    return "X"


# Physical priors per spectral class (geometric albedo, bulk density [kg/m³])
_ALBEDO_DENSITY_PRIORS: Dict[str, Dict[str, float]] = {
    "C": {"albedo": 0.06, "density_kg_m3": 1300.0},
    "S": {"albedo": 0.20, "density_kg_m3": 2700.0},
    "V": {"albedo": 0.40, "density_kg_m3": 3400.0},
    "X": {"albedo": 0.10, "density_kg_m3": 2000.0},
}


def estimate_albedo_density(spectral_class: str) -> Dict[str, float]:
    """Return geometric-albedo and bulk-density priors for a spectral class.

    Parameters
    ----------
    spectral_class : str
        One of ``"C"``, ``"S"``, ``"V"``, ``"X"``.

    Returns
    -------
    dict
        ``{"albedo": float, "density_kg_m3": float}``

    Raises
    ------
    KeyError
        If *spectral_class* is not recognised.
    """
    key = spectral_class.upper().strip()
    if key not in _ALBEDO_DENSITY_PRIORS:
        raise KeyError(
            f"Unknown spectral class '{spectral_class}'. "
            f"Supported: {list(_ALBEDO_DENSITY_PRIORS)}"
        )
    return dict(_ALBEDO_DENSITY_PRIORS[key])


# ============================================================================
# 3.  Comet Catalog
# ============================================================================

def load_comet_catalog(**kwargs):
    """Load the comet orbital-element catalog.

    Returns
    -------
    Dataset or IterableDataset
        Comet catalog with orbital elements and non-gravitational parameters.
    """
    logger.info("Loading comet catalog …")
    return get_dataset("comet_catalog", **kwargs)


def compute_comet_outgassing_acceleration(
    perihelion_dist_au: float,
    current_dist_au: float,
    A1: Optional[float] = None,
    A2: Optional[float] = None,
) -> Dict[str, float]:
    """Compute non-gravitational acceleration using the Marsden model.

    The standard Marsden–Sekanina g(r) function models water-ice sublimation:

        g(r) = α (r / r₀)^{-m} [1 + (r / r₀)^n]^{-k}

    with canonical parameters α = 0.111262, r₀ = 2.808 AU, m = 2.15,
    n = 5.093, k = 4.6142.

    Parameters
    ----------
    perihelion_dist_au : float
        Perihelion distance [AU] (informational context).
    current_dist_au : float
        Current heliocentric distance [AU] at which to evaluate g(r).
    A1 : float or None
        Radial non-gravitational parameter [AU day⁻²].  If *None*, uses
        a generic default of 1e-8.
    A2 : float or None
        Transverse non-gravitational parameter [AU day⁻²].  If *None*,
        uses a generic default of 1e-9.

    Returns
    -------
    dict
        ``{"a_radial_m_s2": float, "a_transverse_m_s2": float, "g_r": float}``
    """
    if A1 is None:
        A1 = 1.0e-8  # AU/day²
    if A2 is None:
        A2 = 1.0e-9

    # Marsden canonical parameters (water-ice sublimation)
    alpha = 0.111262
    r0 = 2.808        # AU
    m_exp = 2.15
    n_exp = 5.093
    k_exp = 4.6142

    rr = current_dist_au / r0
    g_r = alpha * rr**(-m_exp) * (1.0 + rr**n_exp) ** (-k_exp)

    # Convert A1, A2 from AU/day² → m/s²
    au_day2_to_m_s2 = AU_M / (DAY_S**2)
    a_radial = A1 * g_r * au_day2_to_m_s2
    a_transverse = A2 * g_r * au_day2_to_m_s2

    logger.debug(
        "Comet outgassing at r=%.3f AU: g(r)=%.6e, a_R=%.4e m/s², a_T=%.4e m/s²",
        current_dist_au,
        g_r,
        a_radial,
        a_transverse,
    )

    return {
        "a_radial_m_s2": a_radial,
        "a_transverse_m_s2": a_transverse,
        "g_r": g_r,
    }


def predict_tail_vector(
    comet_pos_3d: np.ndarray,
    sun_pos_3d: np.ndarray,
) -> np.ndarray:
    """Predict the ion-tail direction vector (anti-solar).

    The type-I (ion/plasma) tail points approximately in the anti-sunward
    direction.  This function returns the unit vector from the Sun through
    the comet, extended away from the Sun.

    Parameters
    ----------
    comet_pos_3d : array_like, shape (3,)
        Comet position [m] in a heliocentric frame.
    sun_pos_3d : array_like, shape (3,)
        Sun position [m] in the same frame (often ``[0, 0, 0]``).

    Returns
    -------
    np.ndarray, shape (3,)
        Unit vector in the anti-solar (tail) direction.
    """
    comet_pos_3d = np.asarray(comet_pos_3d, dtype=np.float64)
    sun_pos_3d = np.asarray(sun_pos_3d, dtype=np.float64)

    delta = comet_pos_3d - sun_pos_3d
    norm = np.linalg.norm(delta)
    if norm < 1.0:
        logger.warning("Comet–Sun distance is < 1 m; tail vector undefined.")
        return np.array([0.0, 0.0, 0.0])

    tail_unit = delta / norm
    return tail_unit


# ============================================================================
# 4.  Solar System Metadata
# ============================================================================

def load_solar_system_metadata(**kwargs):
    """Load the solar-system planetary metadata dataset.

    Returns
    -------
    Dataset or IterableDataset
        Wikipedia-derived planet and dwarf-planet metadata.
    """
    logger.info("Loading solar system metadata …")
    return get_dataset("solar_system_metadata", **kwargs)


# Precomputed planetary constants table (IAU 2015 / DE440 reference values).
_PLANETARY_CONSTANTS: Dict[str, Dict[str, float]] = {
    "mercury": {
        "gm_m3_s2": GM_MERCURY,
        "radius_km": R_MERCURY_KM,
        "flattening": 0.0,
        "sidereal_period_days": 87.9691,
    },
    "venus": {
        "gm_m3_s2": GM_VENUS,
        "radius_km": R_VENUS_KM,
        "flattening": 0.0,
        "sidereal_period_days": 224.701,
    },
    "earth": {
        "gm_m3_s2": GM_EARTH,
        "radius_km": R_EARTH_KM,
        "flattening": 1.0 / 298.257223563,
        "sidereal_period_days": 365.256363004,
    },
    "mars": {
        "gm_m3_s2": GM_MARS,
        "radius_km": R_MARS_KM,
        "flattening": 0.00589,
        "sidereal_period_days": 686.971,
    },
    "jupiter": {
        "gm_m3_s2": GM_JUPITER,
        "radius_km": R_JUPITER_KM,
        "flattening": 0.06487,
        "sidereal_period_days": 4332.59,
    },
    "saturn": {
        "gm_m3_s2": GM_SATURN,
        "radius_km": R_SATURN_KM,
        "flattening": 0.09796,
        "sidereal_period_days": 10759.22,
    },
    "uranus": {
        "gm_m3_s2": 5.793939e15,
        "radius_km": 25_362.0,
        "flattening": 0.02293,
        "sidereal_period_days": 30688.5,
    },
    "neptune": {
        "gm_m3_s2": 6.836529e15,
        "radius_km": 24_622.0,
        "flattening": 0.01708,
        "sidereal_period_days": 60182.0,
    },
    "sun": {
        "gm_m3_s2": GM_SUN,
        "radius_km": R_SUN_KM,
        "flattening": 9e-6,
        "sidereal_period_days": 0.0,  # not applicable
    },
    "moon": {
        "gm_m3_s2": GM_MOON,
        "radius_km": R_MOON_KM,
        "flattening": 0.0012,
        "sidereal_period_days": 27.321661,
    },
}


def get_planetary_constants(body_name: str) -> Dict[str, float]:
    """Return physical and orbital constants for a solar-system body.

    Parameters
    ----------
    body_name : str
        Case-insensitive body name (e.g. ``"Earth"``, ``"jupiter"``).

    Returns
    -------
    dict
        Keys: ``gm_m3_s2``, ``radius_km``, ``flattening``,
        ``sidereal_period_days``.

    Raises
    ------
    KeyError
        If *body_name* is not in the lookup table.
    """
    key = body_name.strip().lower()
    if key not in _PLANETARY_CONSTANTS:
        raise KeyError(
            f"Unknown body '{body_name}'. "
            f"Available: {sorted(_PLANETARY_CONSTANTS)}"
        )
    return dict(_PLANETARY_CONSTANTS[key])


# ============================================================================
# 5.  Solar System Moons
# ============================================================================

def load_solar_system_moons(**kwargs):
    """Load the solar-system natural-satellite catalog.

    Returns
    -------
    Dataset or IterableDataset
        Moons and their orbital / physical parameters.
    """
    logger.info("Loading solar system moons …")
    return get_dataset("solar_system_moons", **kwargs)


def compute_hill_sphere(
    moon_mass_kg: float,
    planet_mass_kg: float,
    semi_major_axis_m: float,
) -> float:
    """Compute the Hill-sphere radius of a moon orbiting a planet.

    .. math::

        r_{\\text{Hill}} \\approx a \\left(\\frac{m}{3 M}\\right)^{1/3}

    Parameters
    ----------
    moon_mass_kg : float
        Mass of the moon [kg].
    planet_mass_kg : float
        Mass of the host planet [kg].
    semi_major_axis_m : float
        Semi-major axis of the moon's orbit around the planet [m].

    Returns
    -------
    float
        Hill-sphere radius [m].
    """
    if planet_mass_kg <= 0.0:
        raise ValueError("planet_mass_kg must be > 0")
    r_hill = semi_major_axis_m * (moon_mass_kg / (3.0 * planet_mass_kg)) ** (1.0 / 3.0)
    logger.debug(
        "Hill sphere: m=%.3e kg, M=%.3e kg, a=%.3e m → r_H=%.3e m",
        moon_mass_kg,
        planet_mass_kg,
        semi_major_axis_m,
        r_hill,
    )
    return r_hill


# ============================================================================
# 6.  Gaia DR3 Solar System Objects
# ============================================================================

def load_gaia_dr3_sso(**kwargs):
    """Load the Gaia DR3 Solar-System Object epoch-astrometry dataset.

    Returns
    -------
    Dataset or IterableDataset
        Gaia DR3 SSO transit-level astrometry and photometry.
    """
    logger.info("Loading Gaia DR3 SSO dataset …")
    return get_dataset("gaia_dr3_sso", **kwargs)


def extract_astrometric_residuals(
    ds,
    sso_id: Union[str, int],
) -> Dict[str, np.ndarray]:
    """Extract epoch astrometry and residuals for a single SSO.

    Parameters
    ----------
    ds : Dataset
        Gaia DR3 SSO dataset.
    sso_id : str or int
        Source / numbered designation identifying the SSO in the dataset.

    Returns
    -------
    dict
        ``{"epoch_jd": ndarray, "ra_deg": ndarray, "dec_deg": ndarray,
        "residual_ra_mas": ndarray, "residual_dec_mas": ndarray}``
        Any missing columns are returned as empty arrays.
    """
    sso_id_str = str(sso_id)
    logger.info("Extracting astrometric residuals for SSO '%s' …", sso_id_str)

    # Identify the ID column (varies by dataset schema)
    id_candidates = [
        "source_id", "number_mp", "denomination", "sso_id", "id",
    ]
    id_col = None
    for c in id_candidates:
        if c in ds.column_names:
            id_col = c
            break
    if id_col is None:
        logger.warning("No recognisable SSO-ID column found; returning empty result.")
        return {
            "epoch_jd": np.array([]),
            "ra_deg": np.array([]),
            "dec_deg": np.array([]),
            "residual_ra_mas": np.array([]),
            "residual_dec_mas": np.array([]),
        }

    subset = ds.filter(lambda row: str(row[id_col]) == sso_id_str)

    def _safe_col(name: str) -> np.ndarray:
        """Return the column as a numpy array, or empty if missing."""
        if name in subset.column_names:
            return np.array(subset[name], dtype=np.float64)
        return np.array([], dtype=np.float64)

    # Common column-name variants
    epoch = _safe_col("epoch_utc") if "epoch_utc" in subset.column_names else _safe_col("epoch")
    ra = _safe_col("ra")
    dec = _safe_col("dec")
    res_ra = _safe_col("residual_ra") if "residual_ra" in subset.column_names else _safe_col("res_ra")
    res_dec = _safe_col("residual_dec") if "residual_dec" in subset.column_names else _safe_col("res_dec")

    logger.info("Extracted %d astrometric epochs for SSO '%s'.", len(epoch), sso_id_str)

    return {
        "epoch_jd": epoch,
        "ra_deg": ra,
        "dec_deg": dec,
        "residual_ra_mas": res_ra,
        "residual_dec_mas": res_dec,
    }


# ============================================================================
# 7.  Solar 3D Mesh
# ============================================================================

def load_solar_3d_mesh(**kwargs):
    """Load the solar 3D surface-mesh dataset.

    Returns
    -------
    Dataset or IterableDataset
        Vertex / face data for a solar surface mesh.
    """
    logger.info("Loading Solar 3D mesh dataset …")
    return get_dataset("solar_3d", **kwargs)


def generate_solar_surface_grid(
    resolution: int = 100,
) -> Dict[str, np.ndarray]:
    """Generate a UV-sphere mesh grid for solar surface visualisation.

    Parameters
    ----------
    resolution : int
        Number of subdivisions in latitude and longitude.

    Returns
    -------
    dict
        ``{"x": ndarray, "y": ndarray, "z": ndarray, "theta": ndarray,
        "phi": ndarray}`` where each array has shape
        ``(resolution, resolution)`` and x, y, z are in kilometres
        (scaled to the mean solar radius).
    """
    theta = np.linspace(0.0, np.pi, resolution)         # co-latitude
    phi = np.linspace(0.0, 2.0 * np.pi, resolution)     # longitude
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    r = R_SUN_KM  # km
    x = r * np.sin(theta_grid) * np.cos(phi_grid)
    y = r * np.sin(theta_grid) * np.sin(phi_grid)
    z = r * np.cos(theta_grid)

    logger.debug("Solar surface grid: %d × %d vertices", resolution, resolution)

    return {
        "x": x,
        "y": y,
        "z": z,
        "theta": theta_grid,
        "phi": phi_grid,
    }


# ============================================================================
# 8.  Globular Clusters
# ============================================================================

def load_globular_clusters(**kwargs):
    """Load the globular star-cluster catalog.

    Returns
    -------
    Dataset or IterableDataset
        Structural and photometric parameters of globular clusters.
    """
    logger.info("Loading globular cluster catalog …")
    return get_dataset("globular_clusters", **kwargs)


def compute_cluster_potential_field(
    mass_solar: float,
    r_core_pc: float,
    r_pc: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute the Plummer gravitational potential of a globular cluster.

    .. math::

        \\Phi(r) = -\\frac{G M}{\\sqrt{r^2 + a^2}}

    where *a* is the Plummer scale radius (≈ core radius).

    Parameters
    ----------
    mass_solar : float
        Total cluster mass [solar masses].
    r_core_pc : float
        Plummer scale (core) radius [parsec].
    r_pc : float or ndarray
        Radial distance(s) from the cluster centre [parsec].

    Returns
    -------
    float or ndarray
        Gravitational potential Φ(r) [m² s⁻²].
    """
    # Conversions
    M_SUN_KG = 1.98892e30          # solar mass [kg]
    PC_M = 3.0856775814913673e16   # parsec [m]

    mass_kg = mass_solar * M_SUN_KG
    a_m = r_core_pc * PC_M
    r_m = np.asarray(r_pc, dtype=np.float64) * PC_M

    phi = -G * mass_kg / np.sqrt(r_m**2 + a_m**2)

    logger.debug(
        "Plummer potential: M=%.3e M☉, a=%.2f pc → Φ(r_core)=%.4e m²/s²",
        mass_solar,
        r_core_pc,
        float(-G * mass_kg / (a_m * np.sqrt(2.0))),
    )
    return phi


# ============================================================================
# 9.  Hypervelocity Stars
# ============================================================================

def load_hypervelocity_stars(**kwargs):
    """Load the hypervelocity-star catalog (Li et al. 2023).

    Returns
    -------
    Dataset or IterableDataset
        Catalogue of candidate hypervelocity stars with 6-D kinematics.
    """
    logger.info("Loading hypervelocity star catalog …")
    return get_dataset("hypervelocity_stars", **kwargs)


def trace_galactic_origin(
    vx: float,
    vy: float,
    vz: float,
    x0: float,
    y0: float,
    z0: float,
    dt_myr: float,
    n_steps: int = 1000,
) -> Dict[str, np.ndarray]:
    """Backward-integrate a stellar orbit in an approximate Milky Way potential.

    Uses a simple axisymmetric three-component (bulge + disk + NFW halo)
    model with leapfrog integration.

    **Potential components** (all in Galactocentric Cartesian, SI):

    - **Bulge**: Plummer sphere, M_b = 3.4 × 10¹⁰ M☉, a_b = 0.7 kpc
    - **Disk**: Miyamoto–Nagai, M_d = 1.0 × 10¹¹ M☉, a_d = 6.5 kpc, b_d = 0.26 kpc
    - **Halo**: NFW, M_h = 1.0 × 10¹² M☉, r_s = 21.5 kpc, c = 12

    Parameters
    ----------
    vx, vy, vz : float
        Current Galactocentric velocity [km/s].
    x0, y0, z0 : float
        Current Galactocentric position [kpc].
    dt_myr : float
        Total integration time into the past [Myr] (positive = backward).
    n_steps : int
        Number of leapfrog steps.

    Returns
    -------
    dict
        ``{"t_myr": ndarray, "x_kpc": ndarray, "y_kpc": ndarray,
        "z_kpc": ndarray, "vx_km_s": ndarray, "vy_km_s": ndarray,
        "vz_km_s": ndarray}`` — each of shape ``(n_steps + 1,)``.
    """
    # ── physical constants & conversions ──────────────────────────────────
    M_SUN_KG = 1.98892e30
    KPC_M = 3.0856775814913673e19
    KM_M = 1.0e3
    MYR_S = 1.0e6 * YEAR_S

    # ── potential parameters ──────────────────────────────────────────────
    # Bulge (Plummer)
    Mb = 3.4e10 * M_SUN_KG
    ab = 0.7 * KPC_M

    # Disk (Miyamoto–Nagai)
    Md = 1.0e11 * M_SUN_KG
    ad = 6.5 * KPC_M
    bd = 0.26 * KPC_M

    # Halo (NFW, simplified enclosed-mass form)
    Mh = 1.0e12 * M_SUN_KG
    rs = 21.5 * KPC_M
    c_nfw = 12.0
    ln_1pc = np.log(1.0 + c_nfw) - c_nfw / (1.0 + c_nfw)  # normalisation

    def _accel(pos: np.ndarray) -> np.ndarray:
        """Gravitational acceleration [m/s²] at position *pos* [m]."""
        x_, y_, z_ = pos
        R2 = x_**2 + y_**2
        R = np.sqrt(R2)
        r = np.sqrt(R2 + z_**2)

        acc = np.zeros(3)

        # Bulge – Plummer
        denom_b = (r**2 + ab**2) ** 1.5
        acc -= G * Mb * pos / denom_b

        # Disk – Miyamoto–Nagai
        zterm = np.sqrt(z_**2 + bd**2)
        A = ad + zterm
        denom_d = (R2 + A**2) ** 1.5
        acc[0] -= G * Md * x_ / denom_d
        acc[1] -= G * Md * y_ / denom_d
        acc[2] -= G * Md * z_ * A / (zterm * denom_d)

        # Halo – NFW
        s = r / rs
        if s > 1e-10:
            M_enc = Mh * (np.log(1.0 + s) - s / (1.0 + s)) / ln_1pc
            acc -= G * M_enc * pos / r**3

        return acc

    # ── convert inputs to SI ──────────────────────────────────────────────
    pos = np.array([x0, y0, z0]) * KPC_M          # [m]
    vel = np.array([vx, vy, vz]) * KM_M           # [m/s]

    # Backward integration → negate velocities
    vel = -vel

    dt_total = dt_myr * MYR_S
    h = dt_total / n_steps

    # Storage
    positions = np.zeros((n_steps + 1, 3))
    velocities = np.zeros((n_steps + 1, 3))
    positions[0] = pos
    velocities[0] = -vel  # store original (forward) velocity

    # ── leapfrog integration ──────────────────────────────────────────────
    a = _accel(pos)
    vel_half = vel + 0.5 * h * a  # kick half-step

    for step in range(n_steps):
        pos = pos + h * vel_half                       # drift
        a = _accel(pos)
        vel_half_new = vel_half + h * a                # kick
        # Store full-step velocity (average of half-steps) — reverse sign
        # to report in forward-time sense
        vel_full = -(vel_half + vel_half_new) * 0.5
        positions[step + 1] = pos
        velocities[step + 1] = vel_full
        vel_half = vel_half_new

    times = np.linspace(0.0, dt_myr, n_steps + 1)

    logger.info(
        "Galactic backward integration: %.1f Myr, %d steps, "
        "start=(%.2f, %.2f, %.2f) kpc",
        dt_myr,
        n_steps,
        x0,
        y0,
        z0,
    )

    return {
        "t_myr": times,
        "x_kpc": positions[:, 0] / KPC_M,
        "y_kpc": positions[:, 1] / KPC_M,
        "z_kpc": positions[:, 2] / KPC_M,
        "vx_km_s": velocities[:, 0] / KM_M,
        "vy_km_s": velocities[:, 1] / KM_M,
        "vz_km_s": velocities[:, 2] / KM_M,
    }
