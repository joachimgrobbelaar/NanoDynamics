"""Astrodynamics and Space Environment Engine.

Incorporates algorithms from:
- poliastro: Analytical Keplerian <-> Cartesian conversions, orbital invariants, VNC frames
- orbital_link_budget: RF path loss, Doppler shift, carrier-to-noise ratio
- NequickG: Ionospheric electron density and Total Electron Content (TEC) models
- leo-edge-failure-models: South Atlantic Anomaly (SAA) radiation belt exposure and SEU rates
"""

from __future__ import annotations
import math
import numpy as np

# Standard Earth physical parameters (WGS-84)
MU_EARTH = 3.986004418e14  # m^3 / s^2
R_EARTH_M = 6378137.0      # m
SPEED_OF_LIGHT = 299792458.0  # m / s


def rv_to_keplerian(
    r_vec: np.ndarray | list[float],
    v_vec: np.ndarray | list[float],
    mu: float = MU_EARTH,
) -> dict[str, float]:
    """Convert Cartesian state vector [x, y, z], [vx, vy, vz] to osculating Keplerian elements.
    
    Returns:
        dict containing:
            a: Semi-major axis [m]
            e: Eccentricity [-]
            inc: Inclination [rad] (and inc_deg [deg])
            raan: Longitude of ascending node [rad] (and raan_deg [deg])
            arg_p: Argument of periapsis [rad] (and arg_p_deg [deg])
            nu: True anomaly [rad] (and nu_deg [deg])
            period_s: Orbital period [s]
            specific_energy: Specific mechanical energy [J/kg]
            h_mag: Specific angular momentum magnitude [m^2/s]
            p: Semi-latus rectum [m]
    """
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))

    if r_norm < 1e-3:
        raise ValueError("Position vector norm is too small.")

    # 1. Specific angular momentum
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))

    # 2. Vector pointing to ascending node: n = k x h
    k_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = np.cross(k_unit, h)
    n_norm = float(np.linalg.norm(n))

    # 3. Eccentricity vector: e_vec = (v x h)/mu - r/|r|
    e_vec = (np.cross(v, h) / mu) - (r / r_norm)
    e = float(np.linalg.norm(e_vec))

    # 4. Specific mechanical energy
    spec_energy = 0.5 * (v_norm ** 2) - (mu / r_norm)

    # 5. Semi-major axis
    if abs(spec_energy) > 1e-9:
        a = -mu / (2.0 * spec_energy)
    else:
        a = float("inf")

    # 6. Inclination
    inc = math.acos(np.clip(h[2] / h_norm, -1.0, 1.0)) if h_norm > 1e-12 else 0.0

    # 7. Longitude of Ascending Node (RAAN)
    if n_norm > 1e-9:
        raan = math.acos(np.clip(n[0] / n_norm, -1.0, 1.0))
        if n[1] < 0.0:
            raan = 2.0 * math.pi - raan
    else:
        raan = 0.0

    # 8. Argument of Periapsis
    if n_norm > 1e-9 and e > 1e-7:
        arg_p = math.acos(np.clip(np.dot(n, e_vec) / (n_norm * e), -1.0, 1.0))
        if e_vec[2] < 0.0:
            arg_p = 2.0 * math.pi - arg_p
    else:
        arg_p = 0.0

    # 9. True Anomaly
    if e > 1e-7:
        nu = math.acos(np.clip(np.dot(e_vec, r) / (e * r_norm), -1.0, 1.0))
        if np.dot(r, v) < 0.0:
            nu = 2.0 * math.pi - nu
    else:
        nu = 0.0

    # Orbital period (for bound elliptic orbits)
    period_s = (2.0 * math.pi * math.sqrt((a ** 3) / mu)) if a > 0.0 else float("inf")
    p = a * (1.0 - (e ** 2)) if a > 0.0 else h_norm**2 / mu

    return {
        "a": float(a),
        "e": float(e),
        "inc": float(inc),
        "inc_deg": float(math.degrees(inc)),
        "raan": float(raan),
        "raan_deg": float(math.degrees(raan)),
        "arg_p": float(arg_p),
        "arg_p_deg": float(math.degrees(arg_p)),
        "nu": float(nu),
        "nu_deg": float(math.degrees(nu)),
        "period_s": float(period_s),
        "specific_energy": float(spec_energy),
        "h_mag": float(h_norm),
        "p": float(p),
    }


def compute_vnc_basis(r_vec: np.ndarray, v_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate true velocity-aligned VNC (Velocity-Normal-CoNormal) unit basis vectors.
    
    Returns:
        u_v: Prograde unit vector (tangent to velocity)
        u_n: Orbit normal unit vector (aligned with angular momentum)
        u_c: Co-normal / In-plane radial unit vector (u_v x u_n)
    """
    v_norm = np.linalg.norm(v_vec)
    u_v = v_vec / v_norm if v_norm > 1e-12 else np.array([1.0, 0.0, 0.0])

    h = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h)
    u_n = h / h_norm if h_norm > 1e-12 else np.array([0.0, 0.0, 1.0])

    u_c = np.cross(u_v, u_n)
    return u_v, u_n, u_c


def compute_rf_link_budget(
    sat_pos: np.ndarray | list[float],
    sat_vel: np.ndarray | list[float],
    gnd_pos: np.ndarray | list[float] | None = None,
    carrier_freq_hz: float = 2.4e9,  # 2.4 GHz S-band
    tx_power_dbm: float = 30.0,      # 1 Watt
    tx_gain_dbi: float = 3.0,
    rx_gain_dbi: float = 12.0,
) -> dict[str, float]:
    """Compute RF path loss, carrier Doppler shift, and link margin according to Friis transmission.
    
    References:
        gercunderscore4/orbital_link_budget
    """
    r_sat = np.asarray(sat_pos, dtype=np.float64)
    v_sat = np.asarray(sat_vel, dtype=np.float64)

    # Default ground station at sub-satellite footprint if not specified
    if gnd_pos is None:
        r_norm = np.linalg.norm(r_sat)
        gnd_pos = (r_sat / r_norm) * R_EARTH_M

    r_gnd = np.asarray(gnd_pos, dtype=np.float64)
    d_vec = r_sat - r_gnd
    slant_dist_m = float(np.linalg.norm(d_vec))
    slant_dist_km = max(0.1, slant_dist_m / 1000.0)

    # Line-of-sight unit vector from ground to satellite
    los_unit = d_vec / slant_dist_m if slant_dist_m > 1e-6 else np.array([0.0, 0.0, 1.0])

    # Range rate (relative velocity projection)
    range_rate_m_s = float(np.dot(v_sat, los_unit))

    # Classical Doppler shift: delta_f = -f0 * (range_rate / c)
    doppler_shift_hz = -carrier_freq_hz * (range_rate_m_s / SPEED_OF_LIGHT)

    # Free Space Path Loss (FSPL): FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
    fspl_db = 20.0 * math.log10(slant_dist_m) + 20.0 * math.log10(carrier_freq_hz) + 20.0 * math.log10((4.0 * math.pi) / SPEED_OF_LIGHT)

    # Received Power: P_rx = P_tx + G_tx + G_rx - FSPL
    p_rx_dbm = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - fspl_db
    rx_sensitivity_dbm = -110.0  # Typical LEO ground station receiver
    link_margin_db = p_rx_dbm - rx_sensitivity_dbm

    return {
        "slant_range_km": float(slant_dist_km),
        "range_rate_m_s": float(range_rate_m_s),
        "doppler_shift_hz": float(doppler_shift_hz),
        "fspl_db": float(fspl_db),
        "received_power_dbm": float(p_rx_dbm),
        "link_margin_db": float(link_margin_db),
    }


def compute_space_environment(
    lat_deg: float,
    lon_deg: float,
    alt_km: float,
    f107: float = 135.0,
) -> dict[str, float]:
    """Evaluate ionospheric Total Electron Content (TEC) and South Atlantic Anomaly (SAA) radiation dose.
    
    References:
        - tpl2go/NequickG: Ionospheric TEC modeling
        - pfandzelter/leo-edge-failure-models: SAA boundary and edge SEU susceptibility
    """
    # 1. Total Electron Content (TEC in TECU, 1 TECU = 10^16 electrons/m^2)
    # Peak ionosphere F2 layer around 300-450 km, diurnal solar elevation peak
    f2_peak_alt = 350.0
    alt_diff = abs(alt_km - f2_peak_alt)
    base_tec = 25.0 * math.exp(-((alt_diff / 150.0) ** 2))
    # Equatorial ionospheric anomaly (EIA) peaks near +/- 15 deg geomagnetic lat
    eia_factor = 1.0 + 0.6 * math.exp(-((abs(lat_deg) - 15.0) / 10.0) ** 2)
    solar_factor = (max(60.0, float(f107)) / 70.0) ** 0.8
    tec_tecu = max(2.0, base_tec * eia_factor * solar_factor)

    # Ionospheric delay at L-band (1.575 GHz GPS / 2.4 GHz S-band)
    f_ref = 2.4e9
    iono_delay_s = (40.3 * (tec_tecu * 1e16)) / (SPEED_OF_LIGHT * (f_ref ** 2))

    # 2. South Atlantic Anomaly (SAA) Radiation Indicator
    # SAA region roughly: Lat -50 to 0 deg, Lon -90 to 0 deg
    in_saa = False
    saa_flux_factor = 0.0
    if -50.0 <= lat_deg <= 0.0 and -90.0 <= lon_deg <= 0.0 and alt_km >= 200.0:
        in_saa = True
        center_lat, center_lon = -25.0, -45.0
        dist_deg = math.sqrt(((lat_deg - center_lat) ** 2) + ((lon_deg - center_lon) ** 2))
        saa_flux_factor = max(0.0, 1.0 - (dist_deg / 35.0))

    # Expected Single-Event Upset (SEU) rate on 40nm MCU (upsets / hour)
    base_seu_rate = 0.01  # Quiet LEO background
    seu_rate_per_hr = base_seu_rate * (1.0 + saa_flux_factor * 85.0)

    return {
        "tec_tecu": float(tec_tecu),
        "iono_delay_ns": float(iono_delay_s * 1e9),
        "in_saa": 1.0 if in_saa else 0.0,
        "saa_flux_intensity": float(saa_flux_factor),
        "seu_rate_per_hr": float(seu_rate_per_hr),
    }
