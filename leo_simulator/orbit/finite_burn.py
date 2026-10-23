"""Finite-duration orbital maneuver integration with continuous thrust and propellant mass depletion.

Implements realistic low-thrust and chemical continuous burns using Tsiolkovsky rocket dynamics:
  m_dot = - Thrust / (Isp * g0)
  a_thrust(t) = Thrust / m(t) * u_thrust(t)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

from leo_simulator.constants import MU_EARTH, R_EARTH
from leo_simulator.orbit.astrodynamics import compute_vnc_basis

G0 = 9.80665  # Standard sea-level gravitational acceleration [m/s^2]


def compute_finite_burn_duration(
    mass_kg: float,
    delta_v_mag: float,
    thrust_n: float = 20.0,
    isp_s: float = 300.0,
) -> Tuple[float, float]:
    """Calculate finite burn duration and propellant consumed using the rocket equation.

    Returns:
        (burn_duration_s, propellant_used_kg)
    """
    if delta_v_mag <= 1e-6 or thrust_n <= 1e-6 or isp_s <= 1e-6:
        return 0.0, 0.0
    
    c = isp_s * G0
    mass_ratio = math.exp(delta_v_mag / c)
    m_final = mass_kg / mass_ratio
    m_prop = mass_kg - m_final
    m_dot = thrust_n / c
    duration = m_prop / m_dot
    return max(0.1, duration), m_prop


def propagate_finite_burn(
    r_start: np.ndarray,
    v_start: np.ndarray,
    mass_start: float,
    dv_vnc: Tuple[float, float, float],
    t_start: float = 0.0,
    thrust_n: float = 20.0,
    isp_s: float = 300.0,
    mu: float = MU_EARTH,
    max_step: float = 2.0,
) -> Dict[str, Union[np.ndarray, float]]:
    """Propagate continuous finite burn in VNC maneuver orientation."""
    dv_prog, dv_norm, dv_rad = dv_vnc
    dv_mag = math.sqrt(dv_prog**2 + dv_norm**2 + dv_rad**2)

    if dv_mag < 1e-5:
        return {
            "t": np.array([t_start]),
            "r": np.array([r_start]),
            "v": np.array([v_start]),
            "mass": np.array([mass_start]),
            "burn_duration": 0.0,
            "propellant_used": 0.0,
        }

    burn_dur, m_prop = compute_finite_burn_duration(mass_start, dv_mag, thrust_n, isp_s)
    c = isp_s * G0
    m_dot = thrust_n / c

    # Unit vector components in VNC
    u_vnc = np.array([dv_prog, dv_norm, dv_rad], dtype=np.float64) / dv_mag

    def dynamics(t, y):
        r = y[0:3]
        v = y[3:6]
        m = max(1e-3, y[6])

        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / (r_norm**3)

        # Compute dynamic VNC basis
        u_v, u_n, u_c = compute_vnc_basis(r, v)
        u_inertial = u_vnc[0] * u_v + u_vnc[1] * u_n + u_vnc[2] * u_c
        a_thrust = (thrust_n / m) * u_inertial

        drdt = v
        dvdt = a_grav + a_thrust
        dmdt = -m_dot
        return np.concatenate([drdt, dvdt, [dmdt]])

    y0 = np.concatenate([r_start, v_start, [mass_start]])
    t_span = (t_start, t_start + burn_dur)
    
    # Generate fine evaluation grid across burn duration
    n_pts = max(10, int(burn_dur / max_step))
    t_eval = np.linspace(t_start, t_start + burn_dur, n_pts)

    sol = solve_ivp(
        dynamics,
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )

    return {
        "t": sol.t,
        "r": sol.y[0:3].T,
        "v": sol.y[3:6].T,
        "mass": sol.y[6],
        "burn_duration": burn_dur,
        "propellant_used": m_prop,
    }
