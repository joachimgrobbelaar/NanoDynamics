"""Verification of IAU ITRF<->GCRF kinematics and J2 secular nodal regression vs. analytical theory."""

import math
import numpy as np

from src.astrodynamics.force_models import EARTH_J2, MU_EARTH, R_EARTH
from src.astrodynamics.frames import ecef_to_gcrf, gcrf_to_ecef, rv_to_osculating_coe
from src.astrodynamics.propagators import NumericalPropagator


def test_ecef_gcrf_roundtrip_with_coriolis():
    r_ecef = np.array([6878137.0, 120000.0, -45000.0], dtype=np.float64)
    v_ecef = np.array([-110.0, 5200.0, 5400.0], dtype=np.float64)
    t_s = 1845.0

    r_gcrf, v_gcrf = ecef_to_gcrf(r_ecef, v_ecef, t_s)
    r_back, v_back = gcrf_to_ecef(r_gcrf, v_gcrf, t_s)

    assert np.allclose(r_ecef, r_back, atol=1e-7)
    assert np.allclose(v_ecef, v_back, atol=1e-7)


def test_j2_secular_nodal_regression_matches_analytical_theory():
    """Verify numerical J2 secular RAAN regression matches analytical formula:
        dOmega/dt = - (3/2) * J2 * (R_E / p)^2 * n * cos(i)
    """
    a = R_EARTH + 600_000.0
    inc_deg = 53.0
    inc = math.radians(inc_deg)
    raan0 = math.radians(40.0)
    v_circ = math.sqrt(MU_EARTH / a)

    # Initial position at ascending node with RAAN = 40 deg
    r0 = np.array([a * math.cos(raan0), a * math.sin(raan0), 0.0], dtype=np.float64)
    v0 = np.array(
        [
            -v_circ * math.cos(inc) * math.sin(raan0),
            v_circ * math.cos(inc) * math.cos(raan0),
            v_circ * math.sin(inc),
        ],
        dtype=np.float64,
    )
    state0 = np.concatenate([r0, v0])

    # Propagate 3 full orbital periods with J2-J4 only (drag disabled)
    n_mean = math.sqrt(MU_EARTH / (a**3))
    period_s = 2.0 * math.pi / n_mean
    duration_s = 3.0 * period_s

    prop = NumericalPropagator(include_j2_j4=True, include_drag=False, include_srp=False, include_third_body=False)
    traj = prop.propagate_gcrf(state0, duration_s=duration_s, dt_s=60.0)

    coe_0 = rv_to_osculating_coe(traj.r_m[0], traj.v_m_s[0])
    coe_f = rv_to_osculating_coe(traj.r_m[-1], traj.v_m_s[-1])

    delta_raan_num = coe_f["raan_rad"] - coe_0["raan_rad"]
    raan_dot_analytical = -1.5 * EARTH_J2 * ((R_EARTH / a) ** 2) * n_mean * math.cos(inc)
    delta_raan_analytical = raan_dot_analytical * duration_s

    rel_err = abs(delta_raan_num - delta_raan_analytical) / abs(delta_raan_analytical)
    assert rel_err < 0.02
