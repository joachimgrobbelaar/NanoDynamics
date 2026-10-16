"""Programmatic Physics Verification for Track 1: Cislunar Multi-Body & Lunar Harmonics.

Verifies:
1. Jacobi energy conservation along unpowered cislunar trajectories (|Delta C / C_0| < 1e-6).
2. Ballistic capture entry envelope through L1/L2 gateways without retro-burns.
3. Coupled Earth-Moon-Sun (EMS) ephemeris accelerations and LP165P 4x4 lunar mascon perturbations.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from leo_simulator.constants import MU_EARTH, MU_MOON, R_MOON, R_MOON_ORBIT
from leo_simulator.models.dynamics import OrbitalDynamics
from leo_simulator.models.gravity import (
    coupled_ems_acceleration,
    cr3bp_synodic_derivatives,
    detect_l1_l2_gateway,
    jacobi_constant_synodic,
    lunar_harmonics_acceleration,
    moon_position,
    sun_position,
)


def test_jacobi_energy_conservation_cislunar():
    """Assert Jacobi energy conservation along unpowered cislunar trajectories (|Delta C / C_0| < 1e-6)."""
    # Initialize an unpowered cislunar trajectory in the Earth-Moon synodic barycentric frame
    # Starting near L1 gateway transit corridor (r ~ 326,000 km)
    r0_syn = np.array([3.22e8, 1.50e7, 5.0e6], dtype=np.float64)
    v0_syn = np.array([45.0, -110.0, 12.0], dtype=np.float64)
    y0 = np.concatenate([r0_syn, v0_syn])

    c0 = jacobi_constant_synodic(r0_syn, v0_syn)
    assert np.isfinite(c0) and c0 > 0.0

    # Integrate over 5 days in the synodic frame with high-order DOP853
    t_span = (0.0, 5.0 * 86400.0)
    t_eval = np.linspace(t_span[0], t_span[1], 200)
    sol = solve_ivp(
        cr3bp_synodic_derivatives,
        t_span,
        y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-11,
        atol=1e-11,
    )
    assert sol.success

    c_values = np.array([
        jacobi_constant_synodic(sol.y[0:3, k], sol.y[3:6, k])
        for k in range(sol.y.shape[1])
    ])
    rel_drift = np.max(np.abs((c_values - c0) / c0))
    assert rel_drift < 1e-6, f"Jacobi relative drift {rel_drift:.3e} exceeded 1e-6"


def test_l1_l2_wsb_ballistic_capture_envelope():
    """Verify ballistic capture entry envelope through L1/L2 gateways without retro-burns."""
    t = 0.0
    r_m = moon_position(t)  # [384_400_000, 0, 0]
    omega_m = np.sqrt((MU_EARTH + MU_MOON) / (R_MOON_ORBIT**3))
    v_moon = np.array([0.0, omega_m * R_MOON_ORBIT, 0.0], dtype=np.float64)

    # 1. L1 Gateway WSB capture entry state (r ~ 345,000 km, inside L1-Moon neck with negative selenocentric energy)
    r_l1_entry = np.array([3.45e8, 0.0, 0.0], dtype=np.float64)
    # Small relative velocity wrt Moon (~280 m/s, below escape speed sqrt(2*MU_MOON/39400km) ~ 498 m/s)
    v_l1_entry = v_moon + np.array([60.0, -250.0, 0.0], dtype=np.float64)

    res_l1 = detect_l1_l2_gateway(r_l1_entry, v_l1_entry, t=t, r_moon=r_m)
    assert res_l1["in_gateway_regime"] is True
    assert res_l1["gateway"] == "L1"
    assert res_l1["wsb_ballistic_capture"] is True
    assert res_l1["selenocentric_energy"] < 0.0

    # 2. L2 Gateway WSB capture entry state (r ~ 420,000 km, exterior L2 corridor)
    r_l2_entry = np.array([4.20e8, 0.0, 0.0], dtype=np.float64)
    v_l2_entry = v_moon + np.array([-50.0, 220.0, 0.0], dtype=np.float64)

    res_l2 = detect_l1_l2_gateway(r_l2_entry, v_l2_entry, t=t, r_moon=r_m)
    assert res_l2["in_gateway_regime"] is True
    assert res_l2["gateway"] == "L2"
    assert res_l2["wsb_ballistic_capture"] is True

    # 3. Hyperbolic flyby state (high relative velocity -> not captured)
    v_hyperbolic = v_moon + np.array([1200.0, 900.0, 0.0], dtype=np.float64)
    res_flyby = detect_l1_l2_gateway(r_l1_entry, v_hyperbolic, t=t, r_moon=r_m)
    assert res_flyby["wsb_ballistic_capture"] is False
    assert res_flyby["selenocentric_energy"] > 0.0


def test_coupled_ems_and_lp165p_lunar_harmonics():
    """Verify coupled Earth-Moon-Sun ephemerides and LP165P 4x4 mascon perturbations."""
    r_sat = np.array([3.50e8, 1.0e7, 2.0e6], dtype=np.float64)
    a_ems = coupled_ems_acceleration(r_sat, t=12345.0, include_sun=True)
    a_em_only = coupled_ems_acceleration(r_sat, t=12345.0, include_sun=False)
    assert a_ems.shape == (3,)
    assert np.all(np.isfinite(a_ems))
    # Solar third-body perturbation at cislunar distance is non-zero (~1e-6 to 1e-5 m/s^2)
    diff_sun = np.linalg.norm(a_ems - a_em_only)
    assert 1e-8 < diff_sun < 1e-3

    # Low-Lunar Orbit (100 km altitude above R_MOON) experiencing LP165P 4x4 mascon perturbations
    r_llo = np.array([R_MOON + 100_000.0, 250_000.0, 180_000.0], dtype=np.float64)
    a_lp165p = lunar_harmonics_acceleration(r_llo)
    assert a_lp165p.shape == (3,)
    assert np.all(np.isfinite(a_lp165p))
    # At 100 km LLO, lunar J2 + C22 + degree 3/4 harmonics produce ~1e-4 to 5e-3 m/s^2
    mag = float(np.linalg.norm(a_lp165p))
    assert 1e-5 < mag < 1e-2

    # Verify OrbitalDynamics integration
    dyn = OrbitalDynamics(
        include_j2=True,
        include_drag=False,
        include_moon=True,
        include_sun=True,
        include_lunar_harmonics=True,
    )
    acc = dyn.compute_accelerations(r_sat, np.array([0.0, 1000.0, 0.0]), t=100.0)
    assert "sun" in acc and "lunar_harmonics" in acc
    assert np.linalg.norm(acc["sun"]) > 0.0
    assert np.linalg.norm(acc["lunar_harmonics"]) > 0.0
