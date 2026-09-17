"""Programmatic verification test script for LEO Nanosatellite Simulation.

Verifies:
  1. Altitude decays over time due to atmospheric drag (R2, Acceptance Criteria).
  2. Ascending node (RAAN) regresses over time due to Earth J2 oblateness (R2, Acceptance Criteria).
  3. Drag vs No-Drag comparative altitude loss.
  4. Nodal regression rate agreement with analytical perturbation theory.

Can be run via:
    pytest test_simulation.py
or directly:
    python3 test_simulation.py
"""

import os
import sys

import numpy as np

# Ensure project is importable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from leo_simulator import (
    R_EARTH,
    ExponentialAtmosphere,
    OrbitalElements,
    OrbitPropagator,
    Satellite,
    analytical_j2_raan_rate,
)


def create_baseline_orbit(altitude_km: float = 350.0, inclination_deg: float = 51.6) -> OrbitalElements:
    """Helper to construct a standardized near-circular LEO test orbit."""
    return OrbitalElements(
        a=R_EARTH + altitude_km * 1000.0,
        e=0.001,
        i=np.radians(inclination_deg),
        raan=np.radians(45.0),
        arg_pe=np.radians(0.0),
        nu=np.radians(0.0),
    )


def test_altitude_decays_over_time_drag():
    """Acceptance Criterion: Programmatically assert that altitude decays over time (drag)."""
    sat = Satellite.cubesat_3u(mass=4.0, cd=2.2)
    atm = ExponentialAtmosphere(h0=350_000.0, rho0=1.0e-11, scale_height=53_000.0)

    prop_drag = OrbitPropagator(
        satellite=sat,
        atmosphere=atm,
        include_central_gravity=True,
        include_j2=False,
        include_drag=True,
        solver_method="DOP853",
    )

    initial_orbit = create_baseline_orbit(altitude_km=350.0, inclination_deg=51.6)
    duration = 3.0 * initial_orbit.period  # ~4.6 hours

    result = prop_drag.propagate(initial_orbit, duration_seconds=duration, dt_eval=30.0)

    assert result.success, f"Propagation failed: {result.message}"

    # 1. Final altitude must be strictly lower than initial altitude
    assert result.final_altitude < result.initial_altitude, (
        f"Expected altitude decay, but final altitude ({result.final_altitude:.2f} m) "
        f">= initial altitude ({result.initial_altitude:.2f} m)"
    )

    # 2. Net altitude decay must be positive and significant
    assert result.altitude_decay > 50.0, (
        f"Expected net altitude decay > 50 m over 3 orbits, got {result.altitude_decay:.2f} m"
    )

    # 3. Semi-major axis must decrease monotonically over full orbits
    da = result.semi_major_axes[0] - result.semi_major_axes[-1]
    assert da > 0.0, f"Expected semi-major axis to decay, but da={da:.2f} m"


def test_drag_vs_no_drag_comparison():
    """Verify that drag causes strictly greater altitude loss compared to frictionless propagation."""
    sat = Satellite.cubesat_3u(mass=4.0, cd=2.2)
    initial_orbit = create_baseline_orbit(altitude_km=350.0)
    duration = 2.0 * initial_orbit.period

    # Propagator without drag
    prop_nodrag = OrbitPropagator(
        satellite=sat,
        include_central_gravity=True,
        include_j2=False,
        include_drag=False,
    )
    res_nodrag = prop_nodrag.propagate(initial_orbit, duration_seconds=duration)

    # Propagator with drag
    prop_drag = OrbitPropagator(
        satellite=sat,
        include_central_gravity=True,
        include_j2=False,
        include_drag=True,
    )
    res_drag = prop_drag.propagate(initial_orbit, duration_seconds=duration)

    assert res_drag.final_altitude < res_nodrag.final_altitude, (
        f"Drag final altitude ({res_drag.final_altitude:.2f} m) should be strictly less "
        f"than no-drag final altitude ({res_nodrag.final_altitude:.2f} m)"
    )


def test_ascending_node_regresses_j2():
    """Acceptance Criterion: Programmatically assert that the ascending node regresses (J2)."""
    sat = Satellite.cubesat_3u()

    # Prograde inclined orbit (i = 51.6 deg < 90 deg)
    initial_orbit = create_baseline_orbit(altitude_km=350.0, inclination_deg=51.6)
    duration = 4.0 * initial_orbit.period  # ~6.1 hours

    prop_j2 = OrbitPropagator(
        satellite=sat,
        include_central_gravity=True,
        include_j2=True,
        include_drag=False,
        solver_method="DOP853",
    )

    result = prop_j2.propagate(initial_orbit, duration_seconds=duration, dt_eval=30.0)
    assert result.success, f"Propagation failed: {result.message}"

    delta_raan_rad = result.raan_change
    delta_raan_deg = np.degrees(delta_raan_rad)

    # 1. Ascending node must regress (negative change for prograde orbit)
    assert delta_raan_rad < 0.0, (
        f"Expected RAAN to regress (negative change), but got delta_raan={delta_raan_deg:.4f} deg"
    )

    # 2. Check agreement with first-order analytical secular rate:
    # dot_Omega = - 1.5 * J2 * (R_E / p)^2 * n * cos(i)
    analytical_rate = analytical_j2_raan_rate(initial_orbit.a, initial_orbit.e, initial_orbit.i)
    expected_delta_raan_rad = analytical_rate * duration
    expected_delta_raan_deg = np.degrees(expected_delta_raan_rad)

    # Allow small difference due to short-period osculating fluctuations vs secular mean rate (< 1.5%)
    rel_discrepancy = abs(delta_raan_rad - expected_delta_raan_rad) / abs(expected_delta_raan_rad)
    assert rel_discrepancy < 0.015, (
        f"Numerical RAAN change ({delta_raan_deg:.4f} deg) deviates from analytical rate "
        f"({expected_delta_raan_deg:.4f} deg) by {rel_discrepancy * 100.0:.2f}% (tolerance: 1.5%)"
    )


def test_j2_polar_orbit_zero_regression():
    """Verify that a polar orbit (i = 90 deg) exhibits zero secular nodal regression."""
    polar_orbit = create_baseline_orbit(altitude_km=400.0, inclination_deg=90.0)
    prop = OrbitPropagator(include_central_gravity=True, include_j2=True, include_drag=False)

    duration = 2.0 * polar_orbit.period
    res = prop.propagate(polar_orbit, duration_seconds=duration, dt_eval=30.0)

    delta_raan_deg = np.degrees(res.raan_change)
    # Secular change is 0; small osculating variations bounded within 0.005 deg
    assert abs(delta_raan_deg) < 0.005, f"Polar orbit should not exhibit nodal drift, got {delta_raan_deg:.5f} deg"


def test_full_combined_simulation():
    """Verify full simulation run with all active dynamics simultaneously."""
    sat = Satellite.cubesat_3u(mass=4.0, cd=2.2)
    atm = ExponentialAtmosphere(h0=350_000.0, rho0=8.0e-12, scale_height=53_000.0)

    prop = OrbitPropagator(
        satellite=sat,
        atmosphere=atm,
        include_central_gravity=True,
        include_j2=True,
        include_drag=True,
    )

    orbit = create_baseline_orbit(altitude_km=350.0, inclination_deg=51.6)
    duration = 3.0 * orbit.period

    res = prop.propagate(orbit, duration_seconds=duration, dt_eval=30.0)

    assert res.success
    # Both altitude decay and RAAN regression must be present
    assert res.altitude_decay > 30.0
    assert res.raan_change < 0.0


if __name__ == "__main__":
    print("Executing programmatic verification tests...")
    test_altitude_decays_over_time_drag()
    print("  [PASS] test_altitude_decays_over_time_drag")

    test_drag_vs_no_drag_comparison()
    print("  [PASS] test_drag_vs_no_drag_comparison")

    test_ascending_node_regresses_j2()
    print("  [PASS] test_ascending_node_regresses_j2")

    test_j2_polar_orbit_zero_regression()
    print("  [PASS] test_j2_polar_orbit_zero_regression")

    test_full_combined_simulation()
    print("  [PASS] test_full_combined_simulation")

    print("\nAll programmatic verification tests passed successfully!")
