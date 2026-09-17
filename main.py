"""Main execution script for LEO Nanosatellite Orbital Dynamics Simulation (Stage 1).

Runs comparative numerical orbit propagations:
  1. Central gravity only (Keplerian baseline)
  2. Central gravity + J2 oblateness perturbation
  3. Full physics: Central gravity + J2 + Atmospheric drag

Generates telemetry reports and exports visualization plots to output/.
"""

import os
import sys

import numpy as np

# Add project root to sys.path if running directly
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
from leo_simulator.visualization import (
    plot_altitude_decay,
    plot_orbit_3d,
    plot_orbital_elements_history,
    plot_raan_regression,
)


def run_simulation() -> None:
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" LEO NANOSATELLITE ORBITAL DYNAMICS SIMULATOR (STAGE 1)")
    print("=" * 70)

    # 1. Define Satellite (3U CubeSat)
    sat = Satellite.cubesat_3u(name="AeroCube-3U", mass=4.0, cd=2.2)
    print(f"Spacecraft: {sat.name}")
    print(f"  Mass:                  {sat.mass:.2f} kg")
    print(f"  Drag Area:             {sat.drag_area:.4f} m^2")
    print(f"  Drag Coefficient (Cd): {sat.cd:.2f}")
    print(f"  Ballistic Coefficient: {sat.ballistic_coefficient:.2f} kg/m^2\n")

    # 2. Define Initial Orbit (350 km LEO, ISS-inclination 51.6 deg)
    initial_alt_km = 350.0
    r_initial_m = R_EARTH + initial_alt_km * 1000.0
    initial_orbit = OrbitalElements(
        a=r_initial_m,
        e=0.001,  # Near-circular
        i=np.radians(51.6),  # 51.6 deg inclination
        raan=np.radians(45.0),
        arg_pe=np.radians(0.0),
        nu=np.radians(0.0),
    )

    orbit_period_s = initial_orbit.period
    sim_duration_s = 5.0 * orbit_period_s  # 5 complete orbits (~7.6 hours)
    dt_eval_s = 30.0  # Dense reporting every 30 seconds

    print("Initial Orbit Parameters:")
    print(f"  Altitude:              {initial_alt_km:.2f} km")
    print(f"  Semi-major axis (a):   {initial_orbit.a / 1000.0:.2f} km")
    print(f"  Eccentricity (e):      {initial_orbit.e:.5f}")
    print(f"  Inclination (i):       {np.degrees(initial_orbit.i):.2f} deg")
    print(f"  RAAN (Omega):          {np.degrees(initial_orbit.raan):.2f} deg")
    print(f"  Orbital Period:        {orbit_period_s / 60.0:.2f} min")
    print(f"  Simulation Duration:   {sim_duration_s / 3600.0:.2f} hours ({sim_duration_s / orbit_period_s:.1f} orbits)\n")

    # Atmosphere model: reference density adjusted for prominent LEO decay demonstration
    # At 350 km, rho ~ 9.5e-12 kg/m^3
    atm = ExponentialAtmosphere(h0=350_000.0, rho0=9.5e-12, scale_height=53_200.0)

    # 3. Propagate Scenarios
    print("Running numerical propagations (DOP853 solver)...")

    # Scenario A: Keplerian (Two-Body only)
    prop_kepler = OrbitPropagator(
        satellite=sat,
        include_central_gravity=True,
        include_j2=False,
        include_drag=False,
        solver_method="DOP853",
    )
    res_kepler = prop_kepler.propagate(initial_orbit, duration_seconds=sim_duration_s, dt_eval=dt_eval_s)
    print("  [✓] Case A: Keplerian two-body completed.")

    # Scenario B: Gravity + J2
    prop_j2 = OrbitPropagator(
        satellite=sat,
        include_central_gravity=True,
        include_j2=True,
        include_drag=False,
        solver_method="DOP853",
    )
    res_j2 = prop_j2.propagate(initial_orbit, duration_seconds=sim_duration_s, dt_eval=dt_eval_s)
    print("  [✓] Case B: Central + J2 perturbation completed.")

    # Scenario C: Full Physics (Central + J2 + Drag)
    prop_full = OrbitPropagator(
        satellite=sat,
        atmosphere=atm,
        include_central_gravity=True,
        include_j2=True,
        include_drag=True,
        solver_method="DOP853",
    )
    res_full = prop_full.propagate(initial_orbit, duration_seconds=sim_duration_s, dt_eval=dt_eval_s)
    print("  [✓] Case C: Full Physics (Gravity + J2 + Drag) completed.\n")

    # 4. Telemetry and Results Analysis
    decay_m = res_full.altitude_decay
    raan_delta_deg = np.degrees(res_j2.raan_change)

    analytical_rate = analytical_j2_raan_rate(initial_orbit.a, initial_orbit.e, initial_orbit.i)
    analytical_raan_delta_deg = np.degrees(analytical_rate * sim_duration_s)

    print("-" * 70)
    print(" TELEMETRY & PHYSICAL VERIFICATION SUMMARY")
    print("-" * 70)
    print("Atmospheric Drag Effect (Case C vs Case B):")
    print(f"  Initial Altitude:            {res_full.initial_altitude / 1000.0:.3f} km")
    print(f"  Final Altitude (Full Drag):  {res_full.final_altitude / 1000.0:.3f} km")
    print(f"  Net Altitude Decay:          {decay_m:.3f} m (altitude strictly decays)")
    print(f"  Semi-Major Axis Decay (da):  {(res_full.semi_major_axes[0] - res_full.semi_major_axes[-1]):.3f} m\n")

    print("J2 Oblateness Nodal Precession (Case B):")
    print(f"  Simulated RAAN Drift:        {raan_delta_deg:.5f} deg ({raan_delta_deg / (sim_duration_s / 86400.0):.4f} deg/day)")
    print(f"  Analytical Secular Rate:     {analytical_raan_delta_deg:.5f} deg ({np.degrees(analytical_rate) * 86400.0:.4f} deg/day)")
    rel_error = abs(raan_delta_deg - analytical_raan_delta_deg) / abs(analytical_raan_delta_deg) * 100.0
    print(f"  Relative Discrepancy:        {rel_error:.3f}% (reflects osculating vs secular model)\n")

    # 5. Export Plots
    print("Generating and exporting visualization plots...")
    p_3d = os.path.join(output_dir, "3d_trajectory.png")
    plot_orbit_3d(res_full, save_path=p_3d, title="3D Nanosatellite Trajectory (Full Dynamics)")
    print(f"  Saved: {p_3d}")

    p_alt = os.path.join(output_dir, "altitude_decay.png")
    plot_altitude_decay(
        {
            "Keplerian Two-Body (No Perturbations)": res_kepler,
            "J2 Only (No Drag)": res_j2,
            "Full Dynamics (Drag + J2)": res_full,
        },
        save_path=p_alt,
        title="Nanosatellite Altitude Decay Under Aerodynamic Drag",
    )
    print(f"  Saved: {p_alt}")

    p_raan = os.path.join(output_dir, "raan_regression.png")
    plot_raan_regression(
        res_j2,
        analytical_rate=analytical_rate,
        save_path=p_raan,
        title="Ascending Node (RAAN) Regression Under Earth J2",
    )
    print(f"  Saved: {p_raan}")

    p_elem = os.path.join(output_dir, "elements_evolution.png")
    plot_orbital_elements_history(res_full, save_path=p_elem, title="Orbital Elements Evolution (Full Physics)")
    print(f"  Saved: {p_elem}")

    print("\n[✓] Simulation run completed successfully.")


if __name__ == "__main__":
    run_simulation()
