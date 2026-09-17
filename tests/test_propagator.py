"""Unit tests for numerical orbit propagator and conservation laws."""

import numpy as np
import pytest

from leo_simulator.constants import MU_EARTH, R_EARTH
from leo_simulator.orbit.elements import OrbitalElements
from leo_simulator.orbit.satellite import Satellite
from leo_simulator.propagator import OrbitPropagator


class TestOrbitPropagator:
    def test_keplerian_energy_and_momentum_conservation(self):
        # In pure central gravity, energy and angular momentum must be conserved
        prop = OrbitPropagator(
            satellite=Satellite.cubesat_3u(),
            include_central_gravity=True,
            include_j2=False,
            include_drag=False,
            solver_method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        )

        initial_orbit = OrbitalElements(
            a=R_EARTH + 500e3,
            e=0.01,
            i=np.radians(30.0),
            raan=np.radians(10.0),
            arg_pe=np.radians(20.0),
            nu=np.radians(0.0),
        )

        duration = 2.0 * initial_orbit.period  # 2 full orbits (~3 hours)
        result = prop.propagate(initial_orbit, duration_seconds=duration, dt_eval=60.0)

        assert result.success

        # Compute specific orbital energy along trajectory
        # epsilon = v^2 / 2 - mu / r
        r_mags = np.linalg.norm(result.r, axis=1)
        v_mags = np.linalg.norm(result.v, axis=1)
        energies = (v_mags**2) / 2.0 - MU_EARTH / r_mags

        # Energy conservation check (relative change < 1e-8)
        energy_rel_variation = (np.max(energies) - np.min(energies)) / np.abs(energies[0])
        assert energy_rel_variation < 1e-8

        # Angular momentum h = r x v conservation check
        h_vecs = np.cross(result.r, result.v)
        h_mags = np.linalg.norm(h_vecs, axis=1)
        h_rel_variation = (np.max(h_mags) - np.min(h_mags)) / h_mags[0]
        assert h_rel_variation < 1e-8

    def test_reentry_event_termination(self):
        # Trajectory initialized with low periapsis entering dense atmosphere
        prop = OrbitPropagator(
            satellite=Satellite.cubesat_3u(),
            include_central_gravity=True,
            include_j2=False,
            include_drag=False,
            min_altitude_reentry=0.0,  # Earth surface
        )

        # Orbit with periapsis below Earth surface
        suborbital = OrbitalElements(
            a=R_EARTH + 50e3,  # semi-major axis
            e=0.1,             # rp = a * (1 - 0.1) = 0.9 * a < R_EARTH
            i=0.0,
            raan=0.0,
            arg_pe=0.0,
            nu=np.radians(180.0),  # Start at apoapsis (highest point)
        )

        result = prop.propagate(suborbital, duration_seconds=10000.0)
        assert result.reentry_detected
        assert result.reentry_time is not None
        # Final altitude should be near 0 m (surface)
        assert result.final_altitude == pytest.approx(0.0, abs=10.0)

    def test_solver_method_rk45_compatibility(self):
        # Verify RK45 solver works and yields consistent result
        prop_rk45 = OrbitPropagator(
            include_central_gravity=True,
            include_j2=False,
            include_drag=False,
            solver_method="RK45",
        )

        orbit = OrbitalElements(
            a=R_EARTH + 400e3,
            e=0.001,
            i=np.radians(45.0),
            raan=0.0,
            arg_pe=0.0,
            nu=0.0,
        )

        res = prop_rk45.propagate(orbit, duration_seconds=1000.0)
        assert res.success
        assert len(res.t) > 1

    def test_fractional_dt_eval_sorting_stability(self):
        # Durations and fractional dt_eval that would cause duplicate end timestamps in naive arange
        prop = OrbitPropagator(include_central_gravity=True, include_j2=False, include_drag=False)
        orbit = OrbitalElements(
            a=R_EARTH + 400e3,
            e=0.001,
            i=np.radians(45.0),
            raan=0.0,
            arg_pe=0.0,
            nu=0.0,
        )

        for n in [6, 9, 21, 24]:
            dt = 1.0 / n
            res = prop.propagate(orbit, duration_seconds=1.0, dt_eval=dt)
            assert res.success
            assert np.all(np.diff(res.t) > 0)
            assert res.t[-1] == pytest.approx(1.0, abs=1e-12)

    def test_raw_state_vector_input_and_invalid_duration(self):
        prop = OrbitPropagator()
        r0 = np.array([R_EARTH + 400e3, 0.0, 0.0])
        v0 = np.array([0.0, 7670.0, 0.0])
        raw_state = list(np.concatenate([r0, v0]))

        # Should accept list input
        res = prop.propagate(raw_state, duration_seconds=100.0)
        assert res.success
        assert len(res.t) > 1

        # Should raise ValueError on negative duration
        with pytest.raises(ValueError):
            prop.propagate(raw_state, duration_seconds=-10.0)
