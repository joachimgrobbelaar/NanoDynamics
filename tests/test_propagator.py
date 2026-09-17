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

    def test_non_finite_duration_and_epoch_validation(self):
        prop = OrbitPropagator()
        r0 = np.array([R_EARTH + 400e3, 0.0, 0.0])
        v0 = np.array([0.0, 7670.0, 0.0])
        state = np.concatenate([r0, v0])

        for invalid_duration in [float("nan"), float("inf"), float("-inf"), 0.0]:
            with pytest.raises(ValueError):
                prop.propagate(state, duration_seconds=invalid_duration)

        for invalid_t_start in [float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                prop.propagate(state, duration_seconds=100.0, t_start=invalid_t_start)

    def test_invalid_dt_eval_and_non_finite_initial_state(self):
        prop = OrbitPropagator()
        r0 = np.array([R_EARTH + 400e3, 0.0, 0.0])
        v0 = np.array([0.0, 7670.0, 0.0])
        state = np.concatenate([r0, v0])

        for invalid_dt in [-1.0, 0.0, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                prop.propagate(state, duration_seconds=100.0, dt_eval=invalid_dt)

        nan_state = state.copy()
        nan_state[0] = np.nan
        with pytest.raises(ValueError):
            prop.propagate(nan_state, duration_seconds=100.0)

    def test_empty_propagation_result_properties(self):
        from leo_simulator.propagator import PropagationResult

        empty_res = PropagationResult(
            t=np.array([]),
            r=np.empty((0, 3)),
            v=np.empty((0, 3)),
            altitudes=np.array([]),
            speeds=np.array([]),
            semi_major_axes=np.array([]),
            eccentricities=np.array([]),
            inclinations=np.array([]),
            raans=np.array([]),
            arg_pes=np.array([]),
            true_anomalies=np.array([]),
            success=False,
            status=-1,
            message="Failed before integration",
        )

        with pytest.raises(RuntimeError):
            _ = empty_res.final_altitude

        with pytest.raises(RuntimeError):
            _ = empty_res.initial_altitude

        with pytest.raises(RuntimeError):
            _ = empty_res.final_raan

        with pytest.raises(RuntimeError):
            _ = empty_res.raan_change

        with pytest.raises(RuntimeError):
            _ = empty_res.semi_major_axis_decay

    def test_reentry_event_termination_with_dt_eval(self):
        # Verify that terminal re-entry point is included even when dt_eval grid would cut off before event
        prop = OrbitPropagator(
            include_central_gravity=True,
            include_j2=False,
            include_drag=False,
            min_altitude_reentry=0.0,
        )
        suborbital = OrbitalElements(
            a=R_EARTH + 50e3,
            e=0.1,
            i=0.0,
            raan=0.0,
            arg_pe=0.0,
            nu=np.radians(180.0),
        )
        result = prop.propagate(suborbital, duration_seconds=10000.0, dt_eval=10.0)
        assert result.reentry_detected
        assert result.reentry_time is not None
        # Trajectory endpoint must match the re-entry event
        assert result.t[-1] == pytest.approx(result.reentry_time, abs=1e-6)
        assert result.final_altitude == pytest.approx(0.0, abs=1.0)

    def test_subterranean_initial_state_rejected(self):
        prop = OrbitPropagator()
        # Initial position inside Earth (altitude = -100 m)
        r0 = np.array([R_EARTH - 100.0, 0.0, 0.0])
        v0 = np.array([0.0, 7500.0, 0.0])
        state = np.concatenate([r0, v0])
        with pytest.raises(ValueError, match="at or below the minimum re-entry altitude"):
            prop.propagate(state, duration_seconds=100.0)

    def test_zero_effective_duration_rejected(self):
        prop = OrbitPropagator()
        r0 = np.array([R_EARTH + 400e3, 0.0, 0.0])
        v0 = np.array([0.0, 7670.0, 0.0])
        state = np.concatenate([r0, v0])
        with pytest.raises(ValueError, match="Effective duration .* is zero"):
            prop.propagate(state, duration_seconds=1e-4, t_start=1e16)

    def test_propagator_invalid_init_parameters(self):
        with pytest.raises(ValueError):
            OrbitPropagator(rtol=-1.0)
        with pytest.raises(ValueError):
            OrbitPropagator(rtol=float("nan"))
        with pytest.raises(ValueError):
            OrbitPropagator(atol=float("nan"))
        with pytest.raises(ValueError):
            OrbitPropagator(mu=float("nan"))
        with pytest.raises(ValueError):
            OrbitPropagator(r_earth=float("nan"))
        with pytest.raises(ValueError):
            OrbitPropagator(j2=float("nan"))
        with pytest.raises(ValueError):
            OrbitPropagator(omega_earth=float("nan"))
        with pytest.raises(ValueError):
            OrbitPropagator(min_altitude_reentry=float("nan"))

    def test_semi_major_axis_decay_property(self):
        sat = Satellite.cubesat_3u(mass=4.0, cd=2.2)
        prop = OrbitPropagator(satellite=sat, include_central_gravity=True, include_j2=False, include_drag=True)
        orbit = OrbitalElements(a=R_EARTH + 350e3, e=0.001, i=0.5, raan=0.0, arg_pe=0.0, nu=0.0)
        res = prop.propagate(orbit, duration_seconds=1000.0)
        assert res.semi_major_axis_decay > 0.0
        assert res.semi_major_axis_decay == pytest.approx(res.semi_major_axes[0] - res.semi_major_axes[-1])
