"""Unit tests for physical force and acceleration models."""

import numpy as np
import pytest

from leo_simulator.constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH
from leo_simulator.models.drag import (
    ExponentialAtmosphere,
    PiecewiseExponentialAtmosphere,
    aerodynamic_drag_acceleration,
    relative_velocity_vector,
)
from leo_simulator.models.dynamics import OrbitalDynamics
from leo_simulator.models.gravity import (
    central_gravity_acceleration,
    j2_perturbation_acceleration,
    total_gravity_acceleration,
)


class TestGravityModels:
    def test_central_gravity_magnitude_and_direction(self):
        r = np.array([7000e3, 0.0, 0.0])
        a = central_gravity_acceleration(r)

        # Must point along -x
        assert a[0] < 0.0
        assert a[1] == 0.0
        assert a[2] == 0.0

        expected_mag = MU_EARTH / (7000e3**2)
        np.testing.assert_allclose(np.linalg.norm(a), expected_mag, rtol=1e-12)

    def test_central_gravity_invalid_inputs(self):
        with pytest.raises(ValueError):
            central_gravity_acceleration([0.0, 0.0, 0.0])

        with pytest.raises(ValueError):
            central_gravity_acceleration([1000.0, 2000.0])

        with pytest.raises(ValueError):
            central_gravity_acceleration([np.nan, 0.0, 0.0])

        with pytest.raises(ValueError):
            j2_perturbation_acceleration([np.nan, 0.0, 0.0])

    def test_j2_perturbation_equator(self):
        # At equator (z=0), perturbation should be radially inward
        r_eq = np.array([7000e3, 0.0, 0.0])
        a_j2 = j2_perturbation_acceleration(r_eq)

        assert a_j2[0] < 0.0
        assert a_j2[1] == 0.0
        assert a_j2[2] == 0.0

        # Exact formula at equator: - 1.5 * J2 * mu * R_E^2 / r^4
        expected_ax = -1.5 * J2_EARTH * MU_EARTH * (R_EARTH**2) / (7000e3**4)
        np.testing.assert_allclose(a_j2[0], expected_ax, rtol=1e-12)

    def test_j2_perturbation_poles(self):
        # At pole (x=0, y=0, z=r), z^2/r^2 = 1 => 3 - 5 = -2
        # factor * z * (-2) = positive acceleration in +z direction (outward)
        r_pole = np.array([0.0, 0.0, 7000e3])
        a_j2 = j2_perturbation_acceleration(r_pole)

        assert a_j2[0] == 0.0
        assert a_j2[1] == 0.0
        assert a_j2[2] > 0.0

        expected_az = 3.0 * J2_EARTH * MU_EARTH * (R_EARTH**2) / (7000e3**4)
        np.testing.assert_allclose(a_j2[2], expected_az, rtol=1e-12)

    def test_total_gravity_combination(self):
        r = np.array([5000e3, 3000e3, 2000e3])
        a_cent = central_gravity_acceleration(r)
        a_j2 = j2_perturbation_acceleration(r)
        a_tot = total_gravity_acceleration(r, include_j2=True)
        np.testing.assert_allclose(a_tot, a_cent + a_j2, rtol=1e-14)


class TestDragModels:
    def test_exponential_atmosphere_density(self):
        atm = ExponentialAtmosphere(h0=400e3, rho0=2.8e-12, scale_height=50e3)
        # At h = h0
        assert atm.density(400e3) == pytest.approx(2.8e-12)
        # At h = h0 + H
        assert atm.density(450e3) == pytest.approx(2.8e-12 * np.exp(-1.0))
        # At h = h0 - H
        assert atm.density(350e3) == pytest.approx(2.8e-12 * np.exp(1.0))
        # Clamped at min_altitude for negative altitudes
        assert atm.density(-1000.0) == atm.density(0.0)

    def test_piecewise_atmosphere_continuity(self):
        atm = PiecewiseExponentialAtmosphere()
        altitudes = np.linspace(100e3, 800e3, 15)
        densities = [atm.density(h) for h in altitudes]
        # Must be strictly positive and strictly decreasing
        for i in range(len(densities) - 1):
            assert densities[i] > 0.0
            assert densities[i] > densities[i + 1]

    def test_relative_velocity(self):
        r = np.array([7000e3, 0.0, 0.0])
        v = np.array([0.0, 7500.0, 0.0])
        # omega_E x r = [0, 0, omega] x [x, 0, 0] = [0, omega*x, 0]
        v_rel = relative_velocity_vector(r, v, omega_earth=OMEGA_EARTH, include_earth_rotation=True)
        expected_vy_rel = 7500.0 - OMEGA_EARTH * 7000e3
        np.testing.assert_allclose(v_rel, [0.0, expected_vy_rel, 0.0], rtol=1e-12)

    def test_aerodynamic_drag_direction_and_magnitude(self):
        r = np.array([R_EARTH + 300e3, 0.0, 0.0])
        v = np.array([0.0, 7700.0, 0.0])
        cd = 2.2
        area = 0.03
        mass = 4.0

        a_drag = aerodynamic_drag_acceleration(
            r, v, cd=cd, area=area, mass=mass, include_earth_rotation=False
        )

        # Opposite to velocity (+y => drag is -y)
        assert a_drag[1] < 0.0
        assert a_drag[0] == 0.0
        assert a_drag[2] == 0.0

    def test_drag_invalid_inputs(self):
        r = np.array([R_EARTH + 300e3, 0.0, 0.0])
        v = np.array([0.0, 7700.0, 0.0])
        with pytest.raises(ValueError):
            aerodynamic_drag_acceleration(r, v, cd=2.2, area=0.03, mass=-1.0)
        with pytest.raises(ValueError):
            aerodynamic_drag_acceleration(r, v, cd=-1.0, area=0.03, mass=4.0)
        with pytest.raises(ValueError):
            aerodynamic_drag_acceleration([0.0, 0.0, 0.0], v, cd=2.2, area=0.03, mass=4.0)
        with pytest.raises(ValueError):
            aerodynamic_drag_acceleration([np.nan, 0.0, 0.0], v, cd=2.2, area=0.03, mass=4.0)

    def test_relative_velocity_invalid_inputs(self):
        r = np.array([7000e3, 0.0, 0.0])
        v = np.array([0.0, 7500.0, 0.0])
        with pytest.raises(ValueError):
            relative_velocity_vector([1.0, 2.0], v)
        with pytest.raises(ValueError):
            relative_velocity_vector(r, [1.0, 2.0])
        with pytest.raises(ValueError):
            relative_velocity_vector([np.nan, 0.0, 0.0], v)

    def test_atmosphere_non_finite_altitude(self):
        exp_atm = ExponentialAtmosphere()
        with pytest.raises(ValueError):
            exp_atm.density(float("nan"))
        piece_atm = PiecewiseExponentialAtmosphere()
        with pytest.raises(ValueError):
            piece_atm.density(float("nan"))


class TestDynamicsCombination:
    def test_dynamics_state_derivatives(self):
        dyn = OrbitalDynamics()
        state = np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
        dstate = dyn.derivatives(0.0, state)

        # Position derivative is velocity
        np.testing.assert_allclose(dstate[0:3], state[3:6])
        # Velocity derivative is acceleration
        assert dstate[3] < 0.0  # gravity pulls inward (-x)
