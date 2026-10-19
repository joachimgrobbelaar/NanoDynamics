"""Unit tests for Keplerian elements conversions and analytical rates."""

import numpy as np
import pytest

from leo_simulator.constants import MU_EARTH, R_EARTH
from leo_simulator.orbit.elements import (
    OrbitalElements,
    analytical_j2_raan_rate,
    circular_velocity,
    coe_to_rv,
    rv_to_coe,
)
from leo_simulator.orbit.satellite import Satellite


class TestElementsConversions:
    def test_round_trip_general_elliptical_inclined(self):
        # Semi-major axis 7000 km, e=0.05, i=45 deg, Omega=30 deg, omega=60 deg, nu=40 deg
        initial = OrbitalElements(
            a=7000e3,
            e=0.05,
            i=np.radians(45.0),
            raan=np.radians(30.0),
            arg_pe=np.radians(60.0),
            nu=np.radians(40.0),
        )

        r_vec, v_vec = coe_to_rv(initial)
        recovered = rv_to_coe(r_vec, v_vec)

        assert recovered.a == pytest.approx(initial.a, rel=1e-8)
        assert recovered.e == pytest.approx(initial.e, rel=1e-8)
        assert recovered.i == pytest.approx(initial.i, rel=1e-8)
        assert recovered.raan == pytest.approx(initial.raan, rel=1e-8)
        assert recovered.arg_pe == pytest.approx(initial.arg_pe, rel=1e-8)
        assert recovered.nu == pytest.approx(initial.nu, rel=1e-8)

    def test_circular_orbit_edge_case(self):
        # Circular orbit: e = 0
        circ = OrbitalElements(
            a=6800e3,
            e=0.0,
            i=np.radians(28.5),
            raan=np.radians(50.0),
            arg_pe=0.0,
            nu=np.radians(70.0),
        )
        r, v = coe_to_rv(circ)
        recovered = rv_to_coe(r, v)

        assert recovered.a == pytest.approx(circ.a, rel=1e-8)
        assert recovered.e == pytest.approx(0.0, abs=1e-8)
        assert recovered.i == pytest.approx(circ.i, rel=1e-8)
        assert recovered.raan == pytest.approx(circ.raan, rel=1e-8)

    def test_equatorial_orbit_edge_case(self):
        # Equatorial orbit: i = 0
        eq = OrbitalElements(
            a=7200e3,
            e=0.02,
            i=0.0,
            raan=0.0,
            arg_pe=np.radians(120.0),
            nu=np.radians(15.0),
        )
        r, v = coe_to_rv(eq)
        recovered = rv_to_coe(r, v)

        assert recovered.a == pytest.approx(eq.a, rel=1e-8)
        assert recovered.e == pytest.approx(eq.e, rel=1e-8)
        assert recovered.i == pytest.approx(0.0, abs=1e-8)
        assert recovered.arg_pe == pytest.approx(eq.arg_pe, rel=1e-8)
        assert recovered.nu == pytest.approx(eq.nu, rel=1e-8)

    def test_equatorial_retrograde_orbit_edge_case(self):
        # Equatorial retrograde: i = pi
        for test_arg_pe in [np.radians(30.0), np.radians(120.0), np.radians(210.0), np.radians(300.0)]:
            eq_retro = OrbitalElements(
                a=7200e3,
                e=0.02,
                i=np.pi,
                raan=0.0,
                arg_pe=test_arg_pe,
                nu=np.radians(45.0),
            )
            r, v = coe_to_rv(eq_retro)
            recovered = rv_to_coe(r, v)

            assert recovered.a == pytest.approx(eq_retro.a, rel=1e-8)
            assert recovered.e == pytest.approx(eq_retro.e, rel=1e-8)
            assert recovered.i == pytest.approx(np.pi, rel=1e-8)
            assert recovered.arg_pe == pytest.approx(eq_retro.arg_pe, rel=1e-6)
            assert recovered.nu == pytest.approx(eq_retro.nu, rel=1e-6)

    def test_circular_equatorial_retrograde_nu_resolution(self):
        # Circular equatorial retrograde: e = 0, i = pi
        for test_nu in [np.radians(45.0), np.radians(135.0), np.radians(225.0), np.radians(315.0)]:
            circ_retro = OrbitalElements(
                a=6900e3,
                e=0.0,
                i=np.pi,
                raan=0.0,
                arg_pe=0.0,
                nu=test_nu,
            )
            r, v = coe_to_rv(circ_retro)
            recovered = rv_to_coe(r, v)

            assert recovered.a == pytest.approx(circ_retro.a, rel=1e-8)
            assert recovered.e == pytest.approx(0.0, abs=1e-8)
            assert recovered.i == pytest.approx(np.pi, rel=1e-8)
            assert recovered.nu == pytest.approx(test_nu, rel=1e-6)

    def test_circular_velocity(self):
        alt = 400e3
        v_circ = circular_velocity(alt)
        r = R_EARTH + alt
        expected_v = np.sqrt(MU_EARTH / r)
        assert v_circ == pytest.approx(expected_v, rel=1e-12)


class TestAnalyticalRates:
    def test_j2_raan_rate_sign(self):
        # Prograde (i < 90 deg) -> regression (rate < 0)
        rate_prograde = analytical_j2_raan_rate(a=7000e3, e=0.01, i=np.radians(51.6))
        assert rate_prograde < 0.0

        # Polar (i = 90 deg) -> zero precession
        rate_polar = analytical_j2_raan_rate(a=7000e3, e=0.01, i=np.radians(90.0))
        assert rate_polar == pytest.approx(0.0, abs=1e-18)

        # Retrograde (i > 90 deg) -> progression (rate > 0)
        rate_retrograde = analytical_j2_raan_rate(a=7000e3, e=0.01, i=np.radians(120.0))
        assert rate_retrograde > 0.0


class TestSatelliteClass:
    def test_satellite_factory_methods(self):
        sat1u = Satellite.cubesat_1u()
        assert sat1u.mass == 1.33
        assert sat1u.drag_area == 0.01
        assert sat1u.ballistic_coefficient == pytest.approx(1.33 / (2.2 * 0.01))

        sat3u = Satellite.cubesat_3u()
        assert sat3u.mass == 4.0
        assert sat3u.drag_area == 0.03

    def test_satellite_non_finite_parameters(self):
        with pytest.raises(ValueError):
            Satellite(mass=float("nan"))
        with pytest.raises(ValueError):
            Satellite(drag_area=float("nan"))
        with pytest.raises(ValueError):
            Satellite(cd=float("nan"))

    def test_elements_non_finite_validation(self):
        # coe_to_rv with nan
        for field in ["a", "e", "i", "raan", "arg_pe", "nu"]:
            kwargs = {
                "a": 7000e3,
                "e": 0.01,
                "i": 0.5,
                "raan": 0.5,
                "arg_pe": 0.5,
                "nu": 0.5,
            }
            kwargs[field] = float("nan")
            elem = OrbitalElements(**kwargs)
            with pytest.raises(ValueError):
                coe_to_rv(elem)

        # rv_to_coe with invalid shape or nan
        with pytest.raises(ValueError):
            rv_to_coe([1.0, 2.0], [0.0, 7500.0, 0.0])
        with pytest.raises(ValueError):
            rv_to_coe([np.nan, 0.0, 0.0], [0.0, 7500.0, 0.0])
        with pytest.raises(ValueError):
            rv_to_coe([7000e3, 0.0, 0.0], [0.0, np.nan, 0.0])

    def test_round_trip_cardinal_and_boundary_angles(self):
        # Cardinal angles and 2*pi boundary
        for ang in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]:
            elem = OrbitalElements(7000e3, 0.05, np.radians(45.0), 0.5, 1.2, ang)
            r, v = coe_to_rv(elem)
            rec = rv_to_coe(r, v)
            r2, v2 = coe_to_rv(rec)
            # Must achieve sub-micrometer roundtrip consistency without arccos sqrt(eps) precision loss
            assert np.linalg.norm(r2 - r) < 1e-6
            assert np.linalg.norm(v2 - v) < 1e-6

    def test_analytical_j2_raan_rate_validation(self):
        with pytest.raises(ValueError):
            analytical_j2_raan_rate(a=float("nan"), e=0.01, i=0.5)
        with pytest.raises(ValueError):
            analytical_j2_raan_rate(a=7000e3, e=float("nan"), i=0.5)
        with pytest.raises(ValueError):
            analytical_j2_raan_rate(a=-7000e3, e=0.01, i=0.5)
        with pytest.raises(ValueError):
            analytical_j2_raan_rate(a=7000e3, e=1.05, i=0.5)
        with pytest.raises(ValueError):
            analytical_j2_raan_rate(a=7000e3, e=0.01, i=0.5, mu=float("nan"))

    def test_circular_velocity_validation(self):
        with pytest.raises(ValueError):
            circular_velocity(float("nan"))
        with pytest.raises(ValueError):
            circular_velocity(-R_EARTH - 100.0)  # Center/subterranean singularity
        with pytest.raises(ValueError):
            circular_velocity(400e3, mu=-1.0)

    def test_orbital_elements_period_and_mean_motion_non_finite(self):
        elem_nan = OrbitalElements(a=float("nan"), e=0.01, i=0.5, raan=0.0, arg_pe=0.0, nu=0.0)
        assert elem_nan.period == float("inf")
        assert elem_nan.mean_motion == 0.0
        elem_neg = OrbitalElements(a=-7000e3, e=0.01, i=0.5, raan=0.0, arg_pe=0.0, nu=0.0)
        assert elem_neg.period == float("inf")
        assert elem_neg.mean_motion == 0.0
