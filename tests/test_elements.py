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
