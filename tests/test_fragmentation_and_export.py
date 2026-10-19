import numpy as np
import pytest
from leo_simulator.orbit.collision import (
    compute_orbital_collision_impulse,
    generate_collision_debris_fragments,
)
from leo_simulator.orbit.astrodynamics import (
    rv_to_keplerian,
    compute_vnc_basis,
    compute_rf_link_budget,
    compute_space_environment,
)


def test_debris_fragmentation_momentum_conservation():
    m_parent = 100.0
    area_parent = 1.0
    cd_parent = 2.2
    pos = [7000000.0, 0.0, 0.0]
    vel = [0.0, 7500.0, 0.0]
    rel_speed = 500.0

    core_spec, daughters = generate_collision_debris_fragments(
        sat_name="SatTest",
        parent_mass=m_parent,
        parent_drag_area=area_parent,
        parent_cd=cd_parent,
        col_pos=pos,
        post_vel=vel,
        relative_speed=rel_speed,
        num_fragments=2,
        core_mass_fraction=0.50,
    )

    assert len(daughters) == 2
    assert core_spec["mass"] == pytest.approx(50.0)
    assert daughters[0]["mass"] == pytest.approx(25.0)
    assert daughters[1]["mass"] == pytest.approx(25.0)

    # Verify zero-sum net delta-v momentum dispersion
    v_base = np.array(vel)
    delta_p = np.zeros(3)
    for d in daughters:
        v_d = np.array(d["initial_state"][3:])
        delta_v = v_d - v_base
        delta_p += d["mass"] * delta_v

    assert np.linalg.norm(delta_p) < 1e-4


def test_vnc_basis_orthogonality():
    r = np.array([6800000.0, 0.0, 0.0])
    v = np.array([0.0, 7600.0, 0.0])
    u_v, u_n, u_c = compute_vnc_basis(r, v)

    # Unit norms
    assert np.linalg.norm(u_v) == pytest.approx(1.0)
    assert np.linalg.norm(u_n) == pytest.approx(1.0)
    assert np.linalg.norm(u_c) == pytest.approx(1.0)

    # Mutual orthogonality
    assert np.dot(u_v, u_n) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(u_v, u_c) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(u_n, u_c) == pytest.approx(0.0, abs=1e-10)


def test_24_variable_analytical_components():
    r = np.array([6828137.0, 0.0, 0.0])
    v = np.array([0.0, 4744.0, 5985.0])
    mu = 3.986004418e14

    kep = rv_to_keplerian(r, v, mu=mu)
    assert kep["a"] > 6000000.0
    assert 0.0 <= kep["e"] < 1.0
    assert 0.0 <= kep["inc_deg"] <= 180.0

    rf = compute_rf_link_budget(r, v)
    assert "doppler_shift_hz" in rf
    assert "fspl_db" in rf
    assert rf["fspl_db"] > 100.0

    env = compute_space_environment(lat_deg=0.0, lon_deg=0.0, alt_km=450.0)
    assert env["tec_tecu"] > 0.0
    assert env["iono_delay_ns"] > 0.0
    assert env["seu_rate_per_hr"] > 0.0
