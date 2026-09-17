"""Unit tests for visualization routines and error handling."""

import numpy as np
import pytest

from leo_simulator.orbit.elements import OrbitalElements
from leo_simulator.propagator import OrbitPropagator, PropagationResult
from leo_simulator.visualization import (
    plot_altitude_decay,
    plot_orbit_3d,
    plot_orbital_elements_history,
    plot_raan_regression,
)


@pytest.fixture
def sample_result() -> PropagationResult:
    prop = OrbitPropagator(include_central_gravity=True, include_j2=True, include_drag=False)
    orbit = OrbitalElements(a=7000e3, e=0.001, i=0.5, raan=0.2, arg_pe=0.0, nu=0.0)
    return prop.propagate(orbit, duration_seconds=300.0, dt_eval=30.0)


@pytest.fixture
def empty_result() -> PropagationResult:
    return PropagationResult(
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
        message="Empty result",
    )


def test_plot_empty_result_error_handling(empty_result):
    with pytest.raises(ValueError, match="Cannot plot empty"):
        plot_orbit_3d(empty_result)

    with pytest.raises(ValueError, match="Cannot plot empty"):
        plot_raan_regression(empty_result)

    with pytest.raises(ValueError, match="Cannot plot empty"):
        plot_orbital_elements_history(empty_result)

    with pytest.raises(ValueError, match="contains no trajectory points"):
        plot_altitude_decay(empty_result)

    with pytest.raises(ValueError, match="No propagation results provided"):
        plot_altitude_decay({})


def test_plot_raan_regression_with_nonzero_t_start():
    # Verify that analytical regression line starts exactly at initial RAAN when t_start > 0
    prop = OrbitPropagator(include_central_gravity=True, include_j2=True, include_drag=False)
    orbit = OrbitalElements(a=7000e3, e=0.001, i=np.radians(51.6), raan=0.5, arg_pe=0.0, nu=0.0)
    t_start = 7200.0  # 2 hours offset
    res = prop.propagate(orbit, duration_seconds=600.0, t_start=t_start, dt_eval=60.0)

    fig = plot_raan_regression(res, analytical_rate=-1e-6)
    ax = fig.gca()
    lines = ax.get_lines()
    assert len(lines) == 2
    num_line, ana_line = lines[0], lines[1]
    # At t = 0 (start of plot), numerical and analytical values must coincide with res.raans[0]
    expected_start_deg = np.degrees(res.raans[0])
    np.testing.assert_allclose(num_line.get_ydata()[0], expected_start_deg, rtol=1e-6)
    np.testing.assert_allclose(ana_line.get_ydata()[0], expected_start_deg, rtol=1e-6)


def test_valid_plots_generation(sample_result, tmp_path):
    p3d = str(tmp_path / "3d.png")
    fig1 = plot_orbit_3d(sample_result, save_path=p3d)
    assert fig1 is not None

    p_alt = str(tmp_path / "alt.png")
    fig2 = plot_altitude_decay(sample_result, save_path=p_alt)
    assert fig2 is not None

    p_raan = str(tmp_path / "raan.png")
    fig3 = plot_raan_regression(sample_result, analytical_rate=-1e-6, save_path=p_raan)
    assert fig3 is not None

    p_hist = str(tmp_path / "hist.png")
    fig4 = plot_orbital_elements_history(sample_result, save_path=p_hist)
    assert fig4 is not None
