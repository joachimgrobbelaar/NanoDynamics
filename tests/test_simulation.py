"""Test runner redirecting to root test_simulation.py."""

from test_simulation import (
    test_altitude_decays_over_time_drag,
    test_ascending_node_regresses_j2,
    test_drag_vs_no_drag_comparison,
    test_full_combined_simulation,
    test_j2_polar_orbit_zero_regression,
)

__all__ = [
    "test_altitude_decays_over_time_drag",
    "test_ascending_node_regresses_j2",
    "test_drag_vs_no_drag_comparison",
    "test_full_combined_simulation",
    "test_j2_polar_orbit_zero_regression",
]
