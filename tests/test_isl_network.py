"""Unit tests for Inter-Satellite Laser Communication Links (ISL)."""

import numpy as np
import pytest

from leo_simulator.constants import R_EARTH
from leo_simulator.orbit.isl_network import (
    ISLNetworkMesh,
    SatelliteNode,
    check_line_of_sight,
)


def test_line_of_sight_unobstructed():
    # Two satellites in LEO at 500 km altitude, separated by 1000 km along y-axis
    r = R_EARTH + 500_000.0
    pos_a = np.array([r, 0.0, 0.0], dtype=np.float64)
    pos_b = np.array([r, 1_000_000.0, 0.0], dtype=np.float64)

    visible, dist_m, tangent_alt_m = check_line_of_sight(pos_a, pos_b)
    assert visible is True
    assert dist_m == pytest.approx(1_000_000.0, abs=1.0)
    assert tangent_alt_m > 400_000.0


def test_line_of_sight_earth_occlusion():
    # Two satellites on exact opposite sides of Earth
    r = R_EARTH + 500_000.0
    pos_a = np.array([r, 0.0, 0.0], dtype=np.float64)
    pos_b = np.array([-r, 0.0, 0.0], dtype=np.float64)

    visible, dist_m, tangent_alt_m = check_line_of_sight(pos_a, pos_b, max_range_m=20_000_000.0)
    assert visible is False
    # The line passes through Earth core (tangent altitude = -R_EARTH)
    assert tangent_alt_m < 0.0


def test_line_of_sight_max_range():
    # Satellites separated by 6000 km (exceeds default 5000 km limit)
    pos_a = np.array([R_EARTH + 1000_000.0, 0.0, 0.0], dtype=np.float64)
    pos_b = np.array([R_EARTH + 1000_000.0, 6_000_000.0, 0.0], dtype=np.float64)

    visible, dist_m, _ = check_line_of_sight(pos_a, pos_b, max_range_m=5_000_000.0)
    assert visible is False
    assert dist_m > 5_000_000.0


def test_multi_hop_dijkstra_routing():
    mesh = ISLNetworkMesh(max_range_m=3_000_000.0)

    # 3-satellite chain: SatA -> SatB -> SatC (SatA cannot reach SatC directly)
    r = R_EARTH + 500_000.0
    nodes = [
        SatelliteNode(id="sat-A", name="Sat A", position_m=np.array([r, 0.0, 0.0])),
        SatelliteNode(id="sat-B", name="Sat B", position_m=np.array([r, 2_000_000.0, 0.0])),
        SatelliteNode(id="sat-C", name="Sat C", position_m=np.array([r, 4_000_000.0, 0.0])),
    ]

    path = mesh.route_packet("sat-A", "sat-C", nodes)
    assert path is not None
    assert path.hops == ["sat-A", "sat-B", "sat-C"]
    assert path.total_distance_km == pytest.approx(4000.0, abs=10.0)
    assert path.propagation_delay_ms > 10.0
    assert path.total_delay_ms > path.propagation_delay_ms  # Includes hop processing overhead
