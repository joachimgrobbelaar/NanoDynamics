"""Inter-Satellite Laser Communication Links (ISL) and Dynamic Mesh Network.

Computes line-of-sight (LOS) visibility between satellites considering Earth
atmospheric limb occlusion, evaluates multi-hop network paths, and models
packet latency and link budget metrics.
"""

from dataclasses import dataclass
import heapq
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from leo_simulator.constants import R_EARTH

SPEED_OF_LIGHT = 299792458.0  # m/s
DEFAULT_ATM_OCCLUSION_ALT_M = 100_000.0  # 100 km Karman line limb
DEFAULT_MAX_ISL_RANGE_M = 5_000_000.0   # 5000 km optical laser link limit


@dataclass
class SatelliteNode:
    id: str
    name: str
    position_m: np.ndarray  # [x, y, z] in ECI (meters)
    parent_body: str = "Earth"


@dataclass
class LinkPath:
    source_id: str
    target_id: str
    hops: List[str]
    total_distance_km: float
    propagation_delay_ms: float
    total_delay_ms: float
    achievable_bitrate_mbps: float


def check_line_of_sight(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    body_radius_m: float = R_EARTH,
    atm_limb_m: float = DEFAULT_ATM_OCCLUSION_ALT_M,
    max_range_m: float = DEFAULT_MAX_ISL_RANGE_M,
) -> Tuple[bool, float, float]:
    """Evaluate whether an unobstructed optical LOS exists between two satellites.

    Args:
        pos_a: ECI position vector of Satellite A (meters).
        pos_b: ECI position vector of Satellite B (meters).
        body_radius_m: Central body radius (meters).
        atm_limb_m: Atmospheric limb clearance height (meters).
        max_range_m: Maximum optical laser communication range (meters).

    Returns:
        is_visible: True if direct line-of-sight is open.
        distance_m: Distance between satellites in meters.
        tangent_alt_m: Minimum clearance altitude above surface in meters.
    """
    diff = pos_b - pos_a
    distance_m = float(np.linalg.norm(diff))

    if distance_m > max_range_m:
        return False, distance_m, 0.0

    if distance_m < 1e-3:
        return True, 0.0, float(np.linalg.norm(pos_a) - body_radius_m)

    # Parametric closest point on segment to body center (origin [0, 0, 0]):
    # r(t) = pos_a + t * diff, t in [0, 1]
    t_closest = -np.dot(pos_a, diff) / (distance_m ** 2)
    t_clamped = max(0.0, min(1.0, float(t_closest)))

    closest_pt = pos_a + t_clamped * diff
    closest_dist = float(np.linalg.norm(closest_pt))
    tangent_alt_m = closest_dist - body_radius_m

    is_visible = closest_dist >= (body_radius_m + atm_limb_m)
    return is_visible, distance_m, tangent_alt_m


class ISLNetworkMesh:
    """Dynamic mesh network modeling optical laser inter-satellite links."""

    def __init__(
        self,
        body_radius_m: float = R_EARTH,
        atm_limb_m: float = DEFAULT_ATM_OCCLUSION_ALT_M,
        max_range_m: float = DEFAULT_MAX_ISL_RANGE_M,
        hop_processing_delay_ms: float = 0.5,
    ):
        self.body_radius_m = body_radius_m
        self.atm_limb_m = atm_limb_m
        self.max_range_m = max_range_m
        self.hop_processing_delay_ms = hop_processing_delay_ms

    def build_topology(self, nodes: List[SatelliteNode]) -> Dict[str, Any]:
        """Compute the instantaneous full connectivity graph across all satellites."""
        n = len(nodes)
        edges = []
        adjacency: Dict[str, List[Tuple[str, float]]] = {node.id: [] for node in nodes}

        for i in range(n):
            for j in range(i + 1, n):
                node_a = nodes[i]
                node_b = nodes[j]

                # Only link satellites orbiting the same parent body
                if node_a.parent_body != node_b.parent_body:
                    continue

                visible, dist_m, tangent_alt = check_line_of_sight(
                    node_a.position_m,
                    node_b.position_m,
                    body_radius_m=self.body_radius_m,
                    atm_limb_m=self.atm_limb_m,
                    max_range_m=self.max_range_m,
                )

                if visible:
                    delay_ms = (dist_m / SPEED_OF_LIGHT) * 1000.0
                    edges.append({
                        "node_a": node_a.id,
                        "node_b": node_b.id,
                        "distance_km": dist_m / 1000.0,
                        "delay_ms": delay_ms,
                        "tangent_alt_km": tangent_alt / 1000.0,
                    })
                    adjacency[node_a.id].append((node_b.id, dist_m))
                    adjacency[node_b.id].append((node_a.id, dist_m))

        return {
            "node_count": n,
            "link_count": len(edges),
            "links": edges,
            "adjacency": adjacency,
        }

    def route_packet(
        self,
        source_id: str,
        target_id: str,
        nodes: List[SatelliteNode],
    ) -> Optional[LinkPath]:
        """Calculate the shortest multi-hop optical path (Dijkstra) between two satellites."""
        if source_id == target_id:
            return LinkPath(
                source_id=source_id,
                target_id=target_id,
                hops=[source_id],
                total_distance_km=0.0,
                propagation_delay_ms=0.0,
                total_delay_ms=0.0,
                achievable_bitrate_mbps=10000.0,
            )

        topo = self.build_topology(nodes)
        adj = topo["adjacency"]

        if source_id not in adj or target_id not in adj:
            return None

        # Dijkstra shortest path by distance
        dist: Dict[str, float] = {node.id: float("inf") for node in nodes}
        prev: Dict[str, Optional[str]] = {node.id: None for node in nodes}
        dist[source_id] = 0.0

        pq: List[Tuple[float, str]] = [(0.0, source_id)]

        while pq:
            curr_dist, u = heapq.heappop(pq)
            if curr_dist > dist[u]:
                continue
            if u == target_id:
                break

            for v, weight in adj.get(u, []):
                new_dist = curr_dist + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

        if dist[target_id] == float("inf"):
            return None

        # Reconstruct path
        path = []
        curr = target_id
        while curr is not None:
            path.append(curr)
            curr = prev[curr]
        path.reverse()

        tot_dist_km = dist[target_id] / 1000.0
        prop_delay_ms = (dist[target_id] / SPEED_OF_LIGHT) * 1000.0
        num_hops = max(0, len(path) - 1)
        total_delay_ms = prop_delay_ms + (num_hops * self.hop_processing_delay_ms)

        # Baseline 1550 nm optical inter-satellite link data rate
        # Degrades slightly with distance: 10 Gbps close, down to 1 Gbps at 5000 km
        bitrate_mbps = max(1000.0, 10000.0 * math.exp(-tot_dist_km / 10000.0))

        return LinkPath(
            source_id=source_id,
            target_id=target_id,
            hops=path,
            total_distance_km=tot_dist_km,
            propagation_delay_ms=prop_delay_ms,
            total_delay_ms=total_delay_ms,
            achievable_bitrate_mbps=bitrate_mbps,
        )
