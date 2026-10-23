"""Orbital collision mechanics and conjunction kinematics.

Calculates relative separation, closing range rates, 3D inelastic/elastic momentum
exchange impulses, and post-collision velocity deflections for orbiting bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from leo_simulator.constants import MU_EARTH


@dataclass
class CollisionResult:
    """Telemetry and kinematics outcome of an orbital collision."""

    v1_post: np.ndarray
    v2_post: np.ndarray
    dv1: np.ndarray
    dv2: np.ndarray
    dv1_mag: float
    dv2_mag: float
    impulse_mag: float
    relative_speed: float
    separation_distance: float
    restitution: float
    momentum_error: float


def detect_conjunction(
    r1: np.ndarray | list[float],
    r2: np.ndarray | list[float],
    v1: np.ndarray | list[float],
    v2: np.ndarray | list[float],
    threshold_m: float = 500.0,
) -> Tuple[bool, float, float]:
    """Detect whether two satellites are in conjunction within threshold_m and closing.

    Args:
        r1: Position vector of body 1 [m].
        r2: Position vector of body 2 [m].
        v1: Velocity vector of body 1 [m/s].
        v2: Velocity vector of body 2 [m/s].
        threshold_m: Distance threshold in meters to trigger collision [m].

    Returns:
        Tuple of (is_colliding: bool, distance_m: float, range_rate_m_s: float).
    """
    pos1 = np.asarray(r1, dtype=np.float64)
    pos2 = np.asarray(r2, dtype=np.float64)
    vel1 = np.asarray(v1, dtype=np.float64)
    vel2 = np.asarray(v2, dtype=np.float64)

    d_vec = pos1 - pos2
    dist = float(np.linalg.norm(d_vec))

    rel_vel = vel1 - vel2
    range_rate = float(np.dot(d_vec, rel_vel) / dist) if dist > 1e-6 else 0.0

    # Collision triggers when distance is within threshold and bodies are approaching
    is_colliding = (dist <= threshold_m) and (range_rate <= 0.0)
    return is_colliding, dist, range_rate


def compute_orbital_collision_impulse(
    m1: float,
    r1: np.ndarray | list[float],
    v1: np.ndarray | list[float],
    m2: float,
    r2: np.ndarray | list[float],
    v2: np.ndarray | list[float],
    restitution: float = 0.2,
) -> CollisionResult:
    """Compute 3D momentum exchange and post-collision velocities for two satellites.

    Strictly satisfies conservation of total linear momentum:
        m1 * v1 + m2 * v2 == m1 * v1_post + m2 * v2_post

    Args:
        m1: Mass of satellite 1 [kg] (> 0).
        r1: Position vector of satellite 1 [m].
        v1: Velocity vector of satellite 1 [m/s].
        m2: Mass of satellite 2 [kg] (> 0).
        r2: Position vector of satellite 2 [m].
        v2: Velocity vector of satellite 2 [m/s].
        restitution: Coefficient of restitution (0.0 = plastic, 1.0 = elastic).

    Returns:
        CollisionResult containing updated velocity vectors and impulse metrics.
    """
    if m1 <= 0.0 or m2 <= 0.0:
        raise ValueError(f"Satellite masses must be strictly positive, got m1={m1}, m2={m2}")

    e = float(np.clip(restitution, 0.0, 1.0))
    pos1 = np.asarray(r1, dtype=np.float64)
    pos2 = np.asarray(r2, dtype=np.float64)
    vel1 = np.asarray(v1, dtype=np.float64)
    vel2 = np.asarray(v2, dtype=np.float64)

    rel_pos = pos1 - pos2
    dist = float(np.linalg.norm(rel_pos))

    rel_vel = vel1 - vel2
    rel_speed = float(np.linalg.norm(rel_vel))

    # Collision normal points from body 2 to body 1
    if dist > 1e-5:
        n = rel_pos / dist
    elif rel_speed > 1e-6:
        # Fall back to anti-relative velocity direction if exact geometric co-location
        n = -rel_vel / rel_speed
    else:
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Relative normal approach speed
    vn = float(np.dot(rel_vel, n))

    # If already receding, no impulse applied
    if vn >= 0.0:
        return CollisionResult(
            v1_post=vel1.copy(),
            v2_post=vel2.copy(),
            dv1=np.zeros(3, dtype=np.float64),
            dv2=np.zeros(3, dtype=np.float64),
            dv1_mag=0.0,
            dv2_mag=0.0,
            impulse_mag=0.0,
            relative_speed=rel_speed,
            separation_distance=dist,
            restitution=e,
            momentum_error=0.0,
        )

    # Normal collision impulse scalar
    # J = -(1 + e) * vn / (1/m1 + 1/m2)
    j_scalar = -(1.0 + e) * vn / (1.0 / m1 + 1.0 / m2)
    impulse_vec = j_scalar * n

    v1_post = vel1 + impulse_vec / m1
    v2_post = vel2 - impulse_vec / m2

    dv1 = v1_post - vel1
    dv2 = v2_post - vel2

    # Verification: assert momentum conservation
    p_initial = m1 * vel1 + m2 * vel2
    p_post = m1 * v1_post + m2 * v2_post
    p_err = float(np.linalg.norm(p_post - p_initial))

    return CollisionResult(
        v1_post=v1_post,
        v2_post=v2_post,
        dv1=dv1,
        dv2=dv2,
        dv1_mag=float(np.linalg.norm(dv1)),
        dv2_mag=float(np.linalg.norm(dv2)),
        impulse_mag=float(abs(j_scalar)),
        relative_speed=rel_speed,
        separation_distance=dist,
        restitution=e,
        momentum_error=p_err,
    )


def generate_collision_debris_fragments(
    sat_name: str,
    parent_mass: float,
    parent_drag_area: float,
    parent_cd: float,
    col_pos: np.ndarray | list[float],
    post_vel: np.ndarray | list[float],
    relative_speed: float,
    num_fragments: int = 2,
    core_mass_fraction: float = 0.50,
    parent_body: str = "Earth",
    propagation_mode: str = "rk45",
) -> tuple[dict, list[dict]]:
    """Partition colliding satellite into a reduced surviving core and daughter fragments.
    
    Zero-sum momentum dispersion is strictly enforced across daughter fragments:
        sum(m_frag * delta_v_frag) == 0
    
    Returns:
        core_spec: Updated specifications for surviving core (reduced mass, etc.)
        daughters: List of independent daughter fragment specifications with initial states.
    """
    pos = np.asarray(col_pos, dtype=np.float64)
    v_base = np.asarray(post_vel, dtype=np.float64)

    # Core retains fraction of mass
    core_mass = max(1.0, parent_mass * core_mass_fraction)
    debris_mass_total = max(0.5, parent_mass * (1.0 - core_mass_fraction))
    m_frag = debris_mass_total / max(1, num_fragments)

    core_spec = {
        "mass": float(core_mass),
        "drag_area": float(parent_drag_area * ((core_mass / parent_mass) ** (2.0 / 3.0))),
    }

    daughters = []
    if num_fragments <= 0:
        return core_spec, daughters

    # Characteristic dispersion speed: 2% to 6% of relative impact velocity (min 20 m/s, max 400 m/s)
    v_disp_base = float(np.clip(relative_speed * 0.04, 20.0, 400.0))

    candidate_dvs = []
    for _ in range(num_fragments):
        u = float(np.random.uniform(-1.0, 1.0))
        theta = float(np.random.uniform(0.0, 2.0 * math.pi))
        r_xy = math.sqrt(max(0.0, 1.0 - u * u))
        dir_vec = np.array([r_xy * math.cos(theta), r_xy * math.sin(theta), u], dtype=np.float64)
        speed = v_disp_base * float(np.random.uniform(0.85, 1.15))
        candidate_dvs.append(dir_vec * speed)

    candidate_dvs = np.array(candidate_dvs)
    # Subtract center-of-mass dispersion to enforce exact zero-sum momentum
    cm_dv = np.mean(candidate_dvs, axis=0)
    exact_dvs = candidate_dvs - cm_dv

    frag_labels = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    for k in range(num_fragments):
        label = frag_labels[k % len(frag_labels)] if k < len(frag_labels) else f"F{k+1}"
        delta_v = exact_dvs[k]
        v_frag = v_base + delta_v

        # Scaled area with higher A/m ratio for debris
        frag_area = float(parent_drag_area * ((m_frag / parent_mass) ** (2.0 / 3.0)))

        daughters.append({
            "name": f"{sat_name}-{label}",
            "parent_satellite": sat_name,
            "parent_body": parent_body,
            "mass": float(m_frag),
            "drag_area": float(frag_area),
            "cd": float(parent_cd * 1.15),
            "propagation_mode": propagation_mode,
            "initial_state": pos.tolist() + v_frag.tolist(),
            "color": "#f97316" if k % 2 == 0 else "#eab308",
            "icon": "space_station",
            "is_daughter": True,
        })

    return core_spec, daughters

