"""Orbital collision mechanics and conjunction kinematics.

Calculates relative separation, closing range rates, 3D inelastic/elastic momentum
exchange impulses, and post-collision velocity deflections for orbiting bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
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
