"""Canonical non-dimensionalization (R_scale, V_scale, tau_scale) and dataset generation.

Normalizes 6D SI orbital states `[r (m), v (m/s)]` into O(1) dimensionless states
`[r*, v*]` to prevent gradient vanishing/explosion in neural training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from src.astrodynamics.force_models import MU_EARTH, R_EARTH
from src.astrodynamics.propagators import NumericalPropagator


@dataclass(frozen=True)
class CanonicalScaler:
    r_scale_m: float = R_EARTH
    mu_m3_s2: float = MU_EARTH

    @property
    def v_scale_m_s(self) -> float:
        return math.sqrt(self.mu_m3_s2 / self.r_scale_m)

    @property
    def tau_scale_s(self) -> float:
        return math.sqrt((self.r_scale_m**3) / self.mu_m3_s2)

    @property
    def scale_vector_6d(self) -> np.ndarray:
        return np.array([self.r_scale_m] * 3 + [self.v_scale_m_s] * 3, dtype=np.float64)

    def normalize_state(self, state_si: np.ndarray) -> np.ndarray:
        return np.asarray(state_si, dtype=np.float64) / self.scale_vector_6d

    def denormalize_state(self, state_nd: np.ndarray) -> np.ndarray:
        return np.asarray(state_nd, dtype=np.float64) * self.scale_vector_6d


def generate_canonical_training_pairs(
    altitude_km: float = 500.0,
    inclination_deg: float = 53.0,
    b_coeff: float = 0.015,
    f107: float = 150.0,
    kp: float = 3.0,
    dt_s: float = 10.0,
    n_steps: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate canonical `(X_nd, Y_nd, P_cond)` transition pairs from high-precision DOP853."""
    scaler = CanonicalScaler()
    r0_mag = R_EARTH + altitude_km * 1000.0
    v0_mag = math.sqrt(MU_EARTH / r0_mag)
    inc_rad = math.radians(inclination_deg)

    state0 = np.array(
        [r0_mag, 0.0, 0.0, 0.0, v0_mag * math.cos(inc_rad), v0_mag * math.sin(inc_rad)],
        dtype=np.float64,
    )
    prop = NumericalPropagator(b_coeff=b_coeff, f107=f107, kp=kp, include_j2_j4=True, include_drag=True)
    traj = prop.propagate_gcrf(state0, duration_s=dt_s * n_steps, dt_s=dt_s)

    states_si = np.hstack([traj.r_m, traj.v_m_s])
    states_nd = scaler.normalize_state(states_si)

    x_nd = states_nd[:-1].astype(np.float32)
    y_nd = states_nd[1:].astype(np.float32)
    p_vec = np.array([b_coeff, f107 / 200.0, kp / 9.0, dt_s / scaler.tau_scale_s], dtype=np.float32)
    p_cond = np.tile(p_vec, (len(x_nd), 1))
    return x_nd, y_nd, p_cond
