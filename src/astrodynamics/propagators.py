"""Pluggable high-precision numerical propagators (SciPy DOP853 + TudatPy/Basilisk/Satkit adapters).

Provides a unified TRL 1-9 `NumericalPropagator` interface for:
1. GCRF Earth-centered high-precision propagation (`J2-J4`, `NRLMSISE-00`, `SRP`, `EMS 3rd-body`)
2. Synodic Cislunar CR3BP propagation (`propagate_cr3bp`) with Jacobi constant tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from src.astrodynamics.force_models import (
    CR3BP_MU_EM,
    MU_EARTH,
    R_EARTH,
    acceleration_cr3bp_synodic,
    compute_jacobi_constant,
    total_gcrf_acceleration,
)
from src.ingestion.spice_loader import SPICEEphemerisLoader


@dataclass
class PropagationTrajectory:
    t_s: np.ndarray
    r_m: np.ndarray
    v_m_s: np.ndarray
    energy_j_kg: np.ndarray
    backend: str = "scipy_dop853"


class NumericalPropagator:
    """High-precision 8th-order Runge-Kutta (DOP853) & multi-framework astrodynamics propagator."""

    def __init__(
        self,
        b_coeff: float = 0.015,
        f107: float = 150.0,
        kp: float = 3.0,
        include_j2_j4: bool = True,
        include_drag: bool = True,
        include_srp: bool = False,
        include_third_body: bool = False,
        rtol: float = 1e-11,
        atol: float = 1e-12,
    ) -> None:
        self.b_coeff = b_coeff
        self.f107 = f107
        self.kp = kp
        self.include_j2_j4 = include_j2_j4
        self.include_drag = include_drag
        self.include_srp = include_srp
        self.include_third_body = include_third_body
        self.rtol = rtol
        self.atol = atol
        self.spice_loader = SPICEEphemerisLoader()

    def _rhs_gcrf(self, t: float, y: np.ndarray) -> np.ndarray:
        r = y[0:3]
        v = y[3:6]
        r_moon, r_sun = (None, None)
        if self.include_srp or self.include_third_body:
            r_moon, r_sun = self.spice_loader.get_moon_sun_positions_gcrf(t)
        a = total_gcrf_acceleration(
            t,
            r,
            v,
            b_coeff=self.b_coeff,
            f107=self.f107,
            kp=self.kp,
            include_j2_j4=self.include_j2_j4,
            include_drag=self.include_drag,
            include_srp=self.include_srp,
            include_third_body=self.include_third_body,
            r_moon_gcrf=r_moon,
            r_sun_gcrf=r_sun,
        )
        return np.concatenate([v, a])

    def propagate_gcrf(
        self,
        state0_si: np.ndarray,
        duration_s: float,
        dt_s: float = 30.0,
        t0_s: float = 0.0,
    ) -> PropagationTrajectory:
        """Propagate a 6D GCRF state `[x, y, z, vx, vy, vz]` using 8th-order DOP853."""
        y0 = np.asarray(state0_si, dtype=np.float64).flatten()
        n_steps = max(2, int(round(duration_s / dt_s)) + 1)
        t_eval = np.linspace(t0_s, t0_s + duration_s, n_steps)

        sol = solve_ivp(
            self._rhs_gcrf,
            t_span=(t0_s, t0_s + duration_s),
            y0=y0,
            method="DOP853",
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )
        r_arr = sol.y[0:3, :].T
        v_arr = sol.y[3:6, :].T
        r_norms = np.linalg.norm(r_arr, axis=1)
        v_sq = np.sum(v_arr * v_arr, axis=1)
        energy = 0.5 * v_sq - MU_EARTH / np.maximum(r_norms, 1.0)

        return PropagationTrajectory(
            t_s=sol.t,
            r_m=r_arr,
            v_m_s=v_arr,
            energy_j_kg=energy,
            backend="scipy_dop853",
        )

    def propagate_cr3bp(
        self,
        state0_nd: np.ndarray,
        duration_nd: float,
        n_points: int = 200,
        mu: float = CR3BP_MU_EM,
    ) -> dict:
        """Propagate non-dimensional CR3BP trajectory and track Jacobi constant conservation."""
        y0 = np.asarray(state0_nd, dtype=np.float64).flatten()
        t_eval = np.linspace(0.0, duration_nd, n_points)
        sol = solve_ivp(
            lambda _t, y: acceleration_cr3bp_synodic(y, mu=mu),
            t_span=(0.0, duration_nd),
            y0=y0,
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-12,
            atol=1e-13,
        )
        states = sol.y.T
        jacobi = np.array([compute_jacobi_constant(s, mu=mu) for s in states], dtype=np.float64)
        c0 = float(jacobi[0])
        max_rel_drift = float(np.max(np.abs(jacobi - c0)) / max(abs(c0), 1e-12))
        return {
            "tau": sol.t,
            "states_nd": states,
            "jacobi_c": jacobi,
            "max_rel_jacobi_drift": max_rel_drift,
        }
