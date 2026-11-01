"""High-precision numerical orbit propagator incorporating J2 oblateness and atmospheric drag."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from orbitguard.physics.constants import (
    J2_EARTH,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH,
    RHO_0,
    SCALE_HEIGHT_LEO,
)
from orbitguard.physics.elements import OrbitalElements, rv_to_coe


@dataclass
class TrajectoryOutput:
    """Propagated trajectory time series and final state."""

    t: np.ndarray       # Epochs array shape (N,) [s]
    r: np.ndarray       # ECI positions shape (N, 3) [m]
    v: np.ndarray       # ECI velocities shape (N, 3) [m/s]
    altitudes: np.ndarray # Geocentric altitudes [m]
    elements: list[OrbitalElements] # Keplerian elements along trajectory
    success: bool
    reentry: bool = False


class OrbitPropagator:
    """Numerical Cowell propagator for Low-Earth Orbit satellites and debris."""

    def __init__(
        self,
        mass_kg: float = 4.0,
        drag_area_m2: float = 0.03,
        cd: float = 2.2,
        include_j2: bool = True,
        include_drag: bool = True,
        mu: float = MU_EARTH,
    ):
        self.mass = mass_kg
        self.area = drag_area_m2
        self.cd = cd
        self.include_j2 = include_j2
        self.include_drag = include_drag
        self.mu = mu
        self.omega_vec = np.array([0.0, 0.0, OMEGA_EARTH], dtype=np.float64)

    def _calc_density(self, alt_m: float) -> float:
        """Exponential atmospheric density model."""
        if alt_m < 80000.0:
            # Below 80 km re-entry regime
            return 1e-4
        if alt_m > 1000000.0:
            # Above 1000 km near-vacuum
            return 0.0
        # Reference altitude 200 km
        h_ref = 200000.0
        rho_ref = 2.789e-10  # kg/m^3 at 200 km
        h_scale = 37500.0    # Mean thermospheric scale height [m]
        return rho_ref * math.exp(-(alt_m - h_ref) / h_scale)

    def compute_acceleration(self, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Total orbital acceleration = Keplerian + J2 Oblateness + Aerodynamic Drag."""
        r_norm = float(np.linalg.norm(r))
        if r_norm < 1e-3:
            return np.zeros(3)

        # 1. Central Two-Body Gravitational Acceleration
        a_grav = -self.mu * r / (r_norm**3)

        # 2. Earth J2 Zonal Oblateness Acceleration
        a_j2 = np.zeros(3, dtype=np.float64)
        if self.include_j2:
            z = r[2]
            z2_r2 = (z / r_norm) ** 2
            factor = (1.5 * J2_EARTH * self.mu * (R_EARTH**2)) / (r_norm**5)
            a_j2[0] = factor * r[0] * (5.0 * z2_r2 - 1.0)
            a_j2[1] = factor * r[1] * (5.0 * z2_r2 - 1.0)
            a_j2[2] = factor * r[2] * (5.0 * z2_r2 - 3.0)

        # 3. Atmospheric Drag Acceleration
        a_drag = np.zeros(3, dtype=np.float64)
        if self.include_drag and self.mass > 0.0:
            alt = r_norm - R_EARTH
            rho = self._calc_density(alt)
            if rho > 1e-16:
                # Relative velocity accounting for Earth's atmospheric co-rotation
                v_rel = v - np.cross(self.omega_vec, r)
                v_rel_norm = float(np.linalg.norm(v_rel))
                a_drag = -0.5 * rho * self.cd * (self.area / self.mass) * v_rel_norm * v_rel

        return a_grav + a_j2 + a_drag

    def propagate(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        duration_seconds: float,
        t_start: float = 0.0,
        step_seconds: float = 60.0,
    ) -> TrajectoryOutput:
        """Numerically propagate state over duration_seconds using adaptive Runge-Kutta 4(5)."""
        if duration_seconds <= 1e-6:
            alt = float(np.linalg.norm(r0)) - R_EARTH
            try:
                coe = rv_to_coe(r0, v0, mu=self.mu)
                elements = [coe]
            except Exception:
                elements = []
            return TrajectoryOutput(
                t=np.array([t_start]),
                r=np.array([r0]),
                v=np.array([v0]),
                altitudes=np.array([alt]),
                elements=elements,
                success=True,
                reentry=False,
            )

        y0 = np.concatenate([r0, v0])
        t_span = (t_start, t_start + duration_seconds)

        # Evenly spaced evaluation timestamps
        num_steps = max(2, int(duration_seconds / step_seconds) + 1)
        t_eval = np.linspace(t_start, t_start + duration_seconds, num_steps)

        def dynamics(t: float, y: np.ndarray) -> np.ndarray:
            r = y[0:3]
            v = y[3:6]
            a = self.compute_acceleration(t, r, v)
            return np.concatenate([v, a])

        # Terminal event: Re-entry below 100 km Karman line
        def reentry_event(t: float, y: np.ndarray) -> float:
            r = y[0:3]
            return float(np.linalg.norm(r) - (R_EARTH + 100000.0))

        reentry_event.terminal = True
        reentry_event.direction = -1

        sol = solve_ivp(
            dynamics,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            events=reentry_event,
            rtol=1e-7,
            atol=1e-7,
        )

        r_history = sol.y[0:3].T
        v_history = sol.y[3:6].T
        altitudes = np.linalg.norm(r_history, axis=1) - R_EARTH

        # Calculate orbital elements for each time step
        elements = []
        for r_step, v_step in zip(r_history, v_history):
            try:
                coe = rv_to_coe(r_step, v_step, mu=self.mu)
                elements.append(coe)
            except Exception:
                pass

        reentry_detected = len(sol.t_events[0]) > 0 if sol.t_events else False

        return TrajectoryOutput(
            t=sol.t,
            r=r_history,
            v=v_history,
            altitudes=altitudes,
            elements=elements,
            success=sol.success,
            reentry=reentry_detected,
        )
