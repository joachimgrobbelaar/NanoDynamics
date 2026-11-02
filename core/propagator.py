"""High-precision numerical perturbation engine (Cowell RK45/J2/Drag) and SGP4 analytical propagator."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import math
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

from core.dynamics import (
    J2_EARTH,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH,
    RHO_0,
    SCALE_HEIGHT_LEO,
    OrbitalElements,
    rv_to_coe,
)


@dataclass
class TrajectoryOutput:
    """Propagated trajectory time series and state vectors."""

    t: np.ndarray                   # Time relative to epoch [s] (N,)
    r: np.ndarray                   # ECI position vectors [m] (N, 3)
    v: np.ndarray                   # ECI velocity vectors [m/s] (N, 3)
    altitudes: np.ndarray           # Geocentric altitudes above Earth surface [m] (N,)
    elements: List[OrbitalElements] # Keplerian elements along trajectory
    success: bool
    reentry: bool = False

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "reentry": self.reentry,
            "points_count": len(self.t),
            "duration_seconds": float(self.t[-1] - self.t[0]) if len(self.t) > 0 else 0.0,
            "final_position": self.r[-1].tolist() if len(self.r) > 0 else [],
            "final_velocity": self.v[-1].tolist() if len(self.v) > 0 else [],
            "final_altitude_km": float(self.altitudes[-1] / 1000.0) if len(self.altitudes) > 0 else 0.0,
        }


class NumericalCowellPropagator:
    """High-precision Cowell numerical orbital propagator incorporating J2 and atmospheric drag."""

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

    def calc_density(self, alt_m: float) -> float:
        """Exponential atmospheric density model for LEO regime."""
        if alt_m < 80000.0:
            return 1e-4
        if alt_m > 1000000.0:
            return 0.0
        h_ref = 200000.0
        rho_ref = 2.789e-10  # kg/m^3 at 200 km
        h_scale = SCALE_HEIGHT_LEO
        return rho_ref * math.exp(-(alt_m - h_ref) / h_scale)

    def compute_acceleration(self, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Total acceleration = Keplerian + J2 Oblateness + Aerodynamic Drag."""
        r_norm = float(np.linalg.norm(r))
        if r_norm < 1e-3:
            return np.zeros(3)

        # 1. Two-Body Keplerian Acceleration
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

        # 3. Aerodynamic Drag Acceleration (incorporating atmospheric co-rotation)
        a_drag = np.zeros(3, dtype=np.float64)
        if self.include_drag and self.mass > 0.0:
            alt = r_norm - R_EARTH
            rho = self.calc_density(alt)
            if rho > 1e-16:
                v_rel = v - np.cross(self.omega_vec, r)
                v_rel_norm = float(np.linalg.norm(v_rel))
                a_drag = -0.5 * rho * self.cd * (self.area / self.mass) * v_rel_norm * v_rel

        return a_grav + a_j2 + a_drag

    def propagate(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        duration_seconds: float = 5400.0,
        step_seconds: float = 30.0,
        rtol: float = 1e-8,
        atol: float = 1e-9,
    ) -> TrajectoryOutput:
        """Propagate state vector over duration using adaptive Runge-Kutta RK45."""
        r0_arr = np.asarray(r0, dtype=np.float64)
        v0_arr = np.asarray(v0, dtype=np.float64)
        y0 = np.concatenate([r0_arr, v0_arr])

        t_eval = np.arange(0.0, duration_seconds + step_seconds, step_seconds)

        def dydt(t: float, y: np.ndarray) -> np.ndarray:
            r = y[0:3]
            v = y[3:6]
            a = self.compute_acceleration(t, r, v)
            return np.concatenate([v, a])

        # Terminal event for re-entry (altitude < 80 km)
        def reentry_event(t: float, y: np.ndarray) -> float:
            r_norm = float(np.linalg.norm(y[0:3]))
            return r_norm - (R_EARTH + 80000.0)

        reentry_event.terminal = True
        reentry_event.direction = -1

        sol = solve_ivp(
            fun=dydt,
            t_span=(0.0, duration_seconds),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=rtol,
            atol=atol,
            events=[reentry_event],
        )

        t_out = sol.t
        r_out = sol.y[0:3, :].T
        v_out = sol.y[3:6, :].T

        r_norms = np.linalg.norm(r_out, axis=1)
        altitudes = r_norms - R_EARTH
        is_reentry = sol.status == 1

        elements_list = []
        # Compute elements at sample intervals
        sample_step = max(1, len(t_out) // 20)
        for idx in range(0, len(t_out), sample_step):
            try:
                coe = rv_to_coe(r_out[idx], v_out[idx], self.mu)
                coe.epoch = float(t_out[idx])
                elements_list.append(coe)
            except Exception:
                pass

        return TrajectoryOutput(
            t=t_out,
            r=r_out,
            v=v_out,
            altitudes=altitudes,
            elements=elements_list,
            success=sol.success or is_reentry,
            reentry=is_reentry,
        )


class SGP4Propagator:
    """Analytical SGP4 orbit propagator for Two-Line Element (TLE) ephemeris generation."""

    def __init__(self, line1: str, line2: str):
        from sgp4.api import Satrec, WGS72
        self.satrec = Satrec.twoline2rv(line1.strip(), line2.strip(), WGS72)
        self.line1 = line1
        self.line2 = line2

    def propagate(
        self,
        start_datetime: datetime.datetime,
        duration_seconds: float = 5400.0,
        step_seconds: float = 30.0,
    ) -> TrajectoryOutput:
        """Propagate TLE from starting UTC datetime for duration_seconds."""
        from sgp4.api import jday

        t_steps = np.arange(0.0, duration_seconds + step_seconds, step_seconds)
        r_list = []
        v_list = []
        valid_t = []

        for delta_s in t_steps:
            current_dt = start_datetime + datetime.timedelta(seconds=float(delta_s))
            jd, fr = jday(
                current_dt.year,
                current_dt.month,
                current_dt.day,
                current_dt.hour,
                current_dt.minute,
                current_dt.second + current_dt.microsecond * 1e-6,
            )
            e, r, v = self.satrec.sgp4(jd, fr)
            if e == 0:
                # SGP4 outputs r in km and v in km/s, convert to meters and m/s
                r_m = np.array(r, dtype=np.float64) * 1000.0
                v_m_s = np.array(v, dtype=np.float64) * 1000.0
                r_list.append(r_m)
                v_list.append(v_m_s)
                valid_t.append(delta_s)

        if not r_list:
            return TrajectoryOutput(
                t=np.array([]),
                r=np.zeros((0, 3)),
                v=np.zeros((0, 3)),
                altitudes=np.array([]),
                elements=[],
                success=False,
            )

        r_arr = np.array(r_list)
        v_arr = np.array(v_list)
        t_arr = np.array(valid_t)
        altitudes = np.linalg.norm(r_arr, axis=1) - R_EARTH

        elements_list = []
        sample_step = max(1, len(t_arr) // 20)
        for idx in range(0, len(t_arr), sample_step):
            try:
                coe = rv_to_coe(r_arr[idx], v_arr[idx])
                coe.epoch = float(t_arr[idx])
                elements_list.append(coe)
            except Exception:
                pass

        return TrajectoryOutput(
            t=t_arr,
            r=r_arr,
            v=v_arr,
            altitudes=altitudes,
            elements=elements_list,
            success=True,
        )


# Exported standard alias
OrbitPropagator = NumericalCowellPropagator
