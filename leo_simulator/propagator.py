"""Numerical orbit propagator wrapping scipy.integrate.solve_ivp.

Provides high-precision trajectory propagation, orbital elements history tracking,
and terminal event handling for atmospheric re-entry.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from leo_simulator.constants import (
    J2_EARTH,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH,
)
from leo_simulator.models.drag import BaseAtmosphere, ExponentialAtmosphere
from leo_simulator.models.dynamics import OrbitalDynamics
from leo_simulator.orbit.elements import (
    OrbitalElements,
    coe_to_rv,
    rv_to_coe,
)
from leo_simulator.orbit.satellite import Satellite


@dataclass
class PropagationResult:
    """Structured container holding propagation output and computed orbital time series."""

    t: np.ndarray             # Timestamps [s]
    r: np.ndarray             # ECI Position array shape (N, 3) [m]
    v: np.ndarray             # ECI Velocity array shape (N, 3) [m/s]
    altitudes: np.ndarray     # Altitude array shape (N,) [m]
    speeds: np.ndarray        # Speed array shape (N,) [m/s]
    semi_major_axes: np.ndarray # Semi-major axis array shape (N,) [m]
    eccentricities: np.ndarray  # Eccentricity array shape (N,)
    inclinations: np.ndarray    # Inclination array shape (N,) [rad]
    raans: np.ndarray           # RAAN array shape (N,) [rad]
    arg_pes: np.ndarray         # Argument of periapsis array shape (N,) [rad]
    true_anomalies: np.ndarray  # True anomaly array shape (N,) [rad]
    success: bool
    status: int
    message: str
    reentry_detected: bool = False
    reentry_time: float | None = None

    @property
    def final_altitude(self) -> float:
        """Return final propagated altitude in meters."""
        return float(self.altitudes[-1])

    @property
    def initial_altitude(self) -> float:
        """Return initial altitude in meters."""
        return float(self.altitudes[0])

    @property
    def altitude_decay(self) -> float:
        """Compute net altitude change (initial - final) in meters."""
        return float(self.initial_altitude - self.final_altitude)

    @property
    def final_raan(self) -> float:
        """Return final RAAN in radians."""
        return float(self.raans[-1])

    @property
    def initial_raan(self) -> float:
        """Return initial RAAN in radians."""
        return float(self.raans[0])

    @property
    def raan_change(self) -> float:
        """Compute unwrap-adjusted net RAAN change (final - initial) in radians."""
        unwrapped = np.unwrap(self.raans)
        return float(unwrapped[-1] - unwrapped[0])


def create_reentry_event(min_altitude_m: float = 0.0, r_earth: float = R_EARTH) -> Callable:
    """Factory creating a terminal event function triggering on atmospheric re-entry/impact."""

    def reentry_event(t: float, y: np.ndarray) -> float:
        r_norm = np.linalg.norm(y[0:3])
        return r_norm - (r_earth + min_altitude_m)

    reentry_event.terminal = True
    reentry_event.direction = -1.0
    return reentry_event


class OrbitPropagator:
    """Numerical orbit propagator utilizing scipy.integrate.solve_ivp."""

    def __init__(
        self,
        satellite: Satellite | None = None,
        atmosphere: BaseAtmosphere | None = None,
        include_central_gravity: bool = True,
        include_j2: bool = True,
        include_drag: bool = True,
        include_earth_rotation: bool = True,
        mu: float = MU_EARTH,
        r_earth: float = R_EARTH,
        j2: float = J2_EARTH,
        omega_earth: float = OMEGA_EARTH,
        solver_method: str = "DOP853",
        rtol: float = 1e-10,
        atol: float = 1e-12,
        min_altitude_reentry: float = 0.0,
    ) -> None:
        self.satellite = satellite if satellite is not None else Satellite.cubesat_3u()
        self.atmosphere = atmosphere if atmosphere is not None else ExponentialAtmosphere()
        self.include_central_gravity = include_central_gravity
        self.include_j2 = include_j2
        self.include_drag = include_drag
        self.include_earth_rotation = include_earth_rotation
        self.mu = float(mu)
        self.r_earth = float(r_earth)
        self.j2 = float(j2)
        self.omega_earth = float(omega_earth)
        self.solver_method = solver_method
        self.rtol = rtol
        self.atol = atol
        self.min_altitude_reentry = min_altitude_reentry

        # Instantiate dynamics
        self.dynamics = OrbitalDynamics(
            cd=self.satellite.cd,
            area=self.satellite.drag_area,
            mass=self.satellite.mass,
            atmosphere=self.atmosphere,
            include_central_gravity=self.include_central_gravity,
            include_j2=self.include_j2,
            include_drag=self.include_drag,
            include_earth_rotation=self.include_earth_rotation,
            mu=self.mu,
            r_earth=self.r_earth,
            j2=self.j2,
            omega_earth=self.omega_earth,
        )

    def propagate(
        self,
        initial_state: np.ndarray | list | OrbitalElements,
        duration_seconds: float,
        t_start: float = 0.0,
        dt_eval: float | None = None,
        max_step: float = np.inf,
    ) -> PropagationResult:
        """Propagate orbit over a given duration.

        Args:
            initial_state: Either a 6-element Cartesian vector [x, y, z, vx, vy, vz] or OrbitalElements.
            duration_seconds: Total duration to integrate forward in seconds (> 0).
            t_start: Initial simulation epoch timestamp in seconds (default 0.0).
            dt_eval: Output time step spacing for dense evaluation in seconds.
            max_step: Maximum integrator internal step size.

        Returns:
            PropagationResult containing state time series and orbital elements.
        """
        if duration_seconds <= 0.0:
            raise ValueError(f"Duration must be strictly positive, got {duration_seconds}")

        if isinstance(initial_state, OrbitalElements):
            r0, v0 = coe_to_rv(initial_state, mu=self.mu)
            y0 = np.concatenate([r0, v0])
        else:
            y0 = np.asarray(initial_state, dtype=np.float64)
            if y0.shape != (6,):
                raise ValueError(f"Initial state vector must have shape (6,), got {y0.shape}")

        t_span = (float(t_start), float(t_start + duration_seconds))

        t_eval = None
        if dt_eval is not None and dt_eval > 0.0:
            num_steps = int(np.floor((t_span[1] - t_span[0]) / dt_eval))
            t_eval = t_span[0] + np.arange(num_steps + 1) * dt_eval
            if np.isclose(t_eval[-1], t_span[1], atol=1e-8 * dt_eval):
                t_eval[-1] = t_span[1]
            elif t_eval[-1] < t_span[1]:
                t_eval = np.append(t_eval, t_span[1])
            elif t_eval[-1] > t_span[1]:
                t_eval[-1] = t_span[1]

        reentry_event = create_reentry_event(
            min_altitude_m=self.min_altitude_reentry, r_earth=self.r_earth
        )

        sol = solve_ivp(
            fun=self.dynamics,
            t_span=t_span,
            y0=y0,
            method=self.solver_method,
            t_eval=t_eval,
            events=[reentry_event],
            rtol=self.rtol,
            atol=self.atol,
            max_step=max_step,
        )

        # Extract output arrays
        t_arr = sol.t
        y_arr = sol.y  # shape (6, N)
        r_arr = y_arr[0:3, :].T  # shape (N, 3)
        v_arr = y_arr[3:6, :].T  # shape (N, 3)

        r_norms = np.linalg.norm(r_arr, axis=1)
        altitudes = r_norms - self.r_earth
        speeds = np.linalg.norm(v_arr, axis=1)

        # Check for re-entry termination
        reentry_detected = False
        reentry_time = None
        if sol.t_events and len(sol.t_events[0]) > 0:
            reentry_detected = True
            reentry_time = float(sol.t_events[0][0])

        # Compute orbital elements along trajectory
        n_points = len(t_arr)
        a_arr = np.empty(n_points, dtype=np.float64)
        e_arr = np.empty(n_points, dtype=np.float64)
        i_arr = np.empty(n_points, dtype=np.float64)
        raan_arr = np.empty(n_points, dtype=np.float64)
        arg_pe_arr = np.empty(n_points, dtype=np.float64)
        nu_arr = np.empty(n_points, dtype=np.float64)

        for idx in range(n_points):
            coe = rv_to_coe(r_arr[idx], v_arr[idx], mu=self.mu)
            a_arr[idx] = coe.a
            e_arr[idx] = coe.e
            i_arr[idx] = coe.i
            raan_arr[idx] = coe.raan
            arg_pe_arr[idx] = coe.arg_pe
            nu_arr[idx] = coe.nu

        return PropagationResult(
            t=t_arr,
            r=r_arr,
            v=v_arr,
            altitudes=altitudes,
            speeds=speeds,
            semi_major_axes=a_arr,
            eccentricities=e_arr,
            inclinations=i_arr,
            raans=raan_arr,
            arg_pes=arg_pe_arr,
            true_anomalies=nu_arr,
            success=bool(sol.success),
            status=int(sol.status),
            message=str(sol.message),
            reentry_detected=reentry_detected,
            reentry_time=reentry_time,
        )
