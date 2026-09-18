"""Orbital equations of motion and state derivative functions.

Computes d/dt [r, v] = [v, a_total] combining central gravity, J2 perturbation,
and atmospheric drag.
"""

import numpy as np

from leo_simulator.constants import J2_EARTH, MU_EARTH, MU_MOON, OMEGA_EARTH, R_EARTH
from leo_simulator.models.drag import (
    BaseAtmosphere,
    ExponentialAtmosphere,
    aerodynamic_drag_acceleration,
)
from leo_simulator.models.gravity import (
    central_gravity_acceleration,
    j2_perturbation_acceleration,
    moon_position,
    third_body_acceleration,
)


class OrbitalDynamics:
    """Configurable dynamical model providing orbital state derivatives for numerical integration.

    State vector: y = [x, y, z, vx, vy, vz] in ECI meters and m/s.
    Derivative: dy/dt = [vx, vy, vz, ax, ay, az].
    """

    def __init__(
        self,
        cd: float = 2.2,
        area: float = 0.03,  # 3U CubeSat typical frontal area ~ 0.03 m^2 (0.1m x 0.3m)
        mass: float = 4.0,   # 3U CubeSat typical mass ~ 4 kg
        atmosphere: BaseAtmosphere | None = None,
        include_central_gravity: bool = True,
        include_j2: bool = True,
        include_drag: bool = True,
        include_earth_rotation: bool = True,
        include_moon: bool = False,
        mu: float = MU_EARTH,
        r_earth: float = R_EARTH,
        j2: float = J2_EARTH,
        omega_earth: float = OMEGA_EARTH,
        mu_moon: float = MU_MOON,
    ) -> None:
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"Mass must be strictly positive and finite, got {mass}")
        if not np.isfinite(area) or area < 0.0:
            raise ValueError(f"Drag area cannot be negative and must be finite, got {area}")
        if not np.isfinite(cd) or cd < 0.0:
            raise ValueError(f"Drag coefficient cannot be negative and must be finite, got {cd}")
        if not np.isfinite(mu) or mu <= 0.0:
            raise ValueError(f"Gravitational parameter mu must be strictly positive and finite, got {mu}")
        if not np.isfinite(r_earth) or r_earth <= 0.0:
            raise ValueError(f"Earth radius must be strictly positive and finite, got {r_earth}")
        if not np.isfinite(j2):
            raise ValueError(f"J2 harmonic coefficient must be finite, got {j2}")
        if not np.isfinite(omega_earth):
            raise ValueError(f"Earth rotation rate must be finite, got {omega_earth}")
        if not np.isfinite(mu_moon) or mu_moon <= 0.0:
            raise ValueError(f"Moon gravitational parameter mu_moon must be strictly positive and finite, got {mu_moon}")

        self.cd = float(cd)
        self.area = float(area)
        self.mass = float(mass)
        self.atmosphere = atmosphere if atmosphere is not None else ExponentialAtmosphere()
        self.include_central_gravity = include_central_gravity
        self.include_j2 = include_j2
        self.include_drag = include_drag
        self.include_earth_rotation = include_earth_rotation
        self.include_moon = include_moon
        self.mu = float(mu)
        self.r_earth = float(r_earth)
        self.j2 = float(j2)
        self.omega_earth = float(omega_earth)
        self.mu_moon = float(mu_moon)

    def compute_accelerations(
        self,
        r_vec: np.ndarray,
        v_vec: np.ndarray,
        t: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Calculate individual acceleration contributions in m / s^2.

        Args:
            r_vec: Satellite position vector in ECI frame [x, y, z] in meters.
            v_vec: Satellite velocity vector in ECI frame [vx, vy, vz] in m/s.
            t: Current simulation time in seconds (sets Moon ephemeris phase).

        Returns:
            dict containing 'central', 'j2', 'drag', 'moon', and 'total'
            acceleration vectors.
        """
        a_central = (
            central_gravity_acceleration(r_vec, mu=self.mu)
            if self.include_central_gravity
            else np.zeros(3, dtype=np.float64)
        )

        a_j2 = (
            j2_perturbation_acceleration(r_vec, mu=self.mu, r_earth=self.r_earth, j2=self.j2)
            if self.include_j2
            else np.zeros(3, dtype=np.float64)
        )

        a_drag = (
            aerodynamic_drag_acceleration(
                r_vec,
                v_vec,
                cd=self.cd,
                area=self.area,
                mass=self.mass,
                atmosphere_model=self.atmosphere,
                r_earth=self.r_earth,
                omega_earth=self.omega_earth,
                include_earth_rotation=self.include_earth_rotation,
            )
            if self.include_drag
            else np.zeros(3, dtype=np.float64)
        )

        a_moon = (
            third_body_acceleration(r_vec, moon_position(t), mu_moon=self.mu_moon)
            if self.include_moon
            else np.zeros(3, dtype=np.float64)
        )

        a_total = a_central + a_j2 + a_drag + a_moon

        return {
            "central": a_central,
            "j2": a_j2,
            "drag": a_drag,
            "moon": a_moon,
            "total": a_total,
        }

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """Evaluate state derivatives [vx, vy, vz, ax, ay, az].

        Args:
            t: Current simulation time in seconds (sets Moon ephemeris phase
                when lunar perturbation is enabled).
            state: Array-like of length 6 [x, y, z, vx, vy, vz].

        Returns:
            np.ndarray: Length 6 derivative array.
        """
        r_vec = state[0:3]
        v_vec = state[3:6]

        accels = self.compute_accelerations(r_vec, v_vec, t=t)
        a_total = accels["total"]

        dstate = np.empty(6, dtype=np.float64)
        dstate[0:3] = v_vec
        dstate[3:6] = a_total
        return dstate

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        return self.derivatives(t, state)
