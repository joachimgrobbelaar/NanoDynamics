"""Orbital equations of motion and state derivative functions.

Computes d/dt [r, v] = [v, a_total] combining central gravity, J2 perturbation,
and atmospheric drag.
"""

import numpy as np

from leo_simulator.constants import J2_EARTH, MU_EARTH, MU_MOON, MU_SUN, OMEGA_EARTH, R_EARTH, R_MOON, R_SUN_ORBIT
from leo_simulator.models.drag import (
    BaseAtmosphere,
    ExponentialAtmosphere,
    aerodynamic_drag_acceleration,
)
from leo_simulator.models.gravity import (
    central_gravity_acceleration,
    j2_perturbation_acceleration,
    lunar_harmonics_acceleration,
    moon_position,
    sun_position,
    third_body_acceleration,
)

P_SUN_1AU: float = 4.56e-6  # Solar radiation pressure at 1 AU [N / m^2]


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
        include_sun: bool = False,
        include_lunar_harmonics: bool = False,
        include_srp: bool = False,
        cr: float = 1.3,
        mu: float = MU_EARTH,
        r_earth: float = R_EARTH,
        j2: float = J2_EARTH,
        omega_earth: float = OMEGA_EARTH,
        mu_moon: float = MU_MOON,
        mu_sun: float = MU_SUN,
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
        self.include_sun = include_sun
        self.include_lunar_harmonics = include_lunar_harmonics
        self.include_srp = include_srp
        self.cr = float(cr)
        self.mu = float(mu)
        self.r_earth = float(r_earth)
        self.j2 = float(j2)
        self.omega_earth = float(omega_earth)
        self.mu_moon = float(mu_moon)
        self.mu_sun = float(mu_sun)

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
            dict containing 'central', 'j2', 'drag', 'moon', 'sun', 'lunar_harmonics', and 'total'
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

        r_m = moon_position(t) if (self.include_moon or self.include_lunar_harmonics) else None
        a_moon = (
            third_body_acceleration(r_vec, r_m, mu_moon=self.mu_moon)
            if self.include_moon and r_m is not None
            else np.zeros(3, dtype=np.float64)
        )

        r_s = sun_position(t)
        a_sun = (
            third_body_acceleration(r_vec, r_s, mu_moon=self.mu_sun)
            if self.include_sun
            else np.zeros(3, dtype=np.float64)
        )

        # ── Solar Radiation Pressure (SRP) with Ray-Tracing Cylindrical Shadow ──
        if self.include_srp and self.mass > 0:
            d_sun_vec = r_s - r_vec
            dist_sun = float(np.linalg.norm(d_sun_vec))
            u_sun = d_sun_vec / dist_sun

            # Ray-sphere intersection against Earth
            t_closest = -float(np.dot(r_vec, u_sun))
            r_sat_mag_sq = float(np.dot(r_vec, r_vec))
            d_perp_sq = r_sat_mag_sq - (t_closest * t_closest) if t_closest > 0 else r_sat_mag_sq

            if t_closest > 0 and d_perp_sq < (self.r_earth * self.r_earth):
                nu = 0.0  # Full Umbra (Earth shadow)
            elif t_closest > 0 and d_perp_sq < ((self.r_earth + 100_000.0) ** 2):
                nu = (np.sqrt(max(0.0, d_perp_sq)) - self.r_earth) / 100_000.0  # Penumbra
            else:
                nu = 1.0  # Direct solar flux

            flux_scale = (R_SUN_ORBIT / dist_sun) ** 2
            p_srp_mag = P_SUN_1AU * flux_scale * (self.cr * self.area / self.mass) * nu
            # Radiation pressure pushes directly away from the Sun (-u_sun)
            a_srp = -p_srp_mag * u_sun
        else:
            a_srp = np.zeros(3, dtype=np.float64)

        if self.include_lunar_harmonics:
            r_sel = (r_vec - r_m) if (self.include_moon and r_m is not None) else np.asarray(r_vec, dtype=np.float64)
            a_lunar_harm = lunar_harmonics_acceleration(r_sel, mu_moon=self.mu_moon, r_moon=R_MOON)
        else:
            a_lunar_harm = np.zeros(3, dtype=np.float64)

        a_total = a_central + a_j2 + a_drag + a_moon + a_sun + a_srp + a_lunar_harm

        return {
            "central": a_central,
            "j2": a_j2,
            "drag": a_drag,
            "moon": a_moon,
            "sun": a_sun,
            "srp": a_srp,
            "lunar_harmonics": a_lunar_harm,
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
