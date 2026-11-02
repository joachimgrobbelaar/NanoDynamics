"""OrbitGuard AI: Orbital Conjunction Screening, B-Plane Pc Risk Scoring, and CDM Generation."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize_scalar

from core.dynamics import (
    MU_EARTH,
    OrbitalElements,
    compute_vnc_basis,
    rv_to_coe,
)
from core.propagator import NumericalCowellPropagator, OrbitPropagator


@dataclass
class ConjunctionEvent:
    """Detailed kinematic, geometric, and statistical properties of an orbital conjunction encounter."""

    tca_seconds: float                  # Seconds from epoch to TCA
    tca_datetime: datetime.datetime     # UTC timestamp of closest approach
    miss_distance_km: float             # Euclidean scalar miss distance [km]
    radial_miss_km: float               # Radial component in RTN frame [km]
    in_track_miss_km: float             # In-track component in RTN frame [km]
    cross_track_miss_km: float          # Cross-track component in RTN frame [km]
    relative_speed_km_s: float          # Relative velocity magnitude [km/s]
    r1_tca: np.ndarray                  # Primary object position at TCA [m]
    v1_tca: np.ndarray                  # Primary object velocity at TCA [m/s]
    r2_tca: np.ndarray                  # Secondary object position at TCA [m]
    v2_tca: np.ndarray                  # Secondary object velocity at TCA [m/s]
    collision_probability: float        # 2D B-plane Foster/Akella-Alfriend Pc
    risk_tier: str                      # NEGLIGIBLE | LOW | MODERATE | HIGH | CRITICAL
    primary_id: str = "PRIMARY_SAT"
    secondary_id: str = "DEBRIS_OR_SECONDARY"

    def to_dict(self) -> dict:
        return {
            "primary_id": self.primary_id,
            "secondary_id": self.secondary_id,
            "tca_seconds": float(self.tca_seconds),
            "tca_datetime": self.tca_datetime.isoformat(),
            "miss_distance_km": float(self.miss_distance_km),
            "radial_miss_km": float(self.radial_miss_km),
            "in_track_miss_km": float(self.in_track_miss_km),
            "cross_track_miss_km": float(self.cross_track_miss_km),
            "relative_speed_km_s": float(self.relative_speed_km_s),
            "collision_probability": float(self.collision_probability),
            "risk_tier": self.risk_tier,
            "r1_tca": self.r1_tca.tolist(),
            "v1_tca": self.v1_tca.tolist(),
            "r2_tca": self.r2_tca.tolist(),
            "v2_tca": self.v2_tca.tolist(),
        }

    def generate_cdm(self) -> str:
        """Generate CCSDS-compliant Conjunction Data Message (CDM) standard text."""
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        tca_str = self.tca_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        return f"""CCSDS_CDM_VERS = 1.0
COMMENT OrbitGuard AI Automated Collision Conjunction Report
CREATION_DATE = {now_str}
ORIGINATOR = NANODYNAMICS-ORBITGUARD-AI
MESSAGE_ID = CDM-{int(self.tca_seconds)}-{self.primary_id}-{self.secondary_id}

TCA = {tca_str}
MISS_DISTANCE = {self.miss_distance_km:.4f} [km]
RELATIVE_SPEED = {self.relative_speed_km_s:.4f} [km/s]
COLLISION_PROBABILITY = {self.collision_probability:.4e}
COLLISION_RISK_TIER = {self.risk_tier}

OBJECT_1 = {self.primary_id}
OBJECT_1_DESIGNATOR = SAT-01
OBJECT_2 = {self.secondary_id}
OBJECT_2_DESIGNATOR = DEB-TARGET

RTN_RADIAL_MISS = {self.radial_miss_km:.4f} [km]
RTN_INTRACK_MISS = {self.in_track_miss_km:.4f} [km]
RTN_CROSSTRACK_MISS = {self.cross_track_miss_km:.4f} [km]
"""


def compute_collision_plane_basis(v_rel: np.ndarray, r1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute orthonormal encounter frame (x_enc, y_enc, z_enc).

    - y_enc: aligned with relative velocity vector v_rel (encounter direction).
    - x_enc: perpendicular to relative velocity in collision plane.
    - z_enc: completes right-handed triad in collision plane.
    """
    v_norm = float(np.linalg.norm(v_rel))
    if v_norm < 1e-6:
        raise ValueError("Relative velocity is near-zero; encounter plane undefined.")
    y_enc = v_rel / v_norm

    r1_norm = float(np.linalg.norm(r1))
    r_unit = r1 / max(r1_norm, 1e-12)
    x_cross = np.cross(y_enc, r_unit)
    x_norm = float(np.linalg.norm(x_cross))

    if x_norm < 1e-4:
        aux = np.array([1.0, 0.0, 0.0]) if abs(y_enc[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        x_cross = np.cross(y_enc, aux)
        x_norm = float(np.linalg.norm(x_cross))

    x_enc = x_cross / max(x_norm, 1e-12)
    z_enc = np.cross(x_enc, y_enc)
    return x_enc, y_enc, z_enc


def calculate_collision_probability(
    miss_distance_m: float,
    combined_radius_m: float = 2.0,
    sigma_major_m: float = 150.0,
    sigma_minor_m: float = 75.0,
) -> float:
    """Calculate 2D B-plane collision probability (Pc) over combined hard-body sphere.

    Uses the Akella-Alfriend / Foster model integrating bivariate normal distribution.
    For small hard-body radius R << sigma, the uniform disk approximation applies:
      Pc ~= (pi * R^2) / (2 * pi * sigma_x * sigma_z) * exp(-0.5 * (x_m / sigma_x)^2)
    """
    if miss_distance_m < 0.0:
        return 0.0

    r = combined_radius_m
    sx = max(1.0, sigma_major_m)
    sz = max(1.0, sigma_minor_m)
    xm = miss_distance_m

    # If miss distance is extreme compared to uncertainty (> 8 sigma), Pc is zero
    if xm > 8.0 * max(sx, sz):
        return 0.0

    area = math.pi * (r**2)
    norm_const = 1.0 / (2.0 * math.pi * sx * sz)
    exponent = -0.5 * ((xm / sx) ** 2)
    pc_approx = area * norm_const * math.exp(exponent)

    return min(1.0, max(0.0, float(pc_approx)))


def evaluate_encounter_risk(
    miss_distance_km: float,
    combined_radius_m: float = 2.0,
    sigma_radial_m: float = 50.0,
    sigma_in_track_m: float = 250.0,
    sigma_cross_track_m: float = 80.0,
) -> Tuple[float, str]:
    """Evaluate collision probability and assign operational risk tier.

    Risk Tiers:
    - CRITICAL: Pc >= 1e-3 or miss_distance < 0.1 km (100 m)
    - HIGH:     1e-4 <= Pc < 1e-3 or miss_distance < 0.5 km
    - MODERATE: 1e-5 <= Pc < 1e-4 or miss_distance < 1.0 km
    - LOW:      1e-7 <= Pc < 1e-5 or miss_distance < 5.0 km
    - NEGLIGIBLE: Pc < 1e-7 and miss_distance >= 5.0 km
    """
    miss_distance_m = miss_distance_km * 1000.0
    sigma_major = math.sqrt(sigma_in_track_m**2 + sigma_cross_track_m**2)
    sigma_minor = sigma_radial_m

    pc = calculate_collision_probability(
        miss_distance_m=miss_distance_m,
        combined_radius_m=combined_radius_m,
        sigma_major_m=sigma_major,
        sigma_minor_m=sigma_minor,
    )

    if pc >= 1e-3 or miss_distance_km < 0.1:
        tier = "CRITICAL"
    elif pc >= 1e-4 or miss_distance_km < 0.5:
        tier = "HIGH"
    elif pc >= 1e-5 or miss_distance_km < 1.0:
        tier = "MODERATE"
    elif pc >= 1e-7 or miss_distance_km < 5.0:
        tier = "LOW"
    else:
        tier = "NEGLIGIBLE"

    return pc, tier


class OrbitGuardScreener:
    """Performs pairwise 72-hour trajectory conjunction screening and high-precision TCA localization."""

    def __init__(
        self,
        screening_threshold_km: float = 15.0,
        combined_radius_m: float = 2.0,
    ):
        self.threshold_km = screening_threshold_km
        self.combined_radius = combined_radius_m
        self.prop1 = OrbitPropagator()
        self.prop2 = OrbitPropagator()

    def screen_encounter(
        self,
        r1_0: np.ndarray,
        v1_0: np.ndarray,
        r2_0: np.ndarray,
        v2_0: np.ndarray,
        epoch_start: Optional[datetime.datetime] = None,
        duration_hours: float = 72.0,
        coarse_step_s: float = 60.0,
        primary_id: str = "PRIMARY_SAT",
        secondary_id: str = "DEBRIS_OR_SECONDARY",
    ) -> List[ConjunctionEvent]:
        """Screen two orbital trajectories over horizon and locate all close approach events."""
        if epoch_start is None:
            epoch_start = datetime.datetime.now(datetime.timezone.utc)

        duration_s = duration_hours * 3600.0

        # Coarse propagation sweep
        traj1 = self.prop1.propagate(r1_0, v1_0, duration_seconds=duration_s, step_seconds=coarse_step_s)
        traj2 = self.prop2.propagate(r2_0, v2_0, duration_seconds=duration_s, step_seconds=coarse_step_s)

        n_pts = min(len(traj1.t), len(traj2.t))
        if n_pts < 3:
            return []

        t_arr = traj1.t[:n_pts]
        r1_arr = traj1.r[:n_pts]
        r2_arr = traj2.r[:n_pts]

        diff_arr = r1_arr - r2_arr
        dist_km = np.linalg.norm(diff_arr, axis=1) / 1000.0

        events: List[ConjunctionEvent] = []

        # Find candidate minima (interior or boundary within threshold)
        candidate_indices = []
        if dist_km[0] <= self.threshold_km:
            candidate_indices.append(0)

        for i in range(1, n_pts - 1):
            if dist_km[i] <= dist_km[i - 1] and dist_km[i] <= dist_km[i + 1]:
                if dist_km[i] <= self.threshold_km * 2.0:
                    candidate_indices.append(i)

        if dist_km[-1] <= self.threshold_km and (n_pts - 1) not in candidate_indices:
            candidate_indices.append(n_pts - 1)

        for i in candidate_indices:
            t_min_bound = max(0.0, t_arr[max(0, i - 1)])
            t_max_bound = min(duration_s, t_arr[min(n_pts - 1, i + 1)])
            if t_max_bound <= t_min_bound:
                t_max_bound = t_min_bound + coarse_step_s

            # Fine numerical optimization of scalar distance(t)
            def dist_func(t_val: float) -> float:
                sub1 = self.prop1.propagate(r1_0, v1_0, duration_seconds=t_val, step_seconds=t_val)
                sub2 = self.prop2.propagate(r2_0, v2_0, duration_seconds=t_val, step_seconds=t_val)
                r1_f = sub1.r[-1]
                r2_f = sub2.r[-1]
                return float(np.linalg.norm(r1_f - r2_f))

            res = minimize_scalar(dist_func, bounds=(t_min_bound, t_max_bound), method="bounded")
            t_tca = float(res.x)

            fine1 = self.prop1.propagate(r1_0, v1_0, duration_seconds=t_tca, step_seconds=t_tca)
            fine2 = self.prop2.propagate(r2_0, v2_0, duration_seconds=t_tca, step_seconds=t_tca)

            r1_tca = fine1.r[-1]
            v1_tca = fine1.v[-1]
            r2_tca = fine2.r[-1]
            v2_tca = fine2.v[-1]

            r_rel = r2_tca - r1_tca
            v_rel = v2_tca - v1_tca
            miss_km = float(np.linalg.norm(r_rel) / 1000.0)
            rel_speed = float(np.linalg.norm(v_rel) / 1000.0)

            if miss_km <= self.threshold_km:
                v_hat, n_hat, c_hat = compute_vnc_basis(r1_tca, v1_tca)
                radial_miss = float(np.dot(r_rel, c_hat) / 1000.0)
                in_track_miss = float(np.dot(r_rel, v_hat) / 1000.0)
                cross_track_miss = float(np.dot(r_rel, n_hat) / 1000.0)

                pc, tier = evaluate_encounter_risk(miss_km, combined_radius_m=self.combined_radius)
                tca_dt = epoch_start + datetime.timedelta(seconds=t_tca)

                event = ConjunctionEvent(
                    tca_seconds=t_tca,
                    tca_datetime=tca_dt,
                    miss_distance_km=miss_km,
                    radial_miss_km=radial_miss,
                    in_track_miss_km=in_track_miss,
                    cross_track_miss_km=cross_track_miss,
                    relative_speed_km_s=rel_speed,
                    r1_tca=r1_tca,
                    v1_tca=v1_tca,
                    r2_tca=r2_tca,
                    v2_tca=v2_tca,
                    collision_probability=pc,
                    risk_tier=tier,
                    primary_id=primary_id,
                    secondary_id=secondary_id,
                )
                events.append(event)

        return sorted(events, key=lambda x: x.miss_distance_km)


def calculate_avoidance_maneuver(
    event: ConjunctionEvent,
    lead_time_seconds: float = 3600.0,
    target_clearance_km: float = 5.0,
) -> Dict[str, Union[float, str, list]]:
    """Calculate recommended collision avoidance delta-V maneuver (along velocity vector)."""
    # Required delta-position in track at TCA
    req_delta_pos_m = target_clearance_km * 1000.0
    # In linearized Clohessy-Wiltshire / Keplerian dynamics, in-track drift delta_y ~ -1.5 * n * t * delta_a
    # where delta_a ~ 2 * v * delta_v / n^2 => delta_y ~ -3 * (v / n) * delta_v * (t)
    r1_norm = float(np.linalg.norm(event.r1_tca))
    v1_norm = float(np.linalg.norm(event.v1_tca))
    mean_motion = math.sqrt(MU_EARTH / (r1_norm**3))

    delta_v_mag = req_delta_pos_m / (3.0 * (v1_norm / mean_motion) * lead_time_seconds)
    delta_v_mag = max(0.01, min(delta_v_mag, 15.0))  # Cap between 1 cm/s and 15 m/s

    v_hat, _, _ = compute_vnc_basis(event.r1_tca, event.v1_tca)
    delta_v_vec = delta_v_mag * v_hat

    return {
        "status": "MANEUVER_RECOMMENDED",
        "delta_v_magnitude_m_s": float(delta_v_mag),
        "delta_v_vector_eci": delta_v_vec.tolist(),
        "burn_direction": "PROGRADE" if delta_v_mag > 0 else "RETROGRADE",
        "lead_time_seconds": float(lead_time_seconds),
        "target_clearance_km": float(target_clearance_km),
        "projected_miss_km": float(event.miss_distance_km + target_clearance_km),
    }
