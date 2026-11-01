"""72-hour conjunction screening engine and Time of Closest Approach (TCA) optimizer."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from orbitguard.physics.constants import MU_EARTH
from orbitguard.physics.elements import compute_vnc_basis
from orbitguard.physics.propagator import OrbitPropagator


@dataclass
class ConjunctionEvent:
    """Detailed kinematic and geometric properties of an orbital conjunction encounter."""

    tca_seconds: float              # Elapsed seconds from epoch to TCA
    tca_datetime: datetime.datetime # Estimated UTC timestamp of closest approach
    miss_distance_km: float         # Euclidean scalar miss distance [km]
    radial_miss_km: float           # Radial component in RTN frame [km]
    in_track_miss_km: float         # In-track component in RTN frame [km]
    cross_track_miss_km: float      # Cross-track component in RTN frame [km]
    relative_speed_km_s: float      # Magnitude of relative velocity [km/s]
    r1_tca: np.ndarray              # Primary satellite position at TCA [m]
    v1_tca: np.ndarray              # Primary satellite velocity at TCA [m/s]
    r2_tca: np.ndarray              # Secondary object position at TCA [m]
    v2_tca: np.ndarray              # Secondary object velocity at TCA [m/s]
    risk_tier: str                  # NEGLIGIBLE | LOW | MODERATE | HIGH | CRITICAL
    collision_probability: float = 0.0 # B-plane Foster/Akella Pc


class ConjunctionScreener:
    """Performs pairwise 72-hour trajectory screening and high-precision TCA localization."""

    def __init__(
        self,
        propagator1: OrbitPropagator,
        propagator2: OrbitPropagator,
        screening_threshold_km: float = 10.0,
    ):
        self.prop1 = propagator1
        self.prop2 = propagator2
        self.threshold_km = screening_threshold_km

    def screen_encounter(
        self,
        r1_0: np.ndarray,
        v1_0: np.ndarray,
        r2_0: np.ndarray,
        v2_0: np.ndarray,
        epoch_start: datetime.datetime,
        duration_hours: float = 72.0,
        coarse_step_s: float = 60.0,
    ) -> List[ConjunctionEvent]:
        """Screen two orbital trajectories over horizon and locate all close approach events."""
        duration_s = duration_hours * 3600.0

        # Step 1: Coarse propagation sweep
        traj1 = self.prop1.propagate(r1_0, v1_0, duration_seconds=duration_s, step_seconds=coarse_step_s)
        traj2 = self.prop2.propagate(r2_0, v2_0, duration_seconds=duration_s, step_seconds=coarse_step_s)

        # Min timestamps length matching
        n_pts = min(len(traj1.t), len(traj2.t))
        t_arr = traj1.t[:n_pts]
        r1_arr = traj1.r[:n_pts]
        r2_arr = traj2.r[:n_pts]

        diff_arr = r1_arr - r2_arr
        dist_km = np.linalg.norm(diff_arr, axis=1) / 1000.0

        # Step 2: Identify local minima indices below screening threshold
        candidates = []
        for i in range(1, n_pts - 1):
            if dist_km[i] < dist_km[i - 1] and dist_km[i] < dist_km[i + 1]:
                # Local minimum
                if dist_km[i] <= self.threshold_km * 2.5:  # Margin for refined search
                    candidates.append(i)

        # Step 3: Refine each local minimum using Brent's 1D optimization
        events: List[ConjunctionEvent] = []

        for idx in candidates:
            t_center = t_arr[idx]
            t_bracket = (max(0.0, t_center - coarse_step_s), min(duration_s, t_center + coarse_step_s))

            def distance_objective(t_sec: float) -> float:
                # Local propagation from r1_0, v1_0 and r2_0, v2_0 to t_sec
                sub1 = self.prop1.propagate(r1_0, v1_0, duration_seconds=t_sec, step_seconds=max(t_sec, 1.0))
                sub2 = self.prop2.propagate(r2_0, v2_0, duration_seconds=t_sec, step_seconds=max(t_sec, 1.0))
                r1_curr = sub1.r[-1]
                r2_curr = sub2.r[-1]
                return float(np.linalg.norm(r1_curr - r2_curr))

            res = minimize_scalar(
                distance_objective,
                bracket=t_bracket,
                method="brent",
                tol=1e-3,
            )

            tca_s = float(res.x)
            min_dist_m = float(res.fun)
            min_dist_km = min_dist_m / 1000.0

            if min_dist_km <= self.threshold_km:
                # Compute exact states at TCA
                final1 = self.prop1.propagate(r1_0, v1_0, duration_seconds=tca_s, step_seconds=max(tca_s, 1.0))
                final2 = self.prop2.propagate(r2_0, v2_0, duration_seconds=tca_s, step_seconds=max(tca_s, 1.0))

                r1_tca = final1.r[-1]
                v1_tca = final1.v[-1]
                r2_tca = final2.r[-1]
                v2_tca = final2.v[-1]

                delta_r = r1_tca - r2_tca
                delta_v = v1_tca - v2_tca
                rel_speed = float(np.linalg.norm(delta_v)) / 1000.0

                # Compute RTN (Radial, Transverse/In-track, Normal/Cross-track) projections
                r1_norm = np.linalg.norm(r1_tca)
                u_radial = r1_tca / r1_norm
                h1 = np.cross(r1_tca, v1_tca)
                u_cross = h1 / np.linalg.norm(h1)
                u_in_track = np.cross(u_cross, u_radial)

                radial_km = float(np.dot(delta_r, u_radial)) / 1000.0
                in_track_km = float(np.dot(delta_r, u_in_track)) / 1000.0
                cross_track_km = float(np.dot(delta_r, u_cross)) / 1000.0

                # Determine Risk Tier
                if min_dist_km < 1.0:
                    tier = "CRITICAL"
                elif min_dist_km < 3.0:
                    tier = "HIGH"
                elif min_dist_km < 7.0:
                    tier = "MODERATE"
                elif min_dist_km <= self.threshold_km:
                    tier = "LOW"
                else:
                    tier = "NEGLIGIBLE"

                tca_utc = epoch_start + datetime.timedelta(seconds=tca_s)

                events.append(ConjunctionEvent(
                    tca_seconds=tca_s,
                    tca_datetime=tca_utc,
                    miss_distance_km=min_dist_km,
                    radial_miss_km=radial_km,
                    in_track_miss_km=in_track_km,
                    cross_track_miss_km=cross_track_km,
                    relative_speed_km_s=rel_speed,
                    r1_tca=r1_tca,
                    v1_tca=v1_tca,
                    r2_tca=r2_tca,
                    v2_tca=v2_tca,
                    risk_tier=tier,
                ))

        return events
