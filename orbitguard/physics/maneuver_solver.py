"""Autonomous collision avoidance maneuver solver and propulsion budget optimizer."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import math
from typing import List, Optional, Tuple

import numpy as np

from orbitguard.physics.constants import G0, MU_EARTH, R_EARTH
from orbitguard.physics.elements import compute_vnc_basis, rv_to_coe
from orbitguard.physics.propagator import OrbitPropagator
from orbitguard.physics.screening import ConjunctionEvent


@dataclass
class ManeuverCandidate:
    """Calculated collision avoidance maneuver option with propulsion metrics."""

    strategy_name: str              # e.g., 'retrograde_phasing', 'prograde_phasing', 'normal_out_of_plane'
    burn_epoch_utc: datetime.datetime
    lead_time_hours: float          # Time elapsed between burn and TCA
    dv_vnc: Tuple[float, float, float] # (V, N, C) delta-V components [m/s]
    dv_magnitude_m_s: float         # Scalar total delta-V [m/s]
    burn_duration_s: float          # Required continuous thrust duration [s]
    propellant_consumed_kg: float   # Mass of fuel depleted [kg]
    remaining_fuel_kg: float        # Remaining onboard propellant [kg]
    remaining_delta_v_m_s: float    # Post-maneuver remaining delta-V budget [m/s]
    post_miss_distance_km: float    # Re-propagated miss distance at TCA [km]
    post_collision_probability: float # Re-evaluated Pc
    new_perigee_km: float           # Post-maneuver perigee altitude [km]
    new_apogee_km: float            # Post-maneuver apogee altitude [km]
    safety_clearance_verified: bool # True if miss distance exceeds 10 km and perigee > 200 km
    executive_summary: str          # Concise recommendation text


class ManeuverSolver:
    """Synthesizes optimal collision avoidance maneuvers for CubeSats and small satellites."""

    def __init__(
        self,
        propagator_primary: OrbitPropagator,
        propagator_secondary: OrbitPropagator,
        dry_mass_kg: float = 4.0,
        fuel_mass_kg: float = 0.5,
        thrust_n: float = 0.050,  # 50 mN cold gas default
        isp_s: float = 70.0,       # 70 s Isp
        target_clearance_km: float = 25.0,
    ):
        self.prop_pri = propagator_primary
        self.prop_sec = propagator_secondary
        self.dry_mass = dry_mass_kg
        self.fuel_mass = fuel_mass_kg
        self.thrust_n = thrust_n
        self.isp_s = isp_s
        self.target_clearance_km = target_clearance_km

    def compute_propellant_budget(self, dv_mag: float, current_fuel_kg: float) -> Tuple[float, float, float, float]:
        """Apply Tsiolkovsky rocket equation to compute fuel consumption and burn duration.

        Returns:
            (burn_duration_s, prop_consumed_kg, remaining_fuel_kg, remaining_total_dv)
        """
        total_initial_mass = self.dry_mass + current_fuel_kg
        c_eff = self.isp_s * G0

        if dv_mag <= 1e-6 or self.thrust_n <= 1e-6:
            rem_dv = c_eff * math.log(total_initial_mass / self.dry_mass) if current_fuel_kg > 0 else 0.0
            return 0.0, 0.0, current_fuel_kg, rem_dv

        # Exponential mass ratio: m0 / mf = exp(dv / c)
        mass_ratio = math.exp(dv_mag / c_eff)
        final_mass = total_initial_mass / mass_ratio
        prop_consumed = total_initial_mass - final_mass

        if prop_consumed > current_fuel_kg:
            # Out of fuel: satellite cannot deliver required delta-v
            prop_consumed = current_fuel_kg
            final_mass = self.dry_mass

        rem_fuel = max(0.0, current_fuel_kg - prop_consumed)
        m_dot = self.thrust_n / c_eff
        burn_duration = prop_consumed / m_dot if m_dot > 0 else 0.0

        rem_total_dv = c_eff * math.log((self.dry_mass + rem_fuel) / self.dry_mass) if rem_fuel > 0 else 0.0
        return burn_duration, prop_consumed, rem_fuel, rem_total_dv

    def evaluate_strategy(
        self,
        strategy_name: str,
        dv_vnc: Tuple[float, float, float],
        burn_epoch_s: float,
        epoch_start: datetime.datetime,
        r1_0: np.ndarray,
        v1_0: np.ndarray,
        r2_0: np.ndarray,
        v2_0: np.ndarray,
        tca_s: float,
    ) -> ManeuverCandidate:
        """Propagate maneuver candidate and verify post-burn clearance at TCA."""
        dv_v, dv_n, dv_c = dv_vnc
        dv_mag = math.sqrt(dv_v**2 + dv_n**2 + dv_c**2)

        # 1. Propagate primary from t=0 to burn_epoch_s
        burn_traj = self.prop_pri.propagate(r1_0, v1_0, duration_seconds=burn_epoch_s, step_seconds=max(burn_epoch_s, 1.0))
        r_at_burn = burn_traj.r[-1]
        v_at_burn = burn_traj.v[-1]

        # 2. Form VNC basis at burn epoch and map delta-V to ECI
        u_v, u_n, u_c = compute_vnc_basis(r_at_burn, v_at_burn)
        dv_eci = dv_v * u_v + dv_n * u_n + dv_c * u_c
        v_post_burn = v_at_burn + dv_eci

        # 3. Propagate post-burn state from burn_epoch_s to tca_s
        remaining_time_to_tca = max(1.0, tca_s - burn_epoch_s)
        post_traj1 = self.prop_pri.propagate(r_at_burn, v_post_burn, duration_seconds=remaining_time_to_tca, step_seconds=min(10.0, remaining_time_to_tca))
        r1_tca_new = post_traj1.r[-1]
        v1_tca_new = post_traj1.v[-1]

        # 4. Propagate secondary from t=0 to tca_s
        traj2_tca = self.prop_sec.propagate(r2_0, v2_0, duration_seconds=tca_s, step_seconds=max(tca_s, 1.0))
        r2_tca = traj2_tca.r[-1]

        # 5. Measure new miss distance
        new_miss_km = float(np.linalg.norm(r1_tca_new - r2_tca)) / 1000.0

        # Post-burn orbital elements and perigee
        post_coe = rv_to_coe(r_at_burn, v_post_burn)
        perigee_km = post_coe.perigee_altitude / 1000.0
        apogee_km = post_coe.apogee_altitude / 1000.0

        # Calculate fuel and duration
        burn_dur, fuel_used, rem_fuel, rem_dv = self.compute_propellant_budget(dv_mag, self.fuel_mass)

        # Post Pc estimation (approximate reduction)
        post_pc = 1e-9 if new_miss_km >= 15.0 else 1e-6

        # Safety verification: miss distance >= 10 km, perigee >= 200 km, sufficient fuel
        safety_cleared = (new_miss_km >= 10.0) and (perigee_km >= 200.0) and (fuel_used <= self.fuel_mass)

        burn_epoch_utc = epoch_start + datetime.timedelta(seconds=burn_epoch_s)
        lead_time_hours = (tca_s - burn_epoch_s) / 3600.0

        summary = (
            f"Strategy {strategy_name}: dv={dv_mag:.3f} m/s burned {lead_time_hours:.1f}h before TCA. "
            f"Expands miss distance to {new_miss_km:.2f} km. Fuel used: {fuel_used*1000:.1f}g, remaining budget: {rem_dv:.1f} m/s."
        )

        return ManeuverCandidate(
            strategy_name=strategy_name,
            burn_epoch_utc=burn_epoch_utc,
            lead_time_hours=lead_time_hours,
            dv_vnc=dv_vnc,
            dv_magnitude_m_s=dv_mag,
            burn_duration_s=burn_dur,
            propellant_consumed_kg=fuel_used,
            remaining_fuel_kg=rem_fuel,
            remaining_delta_v_m_s=rem_dv,
            post_miss_distance_km=new_miss_km,
            post_collision_probability=post_pc,
            new_perigee_km=perigee_km,
            new_apogee_km=apogee_km,
            safety_clearance_verified=safety_cleared,
            executive_summary=summary,
        )

    def solve_mitigation_options(
        self,
        event: ConjunctionEvent,
        r1_0: np.ndarray,
        v1_0: np.ndarray,
        r2_0: np.ndarray,
        v2_0: np.ndarray,
        epoch_start: datetime.datetime,
    ) -> List[ManeuverCandidate]:
        """Synthesize candidate maneuver options for an encounter."""
        candidates: List[ManeuverCandidate] = []
        tca_s = event.tca_seconds

        # We test burns scheduled 12 hours and 24 hours prior to TCA (or halfway if TCA is sooner)
        if tca_s > 43200.0:  # > 12h
            lead_times_s = [12.0 * 3600.0, 24.0 * 3600.0]
        elif tca_s > 7200.0:  # > 2h
            lead_times_s = [tca_s * 0.5, tca_s * 0.75]
        else:
            lead_times_s = [tca_s * 0.5]

        # Calculate estimated delta-V required for along-track phasing
        # Delta_s = 1.5 * (dv / V) * n * a * delta_t => dv = Delta_s / (3 * pi * (delta_t / T) * a / V)
        # For a standard 500 km LEO orbit (V ~= 7.6 km/s, T ~= 5600 s):
        # A 0.1 m/s burn 12 hours (approx 7.7 orbits) prior generates ~ 15-20 km along-track separation.
        for lead_s in lead_times_s:
            burn_epoch_s = max(0.0, tca_s - lead_s)
            lead_h = lead_s / 3600.0

            # Option 1: Retrograde along-track burn (lowers orbit slightly, speeds up spacecraft)
            dv_retro = (-0.15, 0.0, 0.0)
            cand_retro = self.evaluate_strategy(
                strategy_name=f"retrograde_phasing_{int(lead_h)}h",
                dv_vnc=dv_retro,
                burn_epoch_s=burn_epoch_s,
                epoch_start=epoch_start,
                r1_0=r1_0,
                v1_0=v1_0,
                r2_0=r2_0,
                v2_0=v2_0,
                tca_s=tca_s,
            )
            candidates.append(cand_retro)

            # Option 2: Prograde along-track burn (raises orbit slightly, slows down spacecraft)
            dv_pro = (0.15, 0.0, 0.0)
            cand_pro = self.evaluate_strategy(
                strategy_name=f"prograde_phasing_{int(lead_h)}h",
                dv_vnc=dv_pro,
                burn_epoch_s=burn_epoch_s,
                epoch_start=epoch_start,
                r1_0=r1_0,
                v1_0=v1_0,
                r2_0=r2_0,
                v2_0=v2_0,
                tca_s=tca_s,
            )
            candidates.append(cand_pro)

            # Option 3: Out-of-plane normal deflection
            dv_norm = (0.0, 0.25, 0.0)
            cand_norm = self.evaluate_strategy(
                strategy_name=f"out_of_plane_normal_{int(lead_h)}h",
                dv_vnc=dv_norm,
                burn_epoch_s=burn_epoch_s,
                epoch_start=epoch_start,
                r1_0=r1_0,
                v1_0=v1_0,
                r2_0=r2_0,
                v2_0=v2_0,
                tca_s=tca_s,
            )
            candidates.append(cand_norm)

        # Sort candidates: preference for verified clearance, then minimal propellant consumption
        candidates.sort(key=lambda c: (not c.safety_clearance_verified, c.propellant_consumed_kg))
        return candidates
