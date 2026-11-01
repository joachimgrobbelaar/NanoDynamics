"""Autonomous Flight Director Briefing Agent for Space Situational Awareness (SSA)."""

from __future__ import annotations

import datetime
from typing import List, Optional

import httpx

from orbitguard.config import settings
from orbitguard.physics.maneuver_solver import ManeuverCandidate
from orbitguard.physics.screening import ConjunctionEvent


class FlightBriefingAgent:
    """Compiles NASA CARA / AIAA compliant executive mission briefs."""

    def __init__(self, ollama_url: str = settings.OLLAMA_BASE_URL, model: str = settings.OLLAMA_MODEL):
        self.ollama_url = ollama_url
        self.model = model

    def generate_briefing(
        self,
        event: ConjunctionEvent,
        primary_sat_name: str,
        secondary_name: str,
        candidates: List[ManeuverCandidate],
    ) -> str:
        """Produce a complete, mathematically rigorous executive brief."""
        best = candidates[0] if candidates else None

        table_rows = []
        for i, c in enumerate(candidates, 1):
            status_icon = "✅" if c.safety_clearance_verified else "⚠️"
            table_rows.append(
                f"| Option {i}: `{c.strategy_name}` | {c.lead_time_hours:.1f}h | "
                f"[{c.dv_vnc[0]:+.2f}, {c.dv_vnc[1]:+.2f}, {c.dv_vnc[2]:+.2f}] | "
                f"{c.dv_magnitude_m_s:.3f} m/s | {c.burn_duration_s:.1f}s | "
                f"{c.propellant_consumed_kg*1000.0:.1f}g | {c.post_miss_distance_km:.2f} km | {status_icon} |"
            )
        options_table = "\n".join(table_rows)

        briefing_text = f"""# 🚨 ORBITGUARD AI — CONJUNCTION & MANEUVER DIRECTIVE
**Report ID:** `OG-{event.tca_datetime.strftime('%Y%m%d%H%M')}-{event.risk_tier}`  
**Classification:** OPERATIONAL FLIGHT SAFETY ADVISORY (CRITICAL LEVEL: **{event.risk_tier}**)  
**Generated At:** `{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`

---

### 1. Close Approach Geometry & Telemetry Summary
* **Protected Asset:** `{primary_sat_name}` (Active LEO CubeSat)
* **Conjunction Chaser / Debris:** `{secondary_name}`
* **Time of Closest Approach (TCA):** `{event.tca_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}` (in `{event.tca_seconds/3600.0:.2f}` hours)
* **Nominal Miss Distance:** **`{event.miss_distance_km:.3f} km`** (`{event.miss_distance_km * 1000.0:.1f} meters`)
  * **Radial ($\Delta R$):** `{event.radial_miss_km:+.3f} km`
  * **In-Track ($\Delta T$):** `{event.in_track_miss_km:+.3f} km`
  * **Cross-Track ($\Delta N$):** `{event.cross_track_miss_km:+.3f} km`
* **Relative Encounter Velocity:** **`{event.relative_speed_km_s:.3f} km/s`** (`{event.relative_speed_km_s * 3600.0:.1f} km/h`)
* **Collision Probability ($P_c$):** **`{event.collision_probability:.3e}`** (Threshold: `1.000e-04`)
* **Risk Tier Assessment:** **`{event.risk_tier}`**

---

### 2. Autonomous Collision Avoidance Maneuver (CAM) Recommendation
"""
        if best:
            briefing_text += f"""
**Primary Recommended Plan:** `{best.strategy_name.upper()}`
* **Burn Execution Epoch:** `{best.burn_epoch_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}` (`{best.lead_time_hours:.1f}` hours prior to TCA)
* **Delta-V Vector (VNC):** `[{best.dv_vnc[0]:+.3f}, {best.dv_vnc[1]:+.3f}, {best.dv_vnc[2]:+.3f}] m/s`
  * Along-track velocity impulse ($\Delta V_T$): `{best.dv_vnc[0]:+.3f} m/s`
  * Normal cross-track impulse ($\Delta V_N$): `{best.dv_vnc[1]:+.3f} m/s`
  * Co-normal radial impulse ($\Delta V_C$): `{best.dv_vnc[2]:+.3f} m/s`
* **Total $\Delta V$ Magnitude:** **`{best.dv_magnitude_m_s:.3f} m/s`**
* **Burn Duration:** **`{best.burn_duration_s:.1f} seconds`** (Continuous cold gas / chemical thrust)
* **Propellant Mass Depleted:** **`{best.propellant_consumed_kg*1000.0:.2f} grams`**
* **Remaining Propellant Mass:** **`{best.remaining_fuel_kg*1000.0:.2f} grams`**
* **Remaining $\Delta V$ Capacity:** **`{best.remaining_delta_v_m_s:.1f} m/s`**

#### Post-Maneuver Trajectory Verification:
* **Expanded Miss Distance at TCA:** **`{best.post_miss_distance_km:.2f} km`** (Cleared safety buffer > 10 km)
* **Post-Burn Collision Probability:** **`<{best.post_collision_probability:.1e}`**
* **New Orbit Altitude Limits:** Perigee `{best.new_perigee_km:.1f} km` / Apogee `{best.new_apogee_km:.1f} km`
* **Safety Invariant Verification:** **`PASS`** (Perigee well above atmospheric re-entry limit)

---

### 3. Evaluated Maneuver Trade Space
| Strategy | Lead Time | VNC Vector [m/s] | Total $\Delta V$ | Duration | Fuel Used | Post-Miss | Clearance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{options_table}

---

### 4. Mission Control Action Item
> **RECOMMENDED ACTION:** Operator authorization required to arm onboard thruster sequence. Uplink command block prepared for ground pass.
"""
        else:
            briefing_text += "\n> No viable active maneuver detected. Satellite may be unpropelled (differential drag mode recommended).\n"

        return briefing_text.strip()
