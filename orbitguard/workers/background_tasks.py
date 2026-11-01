"""Background task worker and autonomous agent pipeline for OrbitGuard AI."""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orbitguard.agents.alert_dispatcher import AlertDispatcher
from orbitguard.agents.briefing_agent import FlightBriefingAgent
from orbitguard.config import settings
from orbitguard.db.models import (
    AlertDispatch,
    AuditLog,
    ConjunctionScreening,
    ManeuverRecommendation,
    Satellite,
    Tenant,
    TLERecord,
)
from orbitguard.db.session import AsyncSessionLocal
from orbitguard.ingestion.tle_client import ParsedTLE, TLEClient
from orbitguard.physics.collision_probability import evaluate_encounter_risk
from orbitguard.physics.maneuver_solver import ManeuverSolver
from orbitguard.physics.propagator import OrbitPropagator
from orbitguard.physics.screening import ConjunctionScreener

logger = logging.getLogger(__name__)


class OrbitGuardAutonomousWorker:
    """Orchestrates scheduled ephemeris ingestion, 72h conjunction screening, and maneuver synthesis."""

    def __init__(self):
        self.tle_client = TLEClient()
        self.briefing_agent = FlightBriefingAgent()
        self.alert_dispatcher = AlertDispatcher()

    async def ingest_latest_tles(self, session: AsyncSession) -> int:
        """Scheduled worker: Ingests TLE records into the database."""
        catalog = self.tle_client.load_catalog()
        count = 0

        for item in catalog:
            # Check if record exists for this epoch
            res = await session.execute(
                select(TLERecord).where(
                    TLERecord.norad_cat_id == item.norad_cat_id,
                    TLERecord.epoch_timestamp == item.epoch_utc,
                )
            )
            existing = res.scalar_one_or_none()
            if not existing:
                tle_row = TLERecord(
                    norad_cat_id=item.norad_cat_id,
                    satellite_name=item.name,
                    line1=item.raw_line1,
                    line2=item.raw_line2,
                    epoch_timestamp=item.epoch_utc,
                    bstar=item.bstar,
                    inclination_deg=np.degrees(item.inclination_rad),
                    raan_deg=np.degrees(item.raan_rad),
                    eccentricity=item.eccentricity,
                    arg_perigee_deg=np.degrees(item.arg_perigee_rad),
                    mean_anomaly_deg=np.degrees(item.mean_anomaly_rad),
                    mean_motion_rev_day=item.mean_motion_rev_day,
                    revolution_number=item.revolution_number,
                    source="celestrak",
                )
                session.add(tle_row)
                count += 1

        await session.commit()
        logger.info(f"Ingested {count} fresh TLE records.")
        return count

    async def run_screening_cycle(
        self,
        session: AsyncSession,
        tenant_id: Optional[str] = None,
        duration_hours: float = 72.0,
    ) -> List[Dict[str, Any]]:
        """Physics-informed 72-hour conjunction screening pipeline across all registered satellites."""
        query = select(Satellite).where(Satellite.status == "operational")
        if tenant_id:
            query = query.where(Satellite.tenant_id == tenant_id)

        res = await session.execute(query)
        active_satellites = res.scalars().all()

        catalog = self.tle_client.load_catalog()
        # Separate primary candidates from secondary threat objects
        debris_targets = [item for item in catalog if "THREAT" in item.name.upper() or "DEB" in item.name.upper()]

        detected_conjunctions = []

        for sat in active_satellites:
            # Match TLE for primary satellite
            primary_tle = next((item for item in catalog if item.norad_cat_id == sat.norad_cat_id), None)
            if not primary_tle:
                # Use default AeroCube TLE
                primary_tle = catalog[0]

            coe1, r1_0, v1_0 = primary_tle.to_coe_and_state()
            prop1 = OrbitPropagator(
                mass_kg=sat.dry_mass_kg + sat.fuel_mass_kg,
                drag_area_m2=sat.cross_section_area_m2,
                cd=sat.cd_drag_coefficient,
                include_j2=True,
                include_drag=True,
            )

            # Retrieve tenant details for notification settings
            t_res = await session.execute(select(Tenant).where(Tenant.id == sat.tenant_id))
            tenant = t_res.scalar_one_or_none()

            for sec_tle in debris_targets:
                coe2, r2_0, v2_0 = sec_tle.to_coe_and_state()
                prop2 = OrbitPropagator(
                    mass_kg=25.0,  # Standard debris nominal mass
                    drag_area_m2=0.1,
                    cd=2.2,
                    include_j2=True,
                    include_drag=True,
                )

                screener = ConjunctionScreener(
                    propagator1=prop1,
                    propagator2=prop2,
                    screening_threshold_km=settings.CONJUNCTION_THRESHOLD_KM,
                )

                now_utc = datetime.datetime.now(datetime.timezone.utc)
                events = screener.screen_encounter(
                    r1_0=r1_0,
                    v1_0=v1_0,
                    r2_0=r2_0,
                    v2_0=v2_0,
                    epoch_start=now_utc,
                    duration_hours=duration_hours,
                    coarse_step_s=settings.DEFAULT_STEP_SIZE_SECONDS,
                )

                for ev in events:
                    # Calculate collision probability
                    pc, tier = evaluate_encounter_risk(
                        miss_distance_km=ev.miss_distance_km,
                        combined_radius_m=sat.hard_body_radius_m + 1.0,
                    )
                    ev.collision_probability = pc
                    ev.risk_tier = tier

                    # Record Conjunction in DB
                    conj_row = ConjunctionScreening(
                        tenant_id=sat.tenant_id,
                        primary_sat_id=sat.id,
                        secondary_norad_id=sec_tle.norad_cat_id,
                        secondary_object_name=sec_tle.name,
                        screening_window_start=now_utc,
                        screening_window_end=now_utc + datetime.timedelta(hours=duration_hours),
                        screening_window_hours=int(duration_hours),
                        propagation_method="J2_drag_numerical",
                        tca_epoch=ev.tca_datetime,
                        miss_distance_km=ev.miss_distance_km,
                        radial_miss_km=ev.radial_miss_km,
                        in_track_miss_km=ev.in_track_miss_km,
                        cross_track_miss_km=ev.cross_track_miss_km,
                        relative_speed_km_s=ev.relative_speed_km_s,
                        combined_hard_body_radius_m=sat.hard_body_radius_m + 1.0,
                        collision_probability=pc,
                        risk_tier=tier,
                        status="evaluating_maneuver" if tier in ("MODERATE", "HIGH", "CRITICAL") else "screened",
                    )
                    session.add(conj_row)
                    await session.flush()

                    # Trigger Autonomous Maneuver Solver if risk is elevated
                    maneuver_data = None
                    if tier in ("MODERATE", "HIGH", "CRITICAL") and sat.thruster_type != "none":
                        solver = ManeuverSolver(
                            propagator_primary=prop1,
                            propagator_secondary=prop2,
                            dry_mass_kg=sat.dry_mass_kg,
                            fuel_mass_kg=sat.fuel_mass_kg,
                            thrust_n=sat.thrust_n,
                            isp_s=sat.isp_s,
                        )
                        candidates = solver.solve_mitigation_options(
                            event=ev,
                            r1_0=r1_0,
                            v1_0=v1_0,
                            r2_0=r2_0,
                            v2_0=v2_0,
                            epoch_start=now_utc,
                        )

                        if candidates:
                            best = candidates[0]
                            brief_md = self.briefing_agent.generate_briefing(
                                event=ev,
                                primary_sat_name=sat.name,
                                secondary_name=sec_tle.name,
                                candidates=candidates,
                            )

                            maneuver_row = ManeuverRecommendation(
                                conjunction_id=conj_row.id,
                                satellite_id=sat.id,
                                maneuver_strategy=best.strategy_name,
                                planned_burn_epoch=best.burn_epoch_utc,
                                lead_time_hours=best.lead_time_hours,
                                dv_velocity_v_m_s=best.dv_vnc[0],
                                dv_normal_n_m_s=best.dv_vnc[1],
                                dv_conormal_c_m_s=best.dv_vnc[2],
                                dv_total_magnitude_m_s=best.dv_magnitude_m_s,
                                burn_duration_s=best.burn_duration_s,
                                propellant_consumed_kg=best.propellant_consumed_kg,
                                remaining_fuel_kg=best.remaining_fuel_kg,
                                remaining_delta_v_m_s=best.remaining_delta_v_m_s,
                                post_maneuver_miss_distance_km=best.post_miss_distance_km,
                                post_maneuver_collision_prob=best.post_collision_probability,
                                new_perigee_altitude_km=best.new_perigee_km,
                                new_apogee_altitude_km=best.new_apogee_km,
                                executive_brief_markdown=brief_md,
                                recommendation_status="recommended",
                            )
                            session.add(maneuver_row)
                            conj_row.status = "maneuver_recommended"
                            maneuver_data = best

                            # Multi-Channel Alert Dispatch
                            if tenant:
                                alert_payload = {
                                    "event_type": "CONJUNCTION_COLLISION_ALERT",
                                    "satellite_id": sat.id,
                                    "satellite_name": sat.name,
                                    "threat_object": sec_tle.name,
                                    "tca_epoch_utc": ev.tca_datetime.isoformat(),
                                    "miss_distance_km": round(ev.miss_distance_km, 3),
                                    "collision_probability": ev.collision_probability,
                                    "risk_tier": ev.risk_tier,
                                    "recommended_maneuver": {
                                        "strategy": best.strategy_name,
                                        "delta_v_m_s": round(best.dv_magnitude_m_s, 3),
                                        "burn_epoch_utc": best.burn_epoch_utc.isoformat(),
                                        "fuel_used_grams": round(best.propellant_consumed_kg * 1000.0, 2),
                                        "post_miss_km": round(best.post_miss_distance_km, 2),
                                    },
                                }

                                # Telegram Alert Dispatch
                                tg_chat = tenant.telegram_chat_id or settings.DEFAULT_TELEGRAM_CHAT_ID
                                tg_text = (
                                    f"🚨 *ORBITGUARD CONJUNCTION ALERT: {sat.name}*\n\n"
                                    f"⚠️ *Risk:* `{ev.risk_tier}` (Pc: `{ev.collision_probability:.2e}`)\n"
                                    f"🎯 *Chaser:* `{sec_tle.name}`\n"
                                    f"⏱ *TCA:* `{ev.tca_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}`\n"
                                    f"📏 *Miss Distance:* `{ev.miss_distance_km:.3f} km`\n\n"
                                    f"🚀 *RECOMMENDED CAM:*\n"
                                    f"• Plan: `{best.strategy_name}`\n"
                                    f"• $\\Delta V$: `{best.dv_magnitude_m_s:.3f} m/s` (Burn: `{best.burn_duration_s:.1f}s`)\n"
                                    f"• Clears miss distance to: *`{best.post_miss_distance_km:.2f} km`*\n"
                                    f"• Fuel cost: `{best.propellant_consumed_kg*1000.0:.1f}g`\n\n"
                                    f"[View Mission Control Brief]({tenant.webhook_url or 'http://localhost:8045'})"
                                )
                                tg_res = await self.alert_dispatcher.send_telegram_alert(tg_chat, tg_text)
                                alert_row = AlertDispatch(
                                    conjunction_id=conj_row.id,
                                    tenant_id=sat.tenant_id,
                                    channel="telegram",
                                    recipient_target=tg_chat,
                                    payload_json=alert_payload,
                                    dispatch_status=tg_res.get("status", "delivered"),
                                    delivered_at=datetime.datetime.now(datetime.timezone.utc),
                                )
                                session.add(alert_row)

                                # Webhook Dispatch
                                if tenant.webhook_url:
                                    wh_res = await self.alert_dispatcher.send_webhook_alert(tenant.webhook_url, alert_payload)
                                    wh_row = AlertDispatch(
                                        conjunction_id=conj_row.id,
                                        tenant_id=sat.tenant_id,
                                        channel="webhook",
                                        recipient_target=tenant.webhook_url,
                                        payload_json=alert_payload,
                                        dispatch_status=wh_res.get("status", "delivered"),
                                        delivered_at=datetime.datetime.now(datetime.timezone.utc),
                                    )
                                    session.add(wh_row)

                    # Audit Log
                    audit = AuditLog(
                        tenant_id=sat.tenant_id,
                        action="CONJUNCTION_SCREENED",
                        entity_type="conjunction_screening",
                        entity_id=conj_row.id,
                        details={
                            "satellite": sat.name,
                            "threat": sec_tle.name,
                            "miss_km": ev.miss_distance_km,
                            "risk": ev.risk_tier,
                        },
                    )
                    session.add(audit)

                    detected_conjunctions.append({
                        "conjunction_id": conj_row.id,
                        "satellite_name": sat.name,
                        "secondary_name": sec_tle.name,
                        "tca_epoch": ev.tca_datetime.isoformat(),
                        "miss_distance_km": ev.miss_distance_km,
                        "relative_speed_km_s": ev.relative_speed_km_s,
                        "collision_probability": ev.collision_probability,
                        "risk_tier": ev.risk_tier,
                        "maneuver_recommended": maneuver_data is not None,
                    })

        await session.commit()
        return detected_conjunctions
