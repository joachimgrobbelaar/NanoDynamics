"""Interactive simulation and verification endpoint for autonomous conjunction & CAM workflow."""

import datetime
from typing import Any, Dict
from fastapi import APIRouter, Depends
import numpy as np

from orbitguard.agents.alert_dispatcher import AlertDispatcher
from orbitguard.agents.briefing_agent import FlightBriefingAgent
from orbitguard.api.schemas import ThreatSimulationRequest
from orbitguard.db.session import get_db
from orbitguard.physics.collision_probability import evaluate_encounter_risk
from orbitguard.physics.elements import OrbitalElements, coe_to_rv
from orbitguard.physics.maneuver_solver import ManeuverSolver
from orbitguard.physics.propagator import OrbitPropagator
from orbitguard.physics.screening import ConjunctionEvent

router = APIRouter(prefix="/demo", tags=["Simulation & Demo"])


@router.post("/simulate-threat")
async def simulate_threat_encounter(
    req: ThreatSimulationRequest,
    db: Any = Depends(get_db),
):
    """Inject a realistic close approach threat and execute the full autonomous agentic loop."""
    sat_name = req.satellite_name or "AeroCube-3U"
    dry_mass = 4.0
    fuel_mass = 0.5
    drag_area = 0.03
    cd = 2.2
    thrust_n = 0.050
    isp_s = 70.0
    hard_body_r = 0.5

    # 1. Check if DB session is SQLAlchemy or in-memory
    if hasattr(db, "execute"):
        from sqlalchemy import select
        from orbitguard.db.models import (
            AlertDispatch,
            ConjunctionScreening,
            ManeuverRecommendation,
            Satellite,
            Tenant,
        )
        res = await db.execute(select(Satellite).limit(1))
        sat = res.scalar_one_or_none()
        if sat:
            sat_name = sat.name
            dry_mass = sat.dry_mass_kg
            fuel_mass = sat.fuel_mass_kg
            drag_area = sat.cross_section_area_m2
            cd = sat.cd_drag_coefficient
            thrust_n = sat.thrust_n
            isp_s = sat.isp_s
            hard_body_r = sat.hard_body_radius_m

    # 2. Design realistic 500 km LEO circular orbit for Primary CubeSat
    r_mag = 6378137.0 + 500000.0
    coe1 = OrbitalElements(
        a=r_mag,
        e=0.001,
        i=np.radians(51.64),
        raan=np.radians(120.45),
        arg_pe=np.radians(45.12),
        true_anomaly=np.radians(315.02),
    )
    r1_0, v1_0 = coe_to_rv(coe1)

    # 3. Design secondary debris object crossing at tca_lead_hours
    tca_s = req.tca_lead_hours * 3600.0
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    tca_utc = now_utc + datetime.timedelta(seconds=tca_s)

    prop1 = OrbitPropagator(
        mass_kg=dry_mass + fuel_mass,
        drag_area_m2=drag_area,
        cd=cd,
    )
    prop2 = OrbitPropagator(
        mass_kg=15.0,
        drag_area_m2=0.08,
        cd=2.2,
    )

    # Propagate primary to TCA
    traj1_tca = prop1.propagate(r1_0, v1_0, duration_seconds=tca_s, step_seconds=min(120.0, tca_s))
    r1_tca = traj1_tca.r[-1]
    v1_tca = traj1_tca.v[-1]

    # Position secondary at target miss distance
    offset_vec = np.array([req.miss_distance_target_km * 1000.0, 0.0, 0.0])
    r2_tca = r1_tca + offset_vec
    v2_tca = np.array([-v1_tca[0], v1_tca[1] * 0.8, -v1_tca[2]])

    # Back-propagate secondary to t=0
    back_traj = prop2.propagate(r2_tca, -v2_tca, duration_seconds=tca_s, step_seconds=min(120.0, tca_s))
    r2_0 = back_traj.r[-1]
    v2_0 = -back_traj.v[-1]

    # 4. Form Conjunction Event
    actual_miss_km = float(np.linalg.norm(r1_tca - r2_tca)) / 1000.0
    rel_v = float(np.linalg.norm(v1_tca - v2_tca)) / 1000.0
    pc, risk_tier = evaluate_encounter_risk(
        miss_distance_km=actual_miss_km,
        combined_radius_m=hard_body_r + 1.0,
    )

    event = ConjunctionEvent(
        tca_seconds=tca_s,
        tca_datetime=tca_utc,
        miss_distance_km=actual_miss_km,
        radial_miss_km=actual_miss_km * 0.9,
        in_track_miss_km=actual_miss_km * 0.3,
        cross_track_miss_km=actual_miss_km * 0.1,
        relative_speed_km_s=rel_v,
        r1_tca=r1_tca,
        v1_tca=v1_tca,
        r2_tca=r2_tca,
        v2_tca=v2_tca,
        risk_tier=risk_tier,
        collision_probability=pc,
    )

    # 5. Autonomous Maneuver Solver
    solver = ManeuverSolver(
        propagator_primary=prop1,
        propagator_secondary=prop2,
        dry_mass_kg=dry_mass,
        fuel_mass_kg=fuel_mass,
        thrust_n=thrust_n,
        isp_s=isp_s,
    )
    candidates = solver.solve_mitigation_options(
        event=event,
        r1_0=r1_0,
        v1_0=v1_0,
        r2_0=r2_0,
        v2_0=v2_0,
        epoch_start=now_utc,
    )
    best = candidates[0]

    # 6. Executive Briefing Generation
    briefing_agent = FlightBriefingAgent()
    brief_md = briefing_agent.generate_briefing(
        event=event,
        primary_sat_name=sat_name,
        secondary_name=req.threat_debris_name,
        candidates=candidates,
    )

    # 7. Persist to DB if SQLAlchemy session active
    if hasattr(db, "add") and hasattr(db, "commit"):
        from orbitguard.db.models import (
            AlertDispatch,
            ConjunctionScreening,
            ManeuverRecommendation,
        )
        conj_row = ConjunctionScreening(
            tenant_id=sat.tenant_id if sat else "00000000-0000-0000-0000-000000000001",
            primary_sat_id=sat.id if sat else "00000000-0000-0000-0000-000000000002",
            secondary_norad_id=34123,
            secondary_object_name=req.threat_debris_name,
            screening_window_start=now_utc,
            screening_window_end=now_utc + datetime.timedelta(hours=72),
            screening_window_hours=72,
            tca_epoch=tca_utc,
            miss_distance_km=actual_miss_km,
            radial_miss_km=event.radial_miss_km,
            in_track_miss_km=event.in_track_miss_km,
            cross_track_miss_km=event.cross_track_miss_km,
            relative_speed_km_s=rel_v,
            collision_probability=pc,
            risk_tier=risk_tier,
            status="maneuver_recommended",
        )
        db.add(conj_row)
        await db.flush()

        maneuver_row = ManeuverRecommendation(
            conjunction_id=conj_row.id,
            satellite_id=sat.id if sat else "00000000-0000-0000-0000-000000000002",
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
        db.add(maneuver_row)
        await db.commit()

    # 8. Dispatch Alerts
    dispatcher = AlertDispatcher()
    alert_payload = {
        "event": "SIMULATED_CRITICAL_CONJUNCTION",
        "satellite": sat_name,
        "debris": req.threat_debris_name,
        "tca_epoch": tca_utc.isoformat(),
        "nominal_miss_km": actual_miss_km,
        "collision_prob": pc,
        "risk_tier": risk_tier,
        "best_burn": {
            "strategy": best.strategy_name,
            "dv_m_s": best.dv_magnitude_m_s,
            "post_miss_km": best.post_miss_distance_km,
            "fuel_used_grams": best.propellant_consumed_kg * 1000.0,
        },
    }
    wh_res = await dispatcher.send_webhook_alert("http://127.0.0.1:8045/api/v1/alerts/webhook-demo", alert_payload)

    return {
        "success": True,
        "simulation_report": {
            "satellite_name": sat_name,
            "chaser_debris": req.threat_debris_name,
            "nominal_encounter": {
                "tca_utc": tca_utc.isoformat(),
                "hours_to_tca": round(req.tca_lead_hours, 2),
                "miss_distance_km": round(actual_miss_km, 3),
                "relative_speed_km_s": round(rel_v, 3),
                "collision_probability": pc,
                "risk_tier": risk_tier,
            },
            "autonomous_resolution": {
                "selected_strategy": best.strategy_name,
                "burn_epoch_utc": best.burn_epoch_utc.isoformat(),
                "lead_time_hours": round(best.lead_time_hours, 2),
                "delta_v_magnitude_m_s": round(best.dv_magnitude_m_s, 4),
                "delta_v_vector_vnc_m_s": [round(x, 4) for x in best.dv_vnc],
                "burn_duration_seconds": round(best.burn_duration_s, 1),
                "propellant_consumed_grams": round(best.propellant_consumed_kg * 1000.0, 2),
                "remaining_propellant_grams": round(best.remaining_fuel_kg * 1000.0, 2),
                "post_maneuver_miss_distance_km": round(best.post_miss_distance_km, 2),
                "clearance_multiplier": round(best.post_miss_distance_km / max(0.01, actual_miss_km), 1),
                "safety_clearance_status": "CLEARED (>10 km buffer)" if best.safety_clearance_verified else "CAUTION",
            },
            "executive_brief_preview": brief_md[:600] + "\n\n... [Full Flight Director Briefing in DB]",
            "alert_dispatch_status": wh_res.get("status"),
        },
    }
