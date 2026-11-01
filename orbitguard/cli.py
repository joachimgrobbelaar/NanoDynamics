"""Command-line interface (CLI) for OrbitGuard AI Mission Control."""

import argparse
import asyncio
import json
import sys

from orbitguard.db.session import AsyncSessionLocal, init_db
from orbitguard.workers.background_tasks import OrbitGuardAutonomousWorker


async def cmd_screen(hours: float):
    print(f"🛰️  [OrbitGuard AI] Initiating {hours:.1f}-hour autonomous conjunction screening sweep...")
    await init_db()
    worker = OrbitGuardAutonomousWorker()
    async with AsyncSessionLocal() as session:
        tles_count = await worker.ingest_latest_tles(session)
        print(f"📥 Ingested {tles_count} fresh ephemeris records from CelesTrak.")

        events = await worker.run_screening_cycle(session, duration_hours=hours)
        print(f"\n🔍 Screening Completed. Detected {len(events)} conjunction event(s):")
        for ev in events:
            icon = "🚨" if ev["risk_tier"] in ("CRITICAL", "HIGH") else "⚠️"
            print(
                f"  {icon} [{ev['risk_tier']}] Asset: {ev['satellite_name']} vs {ev['secondary_name']}\n"
                f"     TCA: {ev['tca_epoch']} | Miss: {ev['miss_distance_km']:.3f} km | Pc: {ev['collision_probability']:.2e}\n"
                f"     Autonomous Maneuver Synthesized: {ev['maneuver_recommended']}"
            )


async def cmd_simulate():
    print("🚀 [OrbitGuard AI] Executing Threat Conjunction & Autonomous CAM Simulation...")
    from orbitguard.api.routers.simulation import simulate_threat_encounter
    from orbitguard.api.schemas import ThreatSimulationRequest

    await init_db()
    req = ThreatSimulationRequest(
        satellite_name="AeroCube-3U (Stanford)",
        threat_debris_name="COSMOS-2251 DEBRIS [CAT# 34123]",
        miss_distance_target_km=0.85,
        tca_lead_hours=18.0,
    )
    async with AsyncSessionLocal() as session:
        res = await simulate_threat_encounter(req, db=session)
        report = res["simulation_report"]
        print("\n" + "=" * 70)
        print(" MISSION CONTROL SIMULATION REPORT")
        print("=" * 70)
        print(f"Target Spacecraft: {report['satellite_name']}")
        print(f"Threat Object:     {report['chaser_debris']}")
        print(f"Nominal Miss:      {report['nominal_encounter']['miss_distance_km']:.3f} km")
        print(f"Nominal Risk Tier: {report['nominal_encounter']['risk_tier']} (Pc: {report['nominal_encounter']['collision_probability']:.2e})")
        print(f"TCA Window:        {report['nominal_encounter']['tca_utc']} (T-{report['nominal_encounter']['hours_to_tca']:.1f}h)")
        print("\n🎯 AUTONOMOUS RESOLUTION:")
        auto = report["autonomous_resolution"]
        print(f"  Strategy:        {auto['selected_strategy']}")
        print(f"  Delta-V:         {auto['delta_v_magnitude_m_s']:.4f} m/s (Vector: {auto['delta_v_vector_vnc_m_s']})")
        print(f"  Burn Duration:   {auto['burn_duration_seconds']:.1f} s")
        print(f"  Propellant Used: {auto['propellant_consumed_grams']:.2f} grams")
        print(f"  Cleared Miss:    {auto['post_maneuver_miss_distance_km']:.2f} km ({auto['clearance_multiplier']}x safety expansion)")
        print(f"  Clearance State: {auto['safety_clearance_status']}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="OrbitGuard AI — Mission Control CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    screen_parser = subparsers.add_parser("screen", help="Run conjunction screening sweep")
    screen_parser.add_argument("--hours", type=float, default=72.0, help="Screening window in hours")

    subparsers.add_parser("simulate", help="Inject threat encounter and run autonomous agentic CAM resolution")

    args = parser.parse_args()

    if args.command == "screen":
        asyncio.run(cmd_screen(args.hours))
    elif args.command == "simulate":
        asyncio.run(cmd_simulate())


if __name__ == "__main__":
    main()
