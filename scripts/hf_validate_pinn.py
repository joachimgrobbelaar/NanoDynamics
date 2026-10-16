"""Validate RK45 and PINN orbital propagators against Hugging Face TLE history arcs."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import numpy as np

from leo_simulator import hf_data
from leo_simulator.constants import MU_EARTH, R_EARTH
from leo_simulator.orbit.elements import OrbitalElements, coe_to_rv
from leo_simulator.orbit.pinn_propagator import PINNPropagator
from leo_simulator.orbit.satellite import Satellite
from leo_simulator.propagator import OrbitPropagator


def validate_tle_arc(norad_id: int = 25544, year: int = 2024, max_pairs: int = 5) -> Dict[str, Any]:
    """Propagate between consecutive historical TLE epochs with RK45 and PINN and report RMSE."""
    records = hf_data.load_tle_history(norad_id=norad_id, year=year)
    if len(records) < 2:
        raise ValueError(f"Insufficient TLE records ({len(records)}) for NORAD {norad_id} in {year}")

    sat = Satellite(name=f"NORAD-{norad_id}", mass=420000.0, drag_area=1600.0, cd=2.2)
    rk_prop = OrbitPropagator(satellite=sat, include_j2=True, include_drag=True)
    pinn_prop = PINNPropagator(satellite=sat, mode="pinn_local")

    rk_errors_km = []
    pinn_errors_km = []
    pinn_vs_rk_km = []

    pairs = min(max_pairs, len(records) - 1)
    for idx in range(pairs):
        rec0 = records[idx]
        rec1 = records[idx + 1]
        dt = float(rec1["epoch"] - rec0["epoch"])
        if dt <= 60.0 or dt > 86400.0 * 3:
            continue

        oe0 = OrbitalElements(
            a=rec0["a_m"],
            e=max(1e-6, rec0["e"]),
            i=np.radians(rec0["inc_deg"]),
            raan=np.radians(rec0["raan_deg"]),
            arg_pe=np.radians(rec0["arg_pe_deg"]),
            nu=np.radians(rec0["mean_anomaly_deg"]),
        )
        oe1 = OrbitalElements(
            a=rec1["a_m"],
            e=max(1e-6, rec1["e"]),
            i=np.radians(rec1["inc_deg"]),
            raan=np.radians(rec1["raan_deg"]),
            arg_pe=np.radians(rec1["arg_pe_deg"]),
            nu=np.radians(rec1["mean_anomaly_deg"]),
        )
        r_target, _ = coe_to_rv(oe1, mu=MU_EARTH)

        step = min(60.0, dt / 10.0)
        res_rk = rk_prop.propagate(oe0, duration_seconds=dt, dt_eval=step)
        res_pinn = pinn_prop.propagate(oe0, duration_seconds=dt, dt_eval=step)

        r_rk = res_rk.r[-1]
        r_pinn = res_pinn.r[-1]

        # Compare semi-major axis / radial distance decay & 3D position against RK45 reference
        rk_alt_err = abs(np.linalg.norm(r_rk) - np.linalg.norm(r_target)) / 1000.0
        pinn_alt_err = abs(np.linalg.norm(r_pinn) - np.linalg.norm(r_target)) / 1000.0
        surrogate_err = float(np.linalg.norm(r_pinn - r_rk) / 1000.0)

        rk_errors_km.append(float(rk_alt_err))
        pinn_errors_km.append(float(pinn_alt_err))
        pinn_vs_rk_km.append(surrogate_err)

    return {
        "norad_id": norad_id,
        "year": year,
        "evaluated_arcs": len(rk_errors_km),
        "rk45_radial_rmse_km": float(np.sqrt(np.mean(np.square(rk_errors_km)))) if rk_errors_km else 0.0,
        "pinn_radial_rmse_km": float(np.sqrt(np.mean(np.square(pinn_errors_km)))) if pinn_errors_km else 0.0,
        "pinn_vs_rk45_3d_rmse_km": float(np.sqrt(np.mean(np.square(pinn_vs_rk_km)))) if pinn_vs_rk_km else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate PINN & RK45 against HF TLE history.")
    parser.add_argument("--norad", type=int, default=25544)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--pairs", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(validate_tle_arc(args.norad, args.year, args.pairs), indent=2))
