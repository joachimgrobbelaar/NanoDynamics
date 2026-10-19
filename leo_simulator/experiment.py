"""Parametric experimentation and orbital lifetime analysis suite.

Provides automated, multi-threaded parameter sweeps over orbital and spacecraft parameters
(mass, drag area, ballistic coefficient, altitude, eccentricity, atmospheric models)
measuring deorbit lifetime, energy dissipation, and perturbation rates.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import datetime
import os
import time
import requests
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from leo_simulator.constants import J2_EARTH, MU_EARTH, R_EARTH
from leo_simulator.models.drag import (
    BaseAtmosphere,
    ExponentialAtmosphere,
    PiecewiseExponentialAtmosphere,
)
from leo_simulator.orbit.elements import OrbitalElements
from leo_simulator.orbit.satellite import Satellite
from leo_simulator.propagator import OrbitPropagator


class ScaledAtmosphere(BaseAtmosphere):
    """Wraps an underlying atmosphere model with a density scaling factor."""

    def __init__(self, base_atmosphere: BaseAtmosphere, scale_factor: float = 1.0) -> None:
        if not np.isfinite(scale_factor) or scale_factor < 0.0:
            raise ValueError(f"Scale factor must be non-negative and finite, got {scale_factor}")
        self.base_atmosphere = base_atmosphere
        self.scale_factor = float(scale_factor)

    def density(self, altitude_m: float) -> float:
        return float(self.base_atmosphere.density(altitude_m) * self.scale_factor)


@dataclass
class SweepPointResult:
    """Telemetry and deorbit statistics for a single parameter sweep run."""

    run_id: int
    param_name: str
    param_value: float
    mass_kg: float
    drag_area_m2: float
    cd: float
    ballistic_coeff_kg_m2: float  # B = m / (Cd * A)
    ballistic_factor_m2_kg: float  # B* = (Cd * A) / m
    initial_altitude_km: float
    initial_eccentricity: float
    initial_inclination_deg: float
    initial_raan_deg: float
    final_altitude_km: float
    final_eccentricity: float
    deorbit_detected: bool
    lifetime_seconds: float
    lifetime_hours: float
    lifetime_days: float
    initial_energy_j_kg: float
    final_energy_j_kg: float
    energy_loss_j_kg: float
    mean_decay_rate_km_day: float
    computation_time_ms: float
    max_velocity_km_s: float
    min_altitude_km: float
    max_altitude_km: float


@dataclass
class ExperimentResult:
    """Summary container for an entire parametric experiment sweep."""

    experiment_id: str
    param_name: str
    timestamp: str
    base_params: dict[str, Any]
    atmosphere_type: str
    density_scale: float
    results: list[SweepPointResult]
    csv_filepath: str | None = None
    trajectories: list[dict[str, Any]] | None = None


def get_atmosphere_by_name(name: str = "piecewise", density_scale: float = 1.0) -> BaseAtmosphere:
    """Factory creating atmospheric models based on configuration name."""
    name_lower = name.lower()
    if "piecewise" in name_lower or "us_standard" in name_lower:
        base = PiecewiseExponentialAtmosphere()
    elif "high_solar" in name_lower:
        base = ExponentialAtmosphere(h0=400_000.0, rho0=5.0e-12, scale_height=65_000.0)
    elif "low_solar" in name_lower:
        base = ExponentialAtmosphere(h0=400_000.0, rho0=1.2e-12, scale_height=50_000.0)
    elif "app_default" in name_lower:
        base = ExponentialAtmosphere(h0=350_000.0, rho0=9.5e-12, scale_height=53_200.0)
    else:
        base = PiecewiseExponentialAtmosphere()

    if abs(density_scale - 1.0) > 1e-6:
        return ScaledAtmosphere(base, scale_factor=density_scale)
    return base


def run_single_simulation(
    name: str = "SimRun",
    mass: float = 4.0,
    drag_area: float = 0.03,
    cd: float = 2.2,
    altitude_km: float = 400.0,
    eccentricity: float = 0.001,
    inclination_deg: float = 51.6,
    raan_deg: float = 45.0,
    atmosphere_type: str = "piecewise",
    density_scale: float = 1.0,
    include_j2: bool = True,
    include_drag: bool = True,
    include_moon: bool = False,
    max_duration_seconds: float = 30 * 86400.0,
    dt_eval: float | None = None,
    include_trajectory: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Execute a single orbital simulation and extract decay and deorbit lifetime metrics."""
    t0 = time.perf_counter()

    atm = get_atmosphere_by_name(atmosphere_type, density_scale) if include_drag else None
    sat = Satellite(name=name, mass=mass, drag_area=drag_area, cd=cd)

    # Fast tolerances for sweeps
    rtol = 1e-7 if not include_trajectory else 1e-8
    atol = 1e-8 if not include_trajectory else 1e-9

    prop = OrbitPropagator(
        satellite=sat,
        atmosphere=atm,
        include_central_gravity=True,
        include_j2=include_j2,
        include_drag=include_drag,
        include_moon=include_moon,
        solver_method="DOP853",
        rtol=rtol,
        atol=atol,
    )

    a_m = R_EARTH + altitude_km * 1000.0
    orbit = OrbitalElements(
        a=a_m,
        e=eccentricity,
        i=np.radians(inclination_deg),
        raan=np.radians(raan_deg),
        arg_pe=0.0,
        nu=0.0,
    )

    res = prop.propagate(
        initial_state=orbit,
        duration_seconds=max_duration_seconds,
        dt_eval=dt_eval if dt_eval is not None else 60.0 if include_trajectory else None,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    r_final = res.r[-1]
    v_final = res.v[-1]
    r_final_norm = float(np.linalg.norm(r_final))
    v_final_norm = float(np.linalg.norm(v_final))
    final_alt_km = (r_final_norm - R_EARTH) / 1000.0

    r0_norm = float(np.linalg.norm(res.r[0]))
    v0_norm = float(np.linalg.norm(res.v[0]))

    phi_j2_0 = (MU_EARTH * J2_EARTH * (R_EARTH**2) / (2.0 * r0_norm**3)) * (3.0 * (res.r[0][2] / r0_norm) ** 2 - 1.0)
    e0 = 0.5 * v0_norm**2 - MU_EARTH / r0_norm + phi_j2_0

    phi_j2_f = (MU_EARTH * J2_EARTH * (R_EARTH**2) / (2.0 * r_final_norm**3)) * (3.0 * (r_final[2] / r_final_norm) ** 2 - 1.0)
    ef = 0.5 * v_final_norm**2 - MU_EARTH / r_final_norm + phi_j2_f

    lifetime_sec = float(res.reentry_time) if res.reentry_detected and res.reentry_time is not None else float(res.t[-1])
    lifetime_days = lifetime_sec / 86400.0

    alt_drop_km = altitude_km - final_alt_km
    decay_rate = (alt_drop_km / lifetime_days) if lifetime_days > 1e-4 else 0.0

    b_coeff = mass / (cd * drag_area) if (cd * drag_area) > 0 else float("inf")
    b_star = (cd * drag_area) / mass if mass > 0 else 0.0

    v_norms = np.linalg.norm(res.v, axis=1) if len(res.v) > 0 else np.array([v_final_norm])
    r_norms = np.linalg.norm(res.r, axis=1) if len(res.r) > 0 else np.array([r_final_norm])
    alts_km = (r_norms - R_EARTH) / 1000.0
    
    metrics = {
        "mass_kg": mass,
        "drag_area_m2": drag_area,
        "cd": cd,
        "ballistic_coeff_kg_m2": b_coeff,
        "ballistic_factor_m2_kg": b_star,
        "initial_altitude_km": altitude_km,
        "initial_eccentricity": eccentricity,
        "initial_inclination_deg": inclination_deg,
        "initial_raan_deg": raan_deg,
        "final_altitude_km": final_alt_km,
        "final_eccentricity": float(res.eccentricities[-1]) if len(res.eccentricities) > 0 else eccentricity,
        "deorbit_detected": bool(res.reentry_detected),
        "lifetime_seconds": lifetime_sec,
        "lifetime_hours": lifetime_sec / 3600.0,
        "lifetime_days": lifetime_days,
        "initial_energy_j_kg": float(e0),
        "final_energy_j_kg": float(ef),
        "energy_loss_j_kg": float(e0 - ef),
        "mean_decay_rate_km_day": float(decay_rate),
        "computation_time_ms": float(elapsed_ms),
        "max_velocity_km_s": float(np.max(v_norms)) / 1000.0,
        "min_altitude_km": float(np.min(alts_km)),
        "max_altitude_km": float(np.max(alts_km)),
    }

    traj_dict = None
    if include_trajectory:
        traj_dict = {
            "name": name,
            "t": res.t.tolist(),
            "x": res.r[:, 0].tolist(),
            "y": res.r[:, 1].tolist(),
            "z": res.r[:, 2].tolist(),
            "vx": res.v[:, 0].tolist(),
            "vy": res.v[:, 1].tolist(),
            "vz": res.v[:, 2].tolist(),
            "alt_km": ((np.linalg.norm(res.r, axis=1) - R_EARTH) / 1000.0).tolist(),
            "reentry": bool(res.reentry_detected),
        }

    return metrics, traj_dict


def _evaluate_sweep_point(args: tuple[int, str, float, dict[str, Any], str, float, float, bool]) -> tuple[SweepPointResult, dict[str, Any] | None]:
    idx, param_name, val, base_params, atmosphere_type, density_scale, max_duration_seconds, quiet = args
    run_params = dict(base_params)
    curr_density_scale = density_scale

    if param_name == "altitude_km":
        run_params["altitude_km"] = float(val)
    elif param_name == "mass":
        run_params["mass"] = float(val)
    elif param_name == "drag_area":
        run_params["drag_area"] = float(val)
    elif param_name == "cd":
        run_params["cd"] = float(val)
    elif param_name == "ballistic_coefficient":
        cd = float(run_params.get("cd", 2.2))
        area = float(run_params.get("drag_area", 0.03))
        run_params["mass"] = float(val * cd * area)
    elif param_name == "eccentricity":
        run_params["eccentricity"] = float(val)
    elif param_name == "inclination_deg":
        run_params["inclination_deg"] = float(val)
    elif param_name == "density_scale":
        curr_density_scale = float(val)
    else:
        run_params[param_name] = float(val)

    run_name = f"{param_name}_{val:.4g}"
    
    cloud_url = os.environ.get("FIREBASE_SWEEP_URL")
    if cloud_url:
        payload = {
            "data": {
                "params": {
                    "name": run_name,
                    "mass": float(run_params["mass"]),
                    "drag_area": float(run_params["drag_area"]),
                    "cd": float(run_params["cd"]),
                    "altitude_km": float(run_params["altitude_km"]),
                    "eccentricity": float(run_params["eccentricity"]),
                    "inclination_deg": float(run_params["inclination_deg"]),
                    "raan_deg": float(run_params.get("raan_deg", 0.0)),
                    "include_j2": bool(run_params.get("include_j2", True)),
                    "include_drag": bool(run_params.get("include_drag", True)),
                    "include_moon": bool(run_params.get("include_moon", False)),
                },
                "atmosphere_type": atmosphere_type,
                "density_scale": curr_density_scale,
                "max_duration_seconds": max_duration_seconds,
                "include_trajectory": not quiet
            }
        }
        res = requests.post(cloud_url, json=payload, headers={"Content-Type": "application/json"})
        res.raise_for_status()
        res_data = res.json().get("result", {})
        metrics = res_data.get("metrics")
        traj = res_data.get("trajectory")
    else:
        metrics, traj = run_single_simulation(
            name=run_name,
            mass=float(run_params["mass"]),
            drag_area=float(run_params["drag_area"]),
            cd=float(run_params["cd"]),
            altitude_km=float(run_params["altitude_km"]),
            eccentricity=float(run_params["eccentricity"]),
            inclination_deg=float(run_params["inclination_deg"]),
            raan_deg=float(run_params.get("raan_deg", 0.0)),
            atmosphere_type=atmosphere_type,
            density_scale=curr_density_scale,
            include_j2=bool(run_params.get("include_j2", True)),
            include_drag=bool(run_params.get("include_drag", True)),
            include_moon=bool(run_params.get("include_moon", False)),
            max_duration_seconds=max_duration_seconds,
            include_trajectory=not quiet,
        )

    res_obj = SweepPointResult(
        run_id=idx + 1,
        param_name=param_name,
        param_value=float(val),
        mass_kg=metrics["mass_kg"],
        drag_area_m2=metrics["drag_area_m2"],
        cd=metrics["cd"],
        ballistic_coeff_kg_m2=metrics["ballistic_coeff_kg_m2"],
        ballistic_factor_m2_kg=metrics["ballistic_factor_m2_kg"],
        initial_altitude_km=metrics["initial_altitude_km"],
        initial_eccentricity=metrics["initial_eccentricity"],
        initial_inclination_deg=metrics["initial_inclination_deg"],
        initial_raan_deg=metrics["initial_raan_deg"],
        final_altitude_km=metrics["final_altitude_km"],
        final_eccentricity=metrics["final_eccentricity"],
        deorbit_detected=metrics["deorbit_detected"],
        lifetime_seconds=metrics["lifetime_seconds"],
        lifetime_hours=metrics["lifetime_hours"],
        lifetime_days=metrics["lifetime_days"],
        initial_energy_j_kg=metrics["initial_energy_j_kg"],
        final_energy_j_kg=metrics["final_energy_j_kg"],
        energy_loss_j_kg=metrics["energy_loss_j_kg"],
        mean_decay_rate_km_day=metrics["mean_decay_rate_km_day"],
        computation_time_ms=metrics["computation_time_ms"],
        max_velocity_km_s=metrics["max_velocity_km_s"],
        min_altitude_km=metrics["min_altitude_km"],
        max_altitude_km=metrics["max_altitude_km"],
    )
    return res_obj, traj


def run_parametric_sweep(
    param_name: str,
    values: list[float],
    base_params: dict[str, Any] | None = None,
    atmosphere_type: str = "piecewise",
    density_scale: float = 1.0,
    max_duration_seconds: float = 30 * 86400.0,
    quiet: bool = True,
    output_dir: str | None = None,
) -> ExperimentResult:
    """Run an automated multi-threaded parameter sweep and save clean CSV results."""
    if base_params is None:
        base_params = {
            "name": "SweepSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 300.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
            "include_j2": True,
            "include_drag": True,
            "include_moon": False,
        }

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_id = f"sweep_{param_name}_{timestamp}"

    tasks = [
        (idx, param_name, val, base_params, atmosphere_type, density_scale, max_duration_seconds, quiet)
        for idx, val in enumerate(values)
    ]

    # Execute points concurrently in parallel threads
    num_workers = min(len(values), (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results_and_trajs = list(executor.map(_evaluate_sweep_point, tasks))

    results = [r[0] for r in results_and_trajs]
    trajectories = [r[1] for r in results_and_trajs if r[1] is not None]

    # Sort results by run_id
    results.sort(key=lambda x: x.run_id)

    # Export to CSV
    csv_file = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, f"{experiment_id}.csv")
        save_experiment_to_csv(results, csv_file, base_params, atmosphere_type, density_scale)

    return ExperimentResult(
        experiment_id=experiment_id,
        param_name=param_name,
        timestamp=timestamp,
        base_params=base_params,
        atmosphere_type=atmosphere_type,
        density_scale=density_scale,
        results=results,
        csv_filepath=csv_file,
        trajectories=trajectories if not quiet else None,
    )


def save_experiment_to_csv(
    results: list[SweepPointResult],
    filepath: str,
    base_params: dict[str, Any] | None = None,
    atmosphere_type: str = "piecewise",
    density_scale: float = 1.0,
) -> str:
    """Write experiment sweep results to a cleanly structured CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    fieldnames = [
        "run_id",
        "param_name",
        "param_value",
        "mass_kg",
        "drag_area_m2",
        "cd",
        "ballistic_coeff_kg_m2",
        "ballistic_factor_m2_kg",
        "initial_altitude_km",
        "initial_eccentricity",
        "initial_inclination_deg",
        "initial_raan_deg",
        "final_altitude_km",
        "final_eccentricity",
        "deorbit_detected",
        "lifetime_seconds",
        "lifetime_hours",
        "lifetime_days",
        "initial_energy_j_kg",
        "final_energy_j_kg",
        "energy_loss_j_kg",
        "mean_decay_rate_km_day",
        "max_velocity_km_s",
        "min_altitude_km",
        "max_altitude_km",
        "computation_time_ms",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    return filepath
