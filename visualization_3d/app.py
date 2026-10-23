import asyncio
from dataclasses import asdict
import json
import math
import os
import re
import sys
import urllib.request

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from leo_simulator import (
    R_EARTH,
    ExponentialAtmosphere,
    OrbitalElements,
    OrbitPropagator,
    PINNPropagator,
    Satellite,
    coe_to_rv,
    compute_orbital_collision_impulse,
    detect_conjunction,
    generate_collision_debris_fragments,
    eci_to_geodetic,
    extract_trajectory_key_events,
)
from leo_simulator.orbit.astrodynamics import (
    rv_to_keplerian,
    compute_vnc_basis,
    compute_rf_link_budget,
    compute_space_environment,
)
from leo_simulator.experiment import (
    get_atmosphere_by_name,
    run_parametric_sweep,
    run_single_simulation,
)
from leo_simulator.models.gravity import moon_position
from leo_simulator.tle_tracker import (
    GROUPS as LIVE_SATELLITE_GROUPS,
    get_live_catalog,
    get_satellite_by_norad,
)
from leo_simulator.space_weather import SpaceWeatherManager
from leo_simulator.orbit.poi_orbit_designer import (
    design_orbit_for_poi,
    get_all_pois,
    get_specialized_orbit_presets,
)
from leo_simulator.orbit.isl_network import (
    ISLNetworkMesh,
    SatelliteNode,
)
from leo_simulator.orbit.finite_burn import (
    compute_finite_burn_duration,
    propagate_finite_burn,
)


app = FastAPI(docs_url="/api/docs", redoc_url="/api/redoc")

from hardware_in_the_loop.esp32_manager import ESP32DeviceManager

device_manager = ESP32DeviceManager.get_instance()

class DevicePayload(BaseModel):
    id: str | None = None
    name: str
    mac_address: str = ""
    ip_address: str = ""
    transport: str = "espnow_gateway"
    channel: int = 1


active_hil_worker: WebSocket = None

@app.websocket("/ws/hil-worker")
async def websocket_hil_worker(websocket: WebSocket):
    global active_hil_worker
    await websocket.accept()
    active_hil_worker = websocket
    print("HIL worker connected.")
    try:
        while True:
            # Keep alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        active_hil_worker = None
        print("HIL worker disconnected.")


active_camera_clients: set[WebSocket] = set()

@app.websocket("/ws/camera-feed")
async def websocket_camera_feed(websocket: WebSocket):
    await websocket.accept()
    active_camera_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            dead_clients = set()
            for client in active_camera_clients:
                if client != websocket:
                    try:
                        await client.send_text(data)
                    except Exception:
                        dead_clients.add(client)
            active_camera_clients.difference_update(dead_clients)
    except WebSocketDisconnect:
        active_camera_clients.discard(websocket)
    except Exception:
        active_camera_clients.discard(websocket)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_messages = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []) if l != "body")
        msg = err.get("msg", "")
        msg = msg.removeprefix("Value error, ")
        if loc:
            error_messages.append(f"{loc}: {msg}")
        else:
            error_messages.append(msg)
    clean_detail = "; ".join(error_messages) if error_messages else "Validation error"
    return JSONResponse(status_code=422, content={"detail": clean_detail})


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

SAVED_SIMS_DIR = os.path.join(BASE_DIR, "saved_sims")
os.makedirs(SAVED_SIMS_DIR, exist_ok=True)

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

BODIES = {
    "Earth": {
        "name": "Earth",
        "mu": 3.986004418e14,
        "radius_m": 6378137.0,
        "j2": 1.08262668e-3,
        "omega": 7.2921150e-5,
        "has_atmosphere": True,
        "min_alt_km": 100.0,
        "max_alt_km": 2000.0,
        "default_alt_km": 400.0,
        "default_ecc": 0.001,
        "default_inc": 51.6,
        "default_raan": 45.0,
        "default_chunk_duration": 86400.0,
        "dt_eval": 60.0,
    },
    "Moon": {
        "name": "Moon",
        "mu": 4.902800066e12,
        "radius_m": 1737400.0,
        "j2": 2.027e-4,
        "omega": 2.6617e-6,
        "has_atmosphere": False,
        "min_alt_km": 20.0,
        "max_alt_km": 10000.0,
        "default_alt_km": 100.0,
        "default_ecc": 0.001,
        "default_inc": 90.0,
        "default_raan": 0.0,
        "default_chunk_duration": 86400.0,
        "dt_eval": 60.0,
    },
    "Sun": {
        "name": "Sun",
        "mu": 1.32712440018e20,
        "radius_m": 696340000.0,
        "j2": 2.0e-7,
        "omega": 2.87e-6,
        "has_atmosphere": False,
        "min_alt_km": 10000.0,
        "max_alt_km": 1500000000.0,
        "default_alt_km": 149597870.0,
        "default_ecc": 0.0167,
        "default_inc": 0.0,
        "default_raan": 0.0,
        "default_chunk_duration": 2592000.0,
        "dt_eval": 3600.0,
    },
    "Mercury": {
        "name": "Mercury",
        "mu": 2.2032e13,
        "radius_m": 2439700.0,
        "j2": 6.0e-5,
        "omega": 1.24001e-6,
        "has_atmosphere": False,
        "min_alt_km": 20.0,
        "max_alt_km": 50000.0,
        "default_alt_km": 250.0,
        "default_ecc": 0.001,
        "default_inc": 45.0,
        "default_raan": 0.0,
        "default_chunk_duration": 86400.0,
        "dt_eval": 60.0,
    },
    "Venus": {
        "name": "Venus",
        "mu": 3.24859e14,
        "radius_m": 6051800.0,
        "j2": 4.458e-6,
        "omega": -2.9924e-7,
        "has_atmosphere": True,
        "min_alt_km": 150.0,
        "max_alt_km": 100000.0,
        "default_alt_km": 400.0,
        "default_ecc": 0.001,
        "default_inc": 30.0,
        "default_raan": 0.0,
        "default_chunk_duration": 86400.0,
        "dt_eval": 60.0,
    },
    "Mars": {
        "name": "Mars",
        "mu": 4.282837e13,
        "radius_m": 3389500.0,
        "j2": 1.96045e-3,
        "omega": 7.0882e-5,
        "has_atmosphere": True,
        "min_alt_km": 80.0,
        "max_alt_km": 100000.0,
        "default_alt_km": 350.0,
        "default_ecc": 0.001,
        "default_inc": 25.0,
        "default_raan": 0.0,
        "default_chunk_duration": 86400.0,
        "dt_eval": 60.0,
    },
}


class SatelliteParams(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Satellite name")
    parent_body: str = Field("Earth", description="Central body: Earth or Moon")
    mass: float = Field(..., gt=0.0, le=1000000.0, description="Mass in kg")
    drag_area: float = Field(..., ge=0.0, le=50000.0, description="Cross-section area in m^2")
    cd: float = Field(..., ge=0.0, le=20.0, description="Drag coefficient")
    altitude_km: float = Field(..., description="Altitude in km")
    eccentricity: float = Field(..., ge=0.0, lt=1.0, description="Orbital eccentricity")
    inclination_deg: float = Field(..., ge=0.0, le=180.0, description="Inclination in degrees")
    raan_deg: float = Field(..., ge=0.0, lt=360.0, description="RAAN in degrees")
    arg_periapsis_deg: float = Field(0.0, ge=0.0, lt=360.0, description="Argument of Periapsis in degrees")
    true_anomaly_deg: float = Field(0.0, ge=0.0, lt=360.0, description="True Anomaly (Phase) in degrees")
    # Atmosphere configuration
    atmosphere_type: str = Field("piecewise", description="Atmosphere model: piecewise, high_solar, low_solar, app_default")
    density_scale: float = Field(1.0, ge=0.001, le=1000.0, description="Atmospheric density multiplier")
    # Perturbation toggles
    include_j2: bool = Field(True, description="Include J2 oblateness perturbation")
    include_drag: bool = Field(True, description="Include atmospheric drag")
    include_moon: bool = Field(False, description="Include lunar third-body gravity")
    include_srp: bool = Field(False, description="Include Solar Radiation Pressure (SRP) perturbation with ray-tracing")
    cr: float = Field(1.3, ge=1.0, le=2.0, description="Radiation pressure reflectivity coefficient")
    # Visual representation
    color: str = Field("#38bdf8", description="Hex color for trajectory and indicator")
    icon: str = Field("satellite", description="Avatar icon type: satellite, rocket, astronaut, alien, ufo, sphere")
    model_key: str | None = Field("auto", description="3D model asset: auto, cubesat_vr, cubesat_club, landsat_8, hubble, space_shuttle, space_station, tie_fighter, vega, satellite, simple_satellite, need_some_space")
    t_start: float = Field(0.0, ge=0.0, description="Initial simulation epoch in seconds")
    propagation_mode: str = Field("rk45", description="Propagation mode: 'rk45', 'pinn_local', 'pinn_esp32', or 'pinn'")
    hardware_device_id: str | None = Field("esp32-node-1", description="Target ESP32 Device ID for HIL execution")
    allow_decay: bool = Field(False, description="Allow sub-orbital or decaying orbits (for burns, deorbit, and collision debris)")

    @model_validator(mode="after")
    def validate_orbit_safety(self) -> "SatelliteParams":
        mode_lower = self.propagation_mode.lower()
        if mode_lower.startswith("pinn") and self.parent_body != "Earth":
            raise ValueError(
                "propagation_mode: The PINN surrogate model currently only supports Earth orbits. Select 'rk45' for Moon, Mars, Venus, or Mercury simulations."
            )
        body = BODIES.get(self.parent_body, BODIES["Earth"])
        if not self.allow_decay:
            if self.altitude_km < body["min_alt_km"] or self.altitude_km > body["max_alt_km"]:
                raise ValueError(
                    f"altitude_km: Altitude {self.altitude_km:.1f} km is out of allowed range [{body['min_alt_km']:.1f}, {body['max_alt_km']:.1f}] km for {self.parent_body}."
                )
            
            # Perigee radius check: r_p = a * (1 - e)
            r_initial_m = body["radius_m"] + self.altitude_km * 1000.0
            r_perigee_m = r_initial_m * (1.0 - self.eccentricity)
            min_safe_radius_m = body["radius_m"] + 50_000.0 if self.parent_body == "Earth" else body["radius_m"] + 10_000.0
            if r_perigee_m <= min_safe_radius_m:
                raise ValueError(
                    f"eccentricity: Perigee altitude {(r_perigee_m - body['radius_m']) / 1000.0:.1f} km is below minimum safe threshold of 50 km (collision risk)."
                )
        return self


class StreamRequest(BaseModel):
    """Request a rolling chunk of trajectory from a known Cartesian state."""
    params: SatelliteParams
    t_start: float = Field(..., ge=0.0, description="Simulation time at start of chunk (s)")
    state: list[float] = Field(..., min_length=6, max_length=6,
                               description="[x, y, z, vx, vy, vz] in metres / m/s")
    chunk_duration: float = Field(600.0, ge=10.0, le=315360000.0,
                                  description="Duration of this chunk in seconds")
    dt_eval: float = Field(None, description="Optional custom delta time for evaluation points (s)")


class BurnRequest(BaseModel):
    params: SatelliteParams
    t_burn: float = Field(..., description="Time at burn (s)")
    current_state: list[float] = Field(..., min_length=6, max_length=6, description="[x, y, z, vx, vy, vz]")
    dv_prograde: float = Field(0.0, description="Delta-v in prograde/transverse direction (m/s)")
    dv_normal: float = Field(0.0, description="Delta-v in normal/out-of-plane direction (m/s)")
    dv_radial: float = Field(0.0, description="Delta-v in radial-outward direction (m/s)")
    thrust_n: float = Field(20.0, ge=0.001, description="Finite thrust in Newtons (default 20.0 N)")
    isp_s: float = Field(300.0, ge=10.0, description="Specific impulse in seconds (default 300.0 s)")
    is_finite: bool = Field(False, description="Whether to execute as continuous finite-duration burn")


class CollisionRequest(BaseModel):
    sat1_params: SatelliteParams
    sat1_state: list[float] = Field(..., min_length=6, max_length=6, description="[x, y, z, vx, vy, vz] for satellite 1")
    sat2_params: SatelliteParams
    sat2_state: list[float] = Field(..., min_length=6, max_length=6, description="[x, y, z, vx, vy, vz] for satellite 2")
    t_collision: float = Field(..., ge=0.0, description="Epoch of collision in seconds")
    restitution: float = Field(0.2, ge=0.0, le=1.0, description="Restitution coefficient (0=plastic, 1=elastic)")
    post_collision_duration: float = Field(86400.0, ge=10.0, le=604800.0, description="Post-collision propagation window (s)")
    dt_eval: float = Field(None, description="Evaluation delta time (s)")


class ExperimentRequest(BaseModel):
    param_name: str = Field(..., description="Independent variable name to sweep")
    values: list[float] = Field(..., min_length=1, max_length=1000, description="List of parameter values")
    base_params: SatelliteParams
    atmosphere_type: str = Field("piecewise", description="Atmosphere model type")
    density_scale: float = Field(1.0, ge=0.001, le=1000.0, description="Density scale multiplier")
    max_duration_days: float = Field(30.0, ge=0.1, le=365.0, description="Max propagation duration in days")
    quiet: bool = Field(True, description="True for fast metric sweep, False to include trajectory coordinates")


class SavePayload(BaseModel):
    filename: str
    satellites: list


class LiveTrackRequest(BaseModel):
    norad_id: int = Field(..., description="NORAD Catalog ID of the satellite")
    propagation_mode: str = Field("rk45", description="Propagation mode: 'rk45' or 'pinn'")
    custom_name: str | None = Field(None, description="Optional custom display name")
    t_start: float = Field(0.0, ge=0.0, description="Start time epoch in seconds")


class AnalyticalExportRequest(BaseModel):
    params: SatelliteParams
    t_start: float = Field(0.0, ge=0.0, description="Start time epoch in seconds")
    duration_seconds: float = Field(86400.0, ge=10.0, le=604800.0, description="Dataset duration in seconds")
    dt_eval: float = Field(60.0, ge=1.0, le=3600.0, description="Sampling time interval in seconds")
    initial_state: list[float] | None = Field(None, description="Optional [x,y,z,vx,vy,vz] initial state override")


def sanitize_filename(filename: str) -> str:
    if not re.match(r'^[\w\-]+$', filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename format. Use only alphanumeric characters, dashes, and underscores."
        )
    return filename


def _build_propagator(params: SatelliteParams) -> OrbitPropagator:
    body = BODIES.get(params.parent_body, BODIES["Earth"])
    sat = Satellite(name=params.name, mass=params.mass, drag_area=params.drag_area, cd=params.cd)
    use_drag = body["has_atmosphere"] and params.include_drag
    atm = get_atmosphere_by_name(params.atmosphere_type, params.density_scale) if use_drag else None

    return OrbitPropagator(
        satellite=sat,
        atmosphere=atm,
        include_central_gravity=True,
        include_j2=(body["j2"] > 0 and params.include_j2),
        include_drag=use_drag,
        include_earth_rotation=use_drag,
        include_moon=(params.parent_body == "Earth" and params.include_moon),
        include_srp=params.include_srp,
        cr=params.cr,
        mu=body["mu"],
        r_earth=body["radius_m"],
        j2=body["j2"],
        omega_earth=body["omega"],
        rtol=1e-8,
        atol=1e-9,
        # Terminal re-entry floor matches the input validator's own collision
        # lines (50 km Earth / 10 km Moon): crossing it ends the run instead
        # of flying on to impact.
        min_altitude_reentry=50_000.0 if body["has_atmosphere"] else 10_000.0,
    )


def _compute_forces(prop: OrbitPropagator, t_arr, r_arr, v_arr):
    """Compute force vectors along a trajectory. Returns dict of lists."""
    forces = {"central": [], "j2": [], "j3": [], "j4": [], "j22": [], "drag": [], "moon": [], "sun": [], "srp": [], "tides": [], "magnetic": [], "total": []}
    for i in range(len(t_arr)):
        acc = prop.dynamics.compute_accelerations(r_arr[i], v_arr[i], t=float(t_arr[i]))
        for k in forces:
            if k in acc:
                forces[k].append(acc[k].tolist())
            else:
                forces[k].append([0.0, 0.0, 0.0])
    return forces


ML_LOG_PATH = os.path.join(PROJECT_ROOT, "ai", "data", "sim_log.jsonl")


def _log_sim_for_ml(params: SatelliteParams, dt_eval: float, chunk: dict) -> None:
    """Best-effort append of each /simulate run for the local ML agent.

    Never raises: logging must not break the simulation response.
    """
    try:
        os.makedirs(os.path.dirname(ML_LOG_PATH), exist_ok=True)
        rec = {"name": params.name, "params": params.model_dump(),
               "dt": float(dt_eval),
               "trajectory": {k: chunk[k] for k in ("t", "x", "y", "z", "vx", "vy", "vz")
                              if k in chunk}}
        with open(ML_LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass



async def _propagate_chunk_async(prop, initial_state, t_start: float, chunk_duration: float, dt_eval: float = 10.0):
    if hasattr(prop, 'propagate_async') and prop.network_ws is not None:
        return await prop.propagate_async(
            initial_state=np.asarray(initial_state, dtype=np.float64),
            duration_seconds=chunk_duration,
            t_start=t_start,
            dt_eval=dt_eval,
        )
    else:
        return await asyncio.to_thread(
            prop.propagate,
            initial_state=np.asarray(initial_state, dtype=np.float64),
            duration_seconds=chunk_duration,
            t_start=t_start,
            dt_eval=dt_eval,
        )

def _propagate_chunk(prop: OrbitPropagator, initial_state, t_start: float,
                     chunk_duration: float, dt_eval: float = 10.0):
    """Propagate a chunk. Returns PropagationResult."""
    return prop.propagate(
        initial_state=np.asarray(initial_state, dtype=np.float64),
        duration_seconds=chunk_duration,
        t_start=t_start,
        dt_eval=dt_eval,
    )


def _format_chunk(res, prop: OrbitPropagator, radius_m: float, include_forces: bool = True) -> dict:
    """Convert PropagationResult to columnar JSON-serialisable dict with geodetic coordinates."""
    t = res.t.tolist()
    r = np.array(res.r, copy=True)
    v = np.array(res.v, copy=True)
    r_norms = np.linalg.norm(r, axis=1)
    subsurface = r_norms < radius_m
    reentry_flag = bool(res.reentry_detected)
    reentry_epoch = float(res.reentry_time) if res.reentry_time is not None else None
    if np.any(subsurface):
        first_idx = int(np.argmax(subsurface))
        for i in range(first_idx, len(r)):
            n = r_norms[i]
            if n < radius_m:
                r[i] = r[i] * (radius_m / max(n, 1e-6))
                v[i] = 0.0
        reentry_flag = True
        if reentry_epoch is None:
            reentry_epoch = float(res.t[first_idx])
    altitudes = ((np.linalg.norm(r, axis=1) - radius_m) / 1000.0).tolist()
    lat_deg, lon_deg, _ = eci_to_geodetic(r, res.t, radius_m=radius_m, omega_earth=prop.omega_earth)
    events = extract_trajectory_key_events(res.t, r, radius_m=radius_m, omega_earth=prop.omega_earth)
    impact_data = None
    if reentry_flag and len(lat_deg) > 0:
        idx_reentry = -1
        if reentry_epoch is not None:
            diffs = [abs(ti - reentry_epoch) for ti in t]
            idx_reentry = int(np.argmin(diffs))
        impact_data = {
            "lat": float(lat_deg[idx_reentry]),
            "lon": float(lon_deg[idx_reentry]),
            "epoch": reentry_epoch,
            "dispersion_major_km": 110.0,
            "dispersion_minor_km": 28.0,
            "impact_vector": [float(r[idx_reentry, 0]), float(r[idx_reentry, 1]), float(r[idx_reentry, 2])],
        }
    out = {
        "t": t,
        "x": r[:, 0].tolist(),
        "y": r[:, 1].tolist(),
        "z": r[:, 2].tolist(),
        "vx": v[:, 0].tolist(),
        "vy": v[:, 1].tolist(),
        "vz": v[:, 2].tolist(),
        "alt_km": altitudes,
        "lat_deg": lat_deg.tolist(),
        "lon_deg": lon_deg.tolist(),
        "events": events,
        "reentry": reentry_flag,
        "reentry_time": reentry_epoch,
        "impact": impact_data,
    }
    if include_forces:
        out["forces"] = _compute_forces(prop, res.t, r, v)
    return out


@app.get("/")
def read_root():
    return FileResponse(
        os.path.join(BASE_DIR, "static", "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


@app.get("/graphs")
def read_graphs():
    return FileResponse(
        os.path.join(BASE_DIR, "static", "telemetry_graphs.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


@app.get("/vcam")
def read_vcam():
    return FileResponse(
        os.path.join(BASE_DIR, "static", "vcam_popout.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


@app.get("/docs")
def read_docs():
    return FileResponse(
        os.path.join(BASE_DIR, "static", "docs.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )



@app.get("/api/devices")
def api_get_devices():
    return {"devices": device_manager.list_devices()}

@app.post("/api/devices")
def api_add_device(payload: DevicePayload):
    dev = device_manager.add_or_update_device(payload.model_dump())
    return {"status": "success", "device": asdict(dev)}

@app.delete("/api/devices/{device_id}")
def api_delete_device(device_id: str):
    success = device_manager.remove_device(device_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Device {device_id} not found.")
    return {"status": "success", "message": f"Device {device_id} removed."}

@app.post("/api/devices/{device_id}/ping")
def api_ping_device(device_id: str):
    res = device_manager.ping_device(device_id)
    return res

@app.get("/api/hardware/telemetry")
def api_get_hardware_telemetry(limit: int = 60):
    return device_manager.get_telemetry_summary(limit=limit)


@app.get("/api/space_weather")
def get_space_weather(refresh: bool = False):
    """Return real-time or cached NOAA SWPC space weather indices (F10.7, Kp, dynamic density multiplier)."""
    sw = SpaceWeatherManager.get_instance().fetch_indices(force_refresh=refresh)
    return {
        "f107": sw.f107,
        "kp": sw.kp,
        "ap": sw.ap,
        "density_scale": sw.density_scale,
        "source": sw.source,
        "timestamp": sw.timestamp,
    }


@app.get("/api/aerodynamics")
def get_aerodynamics_profile(
    alt_km: float = 400.0,
    f107: float = 150.0,
    ap: float = 15.0,
    cd: float = 2.2,
    area_to_mass: float = 0.01,
    velocity_ms: float = 7600.0,
):
    """Evaluate thermospheric density and drag acceleration via nanodynamics functional aerodynamics model."""
    from nanodynamics.modules.space_weather_atmos.functions import compute_atmospheric_drag_density, is_storm_condition
    result = compute_atmospheric_drag_density(
        alt_km=alt_km,
        f107=f107,
        ap=ap,
        cd=cd,
        area_to_mass=area_to_mass,
        velocity_ms=velocity_ms,
    )
    result["is_storm"] = is_storm_condition(kp_value=min(9.0, ap / 5.0))
    result["alt_km"] = alt_km
    result["f107"] = f107
    result["ap"] = ap
    return result


class POIOrbitRequest(BaseModel):
    lat: float
    lon: float
    altitude_km: float = 550.0
    inclination_deg: float | None = None
    eccentricity: float = 0.001
    epoch_seconds: float = 0.0
    name: str = "POI-Sat"


class ISLNodePayload(BaseModel):
    id: str
    name: str
    x: float
    y: float
    z: float
    parent_body: str = "Earth"


class ISLRouteRequest(BaseModel):
    source_id: str
    target_id: str
    satellites: list[ISLNodePayload]
    max_range_km: float = 5000.0


@app.get("/api/pois")
def api_get_pois():
    """Return catalog of global major cities and key ground stations."""
    return {"pois": get_all_pois()}


@app.post("/api/orbit/poi")
def api_design_poi_orbit(req: POIOrbitRequest):
    """Compute orbital elements and initial state that flies over target lat/lon."""
    res = design_orbit_for_poi(
        lat_deg=req.lat,
        lon_deg=req.lon,
        target_alt_km=req.altitude_km,
        inclination_deg=req.inclination_deg,
        eccentricity=req.eccentricity,
        epoch_seconds=req.epoch_seconds,
    )
    return {
        "status": "success",
        "name": req.name,
        **res,
    }


class StationaryGEORequest(BaseModel):
    lat: float = 0.0
    lon: float
    epoch_seconds: float = 0.0
    name: str = "GEO-Stationary"


@app.post("/api/orbit/stationary_geo")
def api_stationary_geo(req: StationaryGEORequest):
    """Compute exact geostationary orbit placing satellite permanently stationary over city meridian."""
    r_geo_m = 42_164_140.0
    alt_km = (r_geo_m - R_EARTH) / 1000.0
    mu_earth = 3.986004418e14
    v_circ = math.sqrt(mu_earth / r_geo_m)
    period_s = 2.0 * math.pi * math.sqrt((r_geo_m**3) / mu_earth)

    # Calculate longitude offset from Greenwich sidereal rotation
    omega_earth = 7.292115e-5
    theta_gmst_deg = math.degrees((omega_earth * req.epoch_seconds) % (2.0 * math.pi))
    raan_deg = (req.lon + theta_gmst_deg) % 360.0
    raan_rad = math.radians(raan_deg)

    # Initial Cartesian state in ECI
    rx = r_geo_m * math.cos(raan_rad)
    ry = r_geo_m * math.sin(raan_rad)
    rz = 0.0
    vx = -v_circ * math.sin(raan_rad)
    vy = v_circ * math.cos(raan_rad)
    vz = 0.0

    return {
        "status": "success",
        "name": req.name,
        "orbital_elements": {
            "semi_major_axis_km": r_geo_m / 1000.0,
            "altitude_km": alt_km,
            "eccentricity": 0.0,
            "inclination_deg": 0.0,
            "raan_deg": raan_deg,
            "arg_perigee_deg": 0.0,
            "true_anomaly_deg": 0.0,
            "period_s": period_s,
        },
        "initial_state": {
            "r": [rx, ry, rz],
            "v": [vx, vy, vz],
            "v_mag_kms": v_circ / 1000.0,
        },
        "target_city_lon": req.lon,
    }


class InsertionBurnRequest(BaseModel):
    target_altitude_km: float = 550.0
    target_inclination_deg: float = 51.6
    parking_altitude_km: float = 180.0
    launch_lat_deg: float = 28.5


@app.post("/api/maneuvers/insertion_burn")
def api_insertion_burn(req: InsertionBurnRequest):
    """Calculate multi-impulse orbit insertion maneuvers (Parking -> Transfer -> Target Orbit)."""
    mu = 3.986004418e14
    r1 = R_EARTH + req.parking_altitude_km * 1000.0
    r2 = R_EARTH + req.target_altitude_km * 1000.0
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)

    a_trans = (r1 + r2) / 2.0
    v_t1 = math.sqrt(mu * (2.0 / r1 - 1.0 / a_trans))
    v_t2 = math.sqrt(mu * (2.0 / r2 - 1.0 / a_trans))

    dv1_injection = v_t1 - v1
    dv2_circularize = v2 - v_t2
    di_deg = abs(req.target_inclination_deg - req.launch_lat_deg)
    dv_plane_change = 2.0 * v2 * math.sin(math.radians(di_deg) / 2.0) if di_deg > 0.1 else 0.0
    total_dv = dv1_injection + dv2_circularize + dv_plane_change
    transfer_duration_s = math.pi * math.sqrt((a_trans**3) / mu)

    return {
        "status": "success",
        "parking_altitude_km": req.parking_altitude_km,
        "target_altitude_km": req.target_altitude_km,
        "dv1_injection_ms": round(dv1_injection, 2),
        "dv2_circularization_ms": round(dv2_circularize, 2),
        "dv_plane_change_ms": round(dv_plane_change, 2),
        "total_dv_ms": round(total_dv, 2),
        "transfer_duration_s": round(transfer_duration_s, 1),
        "transfer_duration_min": round(transfer_duration_s / 60.0, 1),
        "burn_nodes": [
            {"node": 1, "name": "Trans-Target Injection", "dv_ms": round(dv1_injection, 2), "altitude_km": req.parking_altitude_km, "frame": "Prograde"},
            {"node": 2, "name": "Orbit Circularization", "dv_ms": round(dv2_circularize, 2), "altitude_km": req.target_altitude_km, "frame": "Prograde"},
            {"node": 3, "name": "Inclination Trim", "dv_ms": round(dv_plane_change, 2), "altitude_km": req.target_altitude_km, "frame": "Normal"}
        ]
    }


MAP_CACHE_DIR = os.path.join(BASE_DIR, "static", "cache_maps")
os.makedirs(MAP_CACHE_DIR, exist_ok=True)


@app.get("/api/map/satellite")
def api_map_satellite(lat: float, lon: float, zoom: float = 0.15, size: int = 400):
    """Return high-resolution real-world aerial satellite imagery tile for coordinates."""
    min_lat, max_lat = max(-85.0, lat - zoom), min(85.0, lat + zoom)
    min_lon, max_lon = max(-180.0, lon - zoom), min(180.0, lon + zoom)
    cache_key = f"sat_{lat:.4f}_{lon:.4f}_{zoom:.3f}_{size}.jpg"
    cache_path = os.path.join(MAP_CACHE_DIR, cache_key)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return FileResponse(cache_path, media_type="image/jpeg")

    url = (
        f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export?"
        f"bbox={min_lon},{min_lat},{max_lon},{max_lat}&bboxSR=4326&imageSR=4326&size={size},{size}&format=jpg&f=image"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LEOSimulator/2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
        with open(cache_path, "wb") as f:
            f.write(data)
        return Response(content=data, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": f"Failed fetching satellite map: {e}"})


@app.get("/api/map/street")
def api_map_street(lat: float, lon: float, zoom: float = 0.15, size: int = 400):
    """Return high-resolution real-world street/road map tile for coordinates."""
    min_lat, max_lat = max(-85.0, lat - zoom), min(85.0, lat + zoom)
    min_lon, max_lon = max(-180.0, lon - zoom), min(180.0, lon + zoom)
    cache_key = f"street_{lat:.4f}_{lon:.4f}_{zoom:.3f}_{size}.jpg"
    cache_path = os.path.join(MAP_CACHE_DIR, cache_key)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return FileResponse(cache_path, media_type="image/jpeg")

    url = (
        f"https://services.arcgisonline.com/arcgis/rest/services/World_Street_Map/MapServer/export?"
        f"bbox={min_lon},{min_lat},{max_lon},{max_lat}&bboxSR=4326&imageSR=4326&size={size},{size}&format=jpg&f=image"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LEOSimulator/2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
        with open(cache_path, "wb") as f:
            f.write(data)
        return Response(content=data, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": f"Failed fetching street map: {e}"})


@app.get("/api/orbit/presets")
def api_get_orbit_presets():
    """Return specialized high-demand orbit presets (GEO, GSO, Molniya, Tundra, SSO, Polar)."""
    return {"presets": get_specialized_orbit_presets()}


@app.post("/api/isl/route")
def api_route_isl(req: ISLRouteRequest):
    """Compute shortest optical multi-hop laser path and latency across active satellites."""
    nodes = [
        SatelliteNode(
            id=s.id,
            name=s.name,
            position_m=np.array([s.x * 1000.0, s.y * 1000.0, s.z * 1000.0], dtype=np.float64),
            parent_body=s.parent_body,
        )
        for s in req.satellites
    ]
    mesh = ISLNetworkMesh(max_range_m=req.max_range_km * 1000.0)
    topo = mesh.build_topology(nodes)
    path = mesh.route_packet(req.source_id, req.target_id, nodes)
    return {
        "status": "success",
        "topology": {
            "node_count": topo["node_count"],
            "link_count": topo["link_count"],
            "links": topo["links"],
        },
        "route": {
            "connected": path is not None,
            "hops": path.hops if path else [],
            "hop_count": max(0, len(path.hops) - 1) if path else 0,
            "total_distance_km": path.total_distance_km if path else 0.0,
            "propagation_delay_ms": path.propagation_delay_ms if path else 0.0,
            "total_delay_ms": path.total_delay_ms if path else 0.0,
            "achievable_bitrate_mbps": path.achievable_bitrate_mbps if path else 0.0,
        } if path else None,
    }


@app.get("/bodies")
def get_bodies():
    """Return parameters and constraints for each central celestial body."""
    return {"bodies": BODIES}


@app.get("/satellites/live/groups")
def get_live_groups():
    """Return list of supported live satellite tracking groups."""
    return {"groups": LIVE_SATELLITE_GROUPS}


@app.get("/satellites/live/catalog")
def get_live_satellite_catalog(
    group: str | None = None,
    search: str | None = None,
    limit: int = 100,
    refresh: bool = False,
):
    """Return catalog of active satellites fetched from live CelesTrak feed / local cache."""
    sats = get_live_catalog(group=group, search=search, limit=limit, force_refresh=refresh)
    return {"count": len(sats), "satellites": sats}


@app.post("/satellites/live/track")
async def track_live_satellite(req: LiveTrackRequest):
    """Instantiate and simulate an active real-time satellite by NORAD ID."""
    sat_data = get_satellite_by_norad(req.norad_id)
    if not sat_data:
        raise HTTPException(status_code=404, detail=f"Satellite with NORAD ID {req.norad_id} not found in active catalog.")

    sat_name = (req.custom_name or f"{sat_data['name']} (NORAD {sat_data['norad_id']})")[:120]

    params = SatelliteParams(
        name=sat_name,
        parent_body="Earth",
        mass=float(sat_data.get("mass", 500.0)),
        drag_area=float(sat_data.get("drag_area", 2.0)),
        cd=float(sat_data.get("cd", 2.2)),
        altitude_km=float(sat_data["altitude_km"]),
        eccentricity=float(sat_data["eccentricity"]),
        inclination_deg=float(sat_data["inclination_deg"]),
        raan_deg=float(sat_data["raan_deg"] % 360.0),
        arg_periapsis_deg=float(sat_data.get("arg_periapsis_deg", 0.0) % 360.0),
        true_anomaly_deg=float(sat_data.get("true_anomaly_deg", 0.0) % 360.0),
        atmosphere_type="piecewise",
        density_scale=1.0,
        include_j2=True,
        include_drag=True,
        include_moon=False,
        color=sat_data.get("color", "#00ffcc"),
        icon=sat_data.get("icon", "satellite"),
        propagation_mode=req.propagation_mode,
        t_start=req.t_start,
    )

    sim_result = await simulate_satellite(params)
    sim_result["norad_id"] = sat_data["norad_id"]
    sim_result["epoch"] = sat_data.get("epoch", "")
    sim_result["group"] = sat_data.get("group", "cubesat")
    return sim_result


@app.post("/simulate")
async def simulate_satellite(params: SatelliteParams):
    """Bootstrap: propagates the first chunk and returns initial state + trajectory."""
    try:
        body = BODIES.get(params.parent_body, BODIES["Earth"])
        mode_str = params.propagation_mode.lower()
        is_pinn = mode_str.startswith("pinn")
        pinn_submode = "pinn_local" if mode_str == "pinn" else mode_str

        if is_pinn:
            prop = PINNPropagator(
                satellite=Satellite(
                    name=params.name,
                    mass=params.mass,
                    drag_area=params.drag_area,
                    cd=params.cd,
                ),
                mu=body["mu"],
                r_earth=body["radius_m"],
                omega_earth=body["omega"],
                min_altitude_reentry=50_000.0,
                mode=pinn_submode,
                device_id=params.hardware_device_id or "esp32-node-1",
            )
            if pinn_submode == "pinn_esp32":
                prop.attach_hardware_agent()
                if active_hil_worker is not None:
                    prop.network_ws = active_hil_worker
        else:
            prop = _build_propagator(params)
        
        a_m = body["radius_m"] + params.altitude_km * 1000.0
        orbit = OrbitalElements(
            a=a_m,
            e=params.eccentricity,
            i=np.radians(params.inclination_deg),
            raan=np.radians(params.raan_deg),
            arg_pe=np.radians(params.arg_periapsis_deg),
            nu=np.radians(params.true_anomaly_deg),
        )

        chunk_duration = body["default_chunk_duration"]
        dt_eval = 60.0 if is_pinn else body["dt_eval"]

        if is_pinn and hasattr(prop, 'propagate_async') and prop.network_ws is not None:
            res = await prop.propagate_async(orbit=orbit, duration_seconds=chunk_duration, t_start=params.t_start, dt_eval=dt_eval)
        else:
            res = await asyncio.to_thread(
                prop.propagate, orbit, chunk_duration, params.t_start, dt_eval
            )
        chunk = _format_chunk(res, prop, body["radius_m"], include_forces=True)

        # Return final Cartesian state for /stream continuity
        last_state = res.r[-1].tolist() + res.v[-1].tolist()
        period_s = 2.0 * np.pi * np.sqrt((a_m**3) / body["mu"])

        if not is_pinn:
            _log_sim_for_ml(params, dt_eval, chunk)

        return {
            "name": params.name,
            "parent_body": params.parent_body,
            "params": params.model_dump(),
            "period_s": float(period_s),
            "trajectory": chunk,
            "last_state": last_state,
            "last_t": float(res.t[-1]),
            "reentry": bool(res.reentry_detected),
            "reentry_time": float(res.reentry_time) if res.reentry_time is not None else None,
            "engine": params.propagation_mode.lower(),
            "telemetry": getattr(res, "telemetry", None),
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream_chunk(req: StreamRequest):
    """Rolling propagation: given last known state, propagate next chunk."""
    try:
        body = BODIES.get(req.params.parent_body, BODIES["Earth"])
        mode_str = getattr(req.params, "propagation_mode", "rk45").lower()
        is_pinn = mode_str.startswith("pinn")
        pinn_submode = "pinn_local" if mode_str == "pinn" else mode_str

        if is_pinn:
            prop = PINNPropagator(
                satellite=Satellite(
                    name=req.params.name,
                    mass=req.params.mass,
                    drag_area=req.params.drag_area,
                    cd=req.params.cd,
                ),
                mu=body["mu"],
                r_earth=body["radius_m"],
                omega_earth=body["omega"],
                min_altitude_reentry=50_000.0,
                mode=pinn_submode,
                device_id=getattr(req.params, "hardware_device_id", "esp32-node-1"),
            )
            if pinn_submode == "pinn_esp32":
                prop.attach_hardware_agent()
                if active_hil_worker is not None:
                    prop.network_ws = active_hil_worker
        else:
            prop = _build_propagator(req.params)
        dt_eval = 60.0 if is_pinn else (req.dt_eval if req.dt_eval else body["dt_eval"])
        res = await _propagate_chunk_async( prop, req.state, req.t_start, req.chunk_duration, dt_eval
        )
        chunk = _format_chunk(res, prop, body["radius_m"], include_forces=True)
        last_state = res.r[-1].tolist() + res.v[-1].tolist()

        return {
            "trajectory": chunk,
            "last_state": last_state,
            "last_t": float(res.t[-1]),
            "reentry": bool(res.reentry_detected),
            "reentry_time": float(res.reentry_time) if res.reentry_time is not None else None,
            "engine": getattr(req.params, "propagation_mode", "rk45").lower(),
            "telemetry": getattr(res, "telemetry", None),
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/burn/preview")
async def preview_burn(req: BurnRequest, duration_s: float = 7200.0):
    """Fast preview of perturbed/post-burn trajectory arc without committing state."""
    try:
        body = BODIES.get(req.params.parent_body, BODIES["Earth"])
        r = np.array(req.current_state[:3], dtype=np.float64)
        v = np.array(req.current_state[3:], dtype=np.float64)

        r_norm = np.linalg.norm(r)
        if r_norm <= 0.0:
            raise ValueError("Position vector norm must be positive.")
        u_v, u_n, u_c = compute_vnc_basis(r, v)
        dv_vec = req.dv_prograde * u_v + req.dv_normal * u_n + req.dv_radial * u_c
        dv_mag = float(np.linalg.norm(dv_vec))
        finite_arc = None
        burn_duration = 0.0
        propellant_used = 0.0

        if req.is_finite and dv_mag > 1e-4:
            f_res = propagate_finite_burn(
                r, v, mass_start=req.params.mass,
                dv_vnc=(req.dv_prograde, req.dv_normal, req.dv_radial),
                t_start=req.t_burn, thrust_n=req.thrust_n, isp_s=req.isp_s,
                mu=body["mu"],
            )
            burn_duration = float(f_res["burn_duration"])
            propellant_used = float(f_res["propellant_used"])
            finite_arc = {
                "t": f_res["t"].tolist(),
                "x": f_res["r"][:, 0].tolist(),
                "y": f_res["r"][:, 1].tolist(),
                "z": f_res["r"][:, 2].tolist(),
            }
            r = f_res["r"][-1]
            v_new = f_res["v"][-1]
            req.t_burn = float(f_res["t"][-1])
            req.params.mass = float(f_res["mass"][-1])
        else:
            v_new = v + dv_vec

        new_state = r.tolist() + v_new.tolist()

        prop = _build_propagator(req.params)
        
        # Adaptive preview duration for cislunar / high-energy orbits
        v_mag_sq = np.dot(v_new, v_new)
        spec_energy = 0.5 * v_mag_sq - (body["mu"] / r_norm)
        if spec_energy < -1e-5:
            a_post = -body["mu"] / (2.0 * spec_energy)
            period_post = 2.0 * np.pi * np.sqrt((a_post**3) / body["mu"])
            preview_dur = min(max(duration_s, period_post * 0.6), 345600.0)
        else:
            # Parabolic / hyperbolic cislunar trajectory
            preview_dur = 259200.0  # 3 days
        
        dt_eval = max(20.0, min(300.0, preview_dur / 300.0))

        res = await _propagate_chunk_async(
            prop, new_state, req.t_burn, preview_dur, dt_eval
        )
        chunk = _format_chunk(res, prop, body["radius_m"], include_forces=False)

        return {
            "status": "success",
            "new_state": new_state,
            "trajectory": chunk,
            "burn_profile": {
                "is_finite": req.is_finite,
                "burn_duration_s": burn_duration,
                "propellant_used_kg": propellant_used,
                "finite_arc": finite_arc,
            },
            "reentry": bool(res.reentry_detected),
            "reentry_time": float(res.reentry_time) if res.reentry_time is not None else None,
            "events": chunk.get("events"),
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/burn")
async def execute_burn(req: BurnRequest):
    """Apply an instantaneous Delta-V in true velocity-aligned VNC frame and propagate forward."""
    try:
        body = BODIES.get(req.params.parent_body, BODIES["Earth"])
        r = np.array(req.current_state[:3], dtype=np.float64)
        v = np.array(req.current_state[3:], dtype=np.float64)

        r_norm = np.linalg.norm(r)
        if r_norm <= 0.0:
            raise ValueError("Position vector norm must be positive.")

        # Compute true velocity-aligned VNC basis vectors
        u_v, u_n, u_c = compute_vnc_basis(r, v)
        dv_vec = req.dv_prograde * u_v + req.dv_normal * u_n + req.dv_radial * u_c
        dv_mag = float(np.linalg.norm(dv_vec))
        finite_arc = None
        burn_duration = 0.0
        propellant_used = 0.0

        if req.is_finite and dv_mag > 1e-4:
            f_res = propagate_finite_burn(
                r, v, mass_start=req.params.mass,
                dv_vnc=(req.dv_prograde, req.dv_normal, req.dv_radial),
                t_start=req.t_burn, thrust_n=req.thrust_n, isp_s=req.isp_s,
                mu=body["mu"],
            )
            burn_duration = float(f_res["burn_duration"])
            propellant_used = float(f_res["propellant_used"])
            finite_arc = {
                "t": f_res["t"].tolist(),
                "x": f_res["r"][:, 0].tolist(),
                "y": f_res["r"][:, 1].tolist(),
                "z": f_res["r"][:, 2].tolist(),
            }
            r = f_res["r"][-1]
            v_new = f_res["v"][-1]
            req.t_burn = float(f_res["t"][-1])
            req.params.mass = float(f_res["mass"][-1])
        else:
            v_new = v + dv_vec

        new_state = r.tolist() + v_new.tolist()

        # Update osculating orbital elements in params so future chunk streaming preserves perturbed orbit
        new_kep = rv_to_keplerian(r, v_new, mu=body["mu"])
        new_params = req.params.model_copy()
        new_params.allow_decay = True
        new_params.altitude_km = float(max(10.0, (r_norm - body["radius_m"]) / 1000.0))
        new_params.eccentricity = float(min(0.999, max(0.0, new_kep["e"])))
        new_params.inclination_deg = float(new_kep["inc_deg"])
        new_params.raan_deg = float(new_kep["raan_deg"])

        is_pinn = getattr(req.params, "propagation_mode", "rk45").lower().startswith("pinn")
        if is_pinn:
            prop = PINNPropagator(
                satellite=Satellite(name=req.params.name, mass=req.params.mass, drag_area=req.params.drag_area, cd=req.params.cd),
                mu=body["mu"], r_earth=body["radius_m"], omega_earth=body["omega"], min_altitude_reentry=50_000.0,
            )
            prop.attach_hardware_agent()
            if active_hil_worker is not None:
                prop.network_ws = active_hil_worker
        else:
            prop = _build_propagator(new_params)

        # Adaptive cislunar transfer chunk duration
        v_mag_sq = np.dot(v_new, v_new)
        spec_energy = 0.5 * v_mag_sq - (body["mu"] / r_norm)
        if spec_energy < -1e-5:
            a_post = -body["mu"] / (2.0 * spec_energy)
            period_post = 2.0 * np.pi * np.sqrt((a_post**3) / body["mu"])
            chunk_duration = min(max(body["default_chunk_duration"], period_post * 0.75), 432000.0)
        else:
            chunk_duration = 345600.0  # 4 days
        
        dt_eval = max(body["dt_eval"], min(300.0, chunk_duration / 400.0))

        res = await _propagate_chunk_async(
            prop, new_state, req.t_burn, chunk_duration, dt_eval
        )
        chunk = _format_chunk(res, prop, body["radius_m"], include_forces=True)
        last_state = res.r[-1].tolist() + res.v[-1].tolist()

        return {
            "status": "success",
            "new_state": new_state,
            "new_params": new_params.model_dump(),
            "trajectory": chunk,
            "last_state": last_state,
            "last_t": float(res.t[-1]),
            "burn_profile": {
                "is_finite": req.is_finite,
                "burn_duration_s": burn_duration,
                "propellant_used_kg": propellant_used,
                "finite_arc": finite_arc,
            },
            "events": chunk.get("events"),
            "reentry": bool(res.reentry_detected),
            "reentry_time": float(res.reentry_time) if res.reentry_time is not None else None,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collision/resolve")
async def resolve_collision(req: CollisionRequest):
    """Compute momentum-exchange impulse and propagate perturbed orbits for colliding satellites."""
    try:
        body1 = BODIES.get(req.sat1_params.parent_body, BODIES["Earth"])
        body2 = BODIES.get(req.sat2_params.parent_body, BODIES["Earth"])
        if req.sat1_params.parent_body != req.sat2_params.parent_body:
            raise ValueError("Colliding satellites must share the same parent celestial body.")

        r1 = np.array(req.sat1_state[:3], dtype=np.float64)
        v1 = np.array(req.sat1_state[3:], dtype=np.float64)
        r2 = np.array(req.sat2_state[:3], dtype=np.float64)
        v2 = np.array(req.sat2_state[3:], dtype=np.float64)

        col_res = compute_orbital_collision_impulse(
            m1=req.sat1_params.mass,
            r1=r1,
            v1=v1,
            m2=req.sat2_params.mass,
            r2=r2,
            v2=v2,
            restitution=req.restitution,
        )

        new_state1 = r1.tolist() + col_res.v1_post.tolist()
        new_state2 = r2.tolist() + col_res.v2_post.tolist()

        # Partition colliding satellites into surviving core and daughter debris fragments
        core1_spec, daughters1 = generate_collision_debris_fragments(
            sat_name=req.sat1_params.name,
            parent_mass=req.sat1_params.mass,
            parent_drag_area=req.sat1_params.drag_area,
            parent_cd=req.sat1_params.cd,
            col_pos=r1,
            post_vel=col_res.v1_post,
            relative_speed=col_res.relative_speed,
            num_fragments=2,
            core_mass_fraction=0.50,
            parent_body=req.sat1_params.parent_body,
            propagation_mode=getattr(req.sat1_params, "propagation_mode", "rk45"),
        )
        core2_spec, daughters2 = generate_collision_debris_fragments(
            sat_name=req.sat2_params.name,
            parent_mass=req.sat2_params.mass,
            parent_drag_area=req.sat2_params.drag_area,
            parent_cd=req.sat2_params.cd,
            col_pos=r2,
            post_vel=col_res.v2_post,
            relative_speed=col_res.relative_speed,
            num_fragments=2,
            core_mass_fraction=0.50,
            parent_body=req.sat2_params.parent_body,
            propagation_mode=getattr(req.sat2_params, "propagation_mode", "rk45"),
        )

        new_kep1 = rv_to_keplerian(r1, col_res.v1_post, mu=body1["mu"])
        new_params1 = req.sat1_params.model_copy()
        new_params1.allow_decay = True
        new_params1.mass = core1_spec["mass"]
        new_params1.drag_area = core1_spec["drag_area"]
        new_params1.altitude_km = float(max(10.0, (np.linalg.norm(r1) - body1["radius_m"]) / 1000.0))
        new_params1.eccentricity = float(min(0.999, max(0.0, new_kep1["e"])))
        new_params1.inclination_deg = float(new_kep1["inc_deg"])
        new_params1.raan_deg = float(new_kep1["raan_deg"])

        new_kep2 = rv_to_keplerian(r2, col_res.v2_post, mu=body2["mu"])
        new_params2 = req.sat2_params.model_copy()
        new_params2.allow_decay = True
        new_params2.mass = core2_spec["mass"]
        new_params2.drag_area = core2_spec["drag_area"]
        new_params2.altitude_km = float(max(10.0, (np.linalg.norm(r2) - body2["radius_m"]) / 1000.0))
        new_params2.eccentricity = float(min(0.999, max(0.0, new_kep2["e"])))
        new_params2.inclination_deg = float(new_kep2["inc_deg"])
        new_params2.raan_deg = float(new_kep2["raan_deg"])

        is_pinn1 = getattr(req.sat1_params, "propagation_mode", "rk45").lower() == "pinn"
        is_pinn2 = getattr(req.sat2_params, "propagation_mode", "rk45").lower() == "pinn"

        if is_pinn1:
            prop1 = PINNPropagator(
                satellite=Satellite(name=new_params1.name, mass=new_params1.mass, drag_area=new_params1.drag_area, cd=new_params1.cd),
                mu=body1["mu"], r_earth=body1["radius_m"], omega_earth=body1["omega"], min_altitude_reentry=50_000.0,
            )
            prop1.attach_hardware_agent()
            if active_hil_worker is not None:
                prop1.network_ws = active_hil_worker
        else:
            prop1 = _build_propagator(new_params1)

        if is_pinn2:
            prop2 = PINNPropagator(
                satellite=Satellite(name=new_params2.name, mass=new_params2.mass, drag_area=new_params2.drag_area, cd=new_params2.cd),
                mu=body2["mu"], r_earth=body2["radius_m"], omega_earth=body2["omega"], min_altitude_reentry=50_000.0,
            )
            prop2.attach_hardware_agent()
            if active_hil_worker is not None:
                prop2.network_ws = active_hil_worker
        else:
            prop2 = _build_propagator(new_params2)

        daughter_list = daughters1 + daughters2
        daughter_props = []
        for d in daughter_list:
            r_d = np.array(d["initial_state"][:3], dtype=np.float64)
            v_d = np.array(d["initial_state"][3:], dtype=np.float64)
            kep_d = rv_to_keplerian(r_d, v_d, mu=body1["mu"])
            d_params = SatelliteParams(
                name=d["name"],
                parent_body=d["parent_body"],
                altitude_km=float(max(10.0, (np.linalg.norm(r_d) - body1["radius_m"]) / 1000.0)),
                eccentricity=float(min(0.999, max(0.0, kep_d["e"]))),
                inclination_deg=float(min(180.0, max(0.0, kep_d["inc_deg"]))),
                raan_deg=float(kep_d["raan_deg"] % 360.0),
                mass=d["mass"],
                drag_area=d["drag_area"],
                cd=d["cd"],
                propagation_mode="rk45",
                allow_decay=True,
            )
            d["params"] = d_params.model_dump()
            d_prop = _build_propagator(d_params)
            daughter_props.append((d, d_prop))

        chunk_dur = min(req.post_collision_duration, body1["default_chunk_duration"])
        dt_eval = req.dt_eval or body1["dt_eval"]

        tasks = [
            asyncio.to_thread(_propagate_chunk, prop1, new_state1, req.t_collision, chunk_dur, dt_eval),
            asyncio.to_thread(_propagate_chunk, prop2, new_state2, req.t_collision, chunk_dur, dt_eval),
        ]
        for d, d_prop in daughter_props:
            tasks.append(asyncio.to_thread(_propagate_chunk, d_prop, d["initial_state"], req.t_collision, chunk_dur, dt_eval))

        all_res = await asyncio.gather(*tasks)
        res1 = all_res[0]
        res2 = all_res[1]
        daughter_results = all_res[2:]

        chunk1 = _format_chunk(res1, prop1, body1["radius_m"], include_forces=True)
        chunk2 = _format_chunk(res2, prop2, body2["radius_m"], include_forces=True)

        daughters_out = []
        for (d, d_prop), d_res in zip(daughter_props, daughter_results):
            d_chunk = _format_chunk(d_res, d_prop, body1["radius_m"], include_forces=True)
            d_dict = dict(d)
            d_dict["trajectory"] = d_chunk
            d_dict["last_state"] = d_res.r[-1].tolist() + d_res.v[-1].tolist()
            d_dict["last_t"] = float(d_res.t[-1])
            d_dict["events"] = d_chunk.get("events")
            d_dict["reentry"] = bool(d_res.reentry_detected)
            d_dict["reentry_time"] = float(d_res.reentry_time) if d_res.reentry_time is not None else None
            daughters_out.append(d_dict)

        return {
            "status": "success",
            "t_collision": float(req.t_collision),
            "separation_m": float(col_res.separation_distance),
            "relative_speed_m_s": float(col_res.relative_speed),
            "impulse_ns": float(col_res.impulse_mag),
            "sat1": {
                "name": req.sat1_params.name,
                "dv_mag_m_s": float(col_res.dv1_mag),
                "new_state": new_state1,
                "new_params": new_params1.model_dump(),
                "trajectory": chunk1,
                "last_state": res1.r[-1].tolist() + res1.v[-1].tolist(),
                "last_t": float(res1.t[-1]),
                "events": chunk1.get("events"),
                "reentry": bool(res1.reentry_detected),
                "reentry_time": float(res1.reentry_time) if res1.reentry_time is not None else None,
                "engine": "pinn" if is_pinn1 else "rk45",
            },
            "sat2": {
                "name": req.sat2_params.name,
                "dv_mag_m_s": float(col_res.dv2_mag),
                "new_state": new_state2,
                "new_params": new_params2.model_dump(),
                "trajectory": chunk2,
                "last_state": res2.r[-1].tolist() + res2.v[-1].tolist(),
                "last_t": float(res2.t[-1]),
                "events": chunk2.get("events"),
                "reentry": bool(res2.reentry_detected),
                "reentry_time": float(res2.reentry_time) if res2.reentry_time is not None else None,
                "engine": "pinn" if is_pinn2 else "rk45",
            },
            "daughters": daughters_out,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/experiment/run")
async def api_run_experiment(req: ExperimentRequest):
    """Run server-side parametric sweep and record clean CSV results."""
    try:
        base_dict = req.base_params.model_dump()
        max_duration_sec = req.max_duration_days * 86400.0

        result = await asyncio.to_thread(
            run_parametric_sweep,
            param_name=req.param_name,
            values=req.values,
            base_params=base_dict,
            atmosphere_type=req.atmosphere_type,
            density_scale=req.density_scale,
            max_duration_seconds=max_duration_sec,
            quiet=req.quiet,
            output_dir=EXPERIMENTS_DIR,
        )

        return {
            "status": "success",
            "experiment_id": result.experiment_id,
            "param_name": result.param_name,
            "timestamp": result.timestamp,
            "csv_filename": f"{result.experiment_id}.csv",
            "results": [asdict(r) for r in result.results],
            "trajectories": result.trajectories if not req.quiet else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiment/download/{filename}")
def api_download_experiment_csv(filename: str):
    """Download a recorded experiment CSV dataset."""
    safe_name = sanitize_filename(filename.replace(".csv", ""))
    filepath = os.path.join(EXPERIMENTS_DIR, f"{safe_name}.csv")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Experiment CSV file not found")
    return FileResponse(filepath, media_type="text/csv", filename=f"{safe_name}.csv")


@app.get("/experiment/list")
def api_list_experiments():
    """List all recorded experiment CSV files."""
    files = [f for f in os.listdir(EXPERIMENTS_DIR) if f.endswith(".csv")]
    files.sort(reverse=True)
    return {"experiments": files}


@app.post("/export/analytical")
async def export_analytical_data(req: AnalyticalExportRequest):
    """Generate high-dimensional 24-variable physical analytical dataset for PINN training and inference."""
    try:
        body = BODIES.get(req.params.parent_body, BODIES["Earth"])
        is_pinn = getattr(req.params, "propagation_mode", "rk45").lower().startswith("pinn")
        if is_pinn:
            prop = PINNPropagator(
                satellite=Satellite(name=req.params.name, mass=req.params.mass, drag_area=req.params.drag_area, cd=req.params.cd),
                mu=body["mu"], r_earth=body["radius_m"], omega_earth=body["omega"], min_altitude_reentry=50_000.0,
            )
            prop.attach_hardware_agent()
            if active_hil_worker is not None:
                prop.network_ws = active_hil_worker
        else:
            prop = _build_propagator(req.params)

        if req.initial_state is not None:
            init_state = np.asarray(req.initial_state, dtype=np.float64)
        else:
            a_m = body["radius_m"] + req.params.altitude_km * 1000.0
            orbit = OrbitalElements(
                a=a_m,
                e=req.params.eccentricity,
                i=math.radians(req.params.inclination_deg),
                raan=math.radians(req.params.raan_deg),
                arg_pe=math.radians(req.params.arg_periapsis_deg),
                nu=math.radians(req.params.true_anomaly_deg),
            )
            r0, v0 = coe_to_rv(orbit, mu=body["mu"])
            init_state = np.concatenate([r0, v0])

        res = await _propagate_chunk_async(prop, init_state, req.t_start, req.duration_seconds, req.dt_eval)
        t_arr = res.t
        r_arr = res.r
        v_arr = res.v
        N = len(t_arr)

        lat_deg, lon_deg, alt_m = eci_to_geodetic(r_arr, t_arr, radius_m=body["radius_m"], omega_earth=prop.omega_earth)
        alt_km = alt_m / 1000.0

        forces = _compute_forces(prop, t_arr, r_arr, v_arr)

        headers = [
            "t_sec", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s", "alt_km",
            "sma_km", "eccentricity", "inclination_deg", "raan_deg", "arg_perigee_deg", "true_anomaly_deg",
            "specific_energy_mj_kg", "ang_momentum_km2_s", "j2_accel_m_s2", "aero_drag_accel_m_s2",
            "tec_tecu", "ionospheric_delay_ns", "in_saa", "seu_rate_per_day",
            "doppler_shift_hz", "fspl_db"
        ]

        lines = [",".join(headers)]
        sw_idx = SpaceWeatherManager.get_instance().fetch_indices()
        for k in range(N):
            rk = r_arr[k]
            vk = v_arr[k]
            tk = float(t_arr[k])
            lk = float(lat_deg[k])
            ok = float(lon_deg[k])
            ak = float(alt_km[k])

            kep = rv_to_keplerian(rk, vk, mu=body["mu"])
            env = compute_space_environment(lk, ok, ak, f107=sw_idx.f107)
            rf = compute_rf_link_budget(rk, vk)

            j2_vec = forces["j2"][k]
            drag_vec = forces["drag"][k]
            j2_mag = math.sqrt(j2_vec[0]**2 + j2_vec[1]**2 + j2_vec[2]**2)
            drag_mag = math.sqrt(drag_vec[0]**2 + drag_vec[1]**2 + drag_vec[2]**2)

            row = [
                f"{tk:.2f}",
                f"{rk[0]/1000.0:.4f}",
                f"{rk[1]/1000.0:.4f}",
                f"{rk[2]/1000.0:.4f}",
                f"{vk[0]/1000.0:.5f}",
                f"{vk[1]/1000.0:.5f}",
                f"{vk[2]/1000.0:.5f}",
                f"{ak:.3f}",
                f"{kep['a']/1000.0:.3f}",
                f"{kep['e']:.6f}",
                f"{kep['inc_deg']:.4f}",
                f"{kep['raan_deg']:.4f}",
                f"{kep['arg_p_deg']:.4f}",
                f"{kep['nu_deg']:.4f}",
                f"{kep['specific_energy']/1e6:.6f}",
                f"{kep['h_mag']/1e6:.4f}",
                f"{j2_mag:.6e}",
                f"{drag_mag:.6e}",
                f"{env['tec_tecu']:.3f}",
                f"{env['iono_delay_ns']:.3f}",
                f"{int(env['in_saa'])}",
                f"{env['seu_rate_per_hr']*24.0:.4f}",
                f"{rf['doppler_shift_hz']:.2f}",
                f"{rf['fspl_db']:.2f}",
            ]
            lines.append(",".join(row))

        csv_content = "\n".join(lines)
        safe_sat_name = sanitize_filename(req.params.name.replace(" ", "_"))
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={safe_sat_name}_analytical_24var.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save")
def save_simulation(payload: SavePayload):
    safe_name = sanitize_filename(payload.filename)
    try:
        # Try Firestore first
        try:
            from firebase_admin import firestore
            db = firestore.client()
            db.collection("scenarios").document(safe_name).set({"satellites": payload.satellites})
            return {"status": "success", "message": f"Saved {safe_name} to Firestore"}
        except Exception as e:
            # Fallback to local
            filepath = os.path.join(SAVED_SIMS_DIR, f"{safe_name}.json")
            with open(filepath, "w") as f:
                json.dump(payload.satellites, f)
            return {"status": "success", "message": f"Saved {safe_name} locally"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/load/{filename}")
def load_simulation(filename: str):
    safe_name = sanitize_filename(filename)
    try:
        # Try Firestore first
        try:
            from firebase_admin import firestore
            db = firestore.client()
            doc = db.collection("scenarios").document(safe_name).get()
            if doc.exists:
                return doc.to_dict()["satellites"]
        except Exception:
            pass
            
        # Fallback to local
        filepath = os.path.join(SAVED_SIMS_DIR, f"{safe_name}.json")
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Scenario not found in Firestore or local")
        with open(filepath, "r") as f:
            data = json.load(f)
        return {"satellites": data}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scenarios/presets")
def get_scenario_presets():
    """Returns curated scenarios including autonomous conjunction avoidance and rapid re-entry."""
    return {
        "presets": [
            {
                "id": "conjunction_pinn_isl",
                "name": "PINN Autonomous Conjunction Avoidance (ESP32)",
                "description": "Two opposing satellites running PINN propagation on ESP32 Node 1 & 2. Range sensor triggers autonomous evasive burn when distance < 5 km.",
                "satellites": [
                    {
                        "name": "PINN-ALPHA (Node 1)",
                        "parent_body": "Earth",
                        "propagation_mode": "pinn_esp32",
                        "altitude_km": 490.0,
                        "eccentricity": 0.0005,
                        "inclination_deg": 51.6,
                        "raan_deg": 0.0,
                        "arg_pe_deg": 0.0,
                        "true_anomaly_deg": 0.0,
                        "mass": 12.0,
                        "drag_area": 0.04,
                        "cd": 2.2,
                        "device_id": "esp32-node-1",
                        "color": "#00ffcc",
                        "autonomous_avoidance": True,
                        "avoidance_range_km": 5.0,
                        "avoidance_dv": 4.0,
                    },
                    {
                        "name": "PINN-BETA (Node 2)",
                        "parent_body": "Earth",
                        "propagation_mode": "pinn_esp32",
                        "altitude_km": 490.05,
                        "eccentricity": 0.0005,
                        "inclination_deg": 51.65,
                        "raan_deg": 0.12,
                        "arg_pe_deg": 0.0,
                        "true_anomaly_deg": 180.0,
                        "mass": 14.0,
                        "drag_area": 0.05,
                        "cd": 2.2,
                        "device_id": "esp32-node-2",
                        "color": "#ff3366",
                        "autonomous_avoidance": True,
                        "avoidance_range_km": 5.0,
                        "avoidance_dv": -4.0,
                    }
                ]
            },
            {
                "id": "dense_atmosphere_deorbit",
                "name": "Atmospheric Reentry & Impact Footprint Analysis",
                "description": "Low-altitude satellite encountering exponential atmospheric decay with predicted landing crater and dispersion ellipse.",
                "satellites": [
                    {
                        "name": "DEORBIT-PROBE-01",
                        "parent_body": "Earth",
                        "propagation_mode": "rk45",
                        "altitude_km": 140.0,
                        "eccentricity": 0.002,
                        "inclination_deg": 45.0,
                        "raan_deg": 25.0,
                        "arg_pe_deg": 0.0,
                        "true_anomaly_deg": 0.0,
                        "mass": 45.0,
                        "drag_area": 0.85,
                        "cd": 2.4,
                        "allow_decay": True,
                        "color": "#ff9900",
                    }
                ]
            },
            {
                "id": "starlink_dense_train",
                "name": "Starlink LEO Constellation Train (6 Sats)",
                "description": "Orbital plane injection train with inter-satellite laser links (ISL) active.",
                "satellites": [
                    {
                        "name": f"STARLINK-V2-{i+1}",
                        "parent_body": "Earth",
                        "propagation_mode": "rk45",
                        "altitude_km": 550.0,
                        "eccentricity": 0.0002,
                        "inclination_deg": 53.2,
                        "raan_deg": 40.0,
                        "arg_pe_deg": 0.0,
                        "true_anomaly_deg": float(i * 1.5),
                        "mass": 260.0,
                        "drag_area": 2.2,
                        "cd": 2.2,
                        "color": "#38bdf8",
                    }
                    for i in range(6)
                ]
            },
            {
                "id": "geo_stationkeeping",
                "name": "Geostationary High-Throughput Satellite (GEO Commsat)",
                "description": "Equatorial 35,786 km geostationary orbit with triaxial lunar perturbations.",
                "satellites": [
                    {
                        "name": "ASTRA-GEO-HTS",
                        "parent_body": "Earth",
                        "propagation_mode": "rk45",
                        "altitude_km": 35786.0,
                        "eccentricity": 0.0001,
                        "inclination_deg": 0.05,
                        "raan_deg": 0.0,
                        "arg_pe_deg": 0.0,
                        "true_anomaly_deg": 65.0,
                        "mass": 3200.0,
                        "drag_area": 18.0,
                        "cd": 2.2,
                        "color": "#eab308",
                    }
                ],
            },
            {
                "id": "starlink_mega_shell",
                "name": "Starlink Mega-Constellation Shell (24 Sats, 3 Planes)",
                "description": "Walker Delta 24/3/1 constellation shell at 550 km altitude, 53° inclination, with inter-satellite optical laser routing mesh.",
                "satellites": [
                    {
                        "name": f"STARLINK-P{plane+1}-S{s+1}",
                        "parent_body": "Earth",
                        "propagation_mode": "rk45",
                        "altitude_km": 550.0,
                        "eccentricity": 0.0003,
                        "inclination_deg": 53.0,
                        "raan_deg": float(plane * 120.0),
                        "arg_pe_deg": 0.0,
                        "true_anomaly_deg": float((s * 45.0 + plane * 15.0) % 360.0),
                        "mass": 280.0,
                        "drag_area": 2.4,
                        "cd": 2.2,
                        "color": ["#38bdf8", "#00ffcc", "#818cf8"][plane],
                    }
                    for plane in range(3)
                    for s in range(8)
                ]
            },
            {
                "id": "kessler_catastrophic_collision",
                "name": "Catastrophic LEO Kessler Collision Disaster (14.2 km/s)",
                "description": "Counter-rotating orbits resulting in direct high-speed hypervelocity conjunction (Cosmos-Iridium scenario) triggering debris dispersal.",
                "satellites": [
                    {
                        "name": "COSMOS-2251-DERELICT",
                        "parent_body": "Earth",
                        "propagation_mode": "rk45",
                        "altitude_km": 776.0,
                        "eccentricity": 0.0012,
                        "inclination_deg": 74.0,
                        "raan_deg": 10.0,
                        "arg_pe_deg": 0.0,
                        "true_anomaly_deg": 45.0,
                        "mass": 900.0,
                        "drag_area": 4.5,
                        "cd": 2.2,
                        "color": "#ef4444",
                    },
                    {
                        "name": "IRIDIUM-33-COMMERCIAL",
                        "parent_body": "Earth",
                        "propagation_mode": "rk45",
                        "altitude_km": 776.05,
                        "eccentricity": 0.0012,
                        "inclination_deg": 86.4,
                        "raan_deg": 188.0,
                        "arg_pe_deg": 0.0,
                        "true_anomaly_deg": 45.02,
                        "mass": 560.0,
                        "drag_area": 3.8,
                        "cd": 2.2,
                        "color": "#f59e0b",
                    }
                ]
            },
            {
                "id": "pinn_autonomous_evasion_burn",
                "name": "PINN Real-Time Autonomous Boot Burn Avoidance",
                "description": "Onboard PINN detects conjunction in <2ms and fires an emergency +15 m/s impulsive boot burn raising periapsis to avert disaster.",
                "satellites": [
                    {
                        "name": "TARGET-DEBRIS-PIECE",
                        "parent_body": "Earth",
                        "propagation_mode": "rk45",
                        "altitude_km": 500.0,
                        "eccentricity": 0.0005,
                        "inclination_deg": 51.6,
                        "raan_deg": 0.0,
                        "arg_pe_deg": 0.0,
                        "true_anomaly_deg": 0.0,
                        "mass": 150.0,
                        "drag_area": 1.2,
                        "cd": 2.2,
                        "color": "#dc2626",
                    },
                    {
                        "name": "NANODYNAMICS-PINN-GNC",
                        "parent_body": "Earth",
                        "propagation_mode": "pinn_local",
                        "altitude_km": 500.02,
                        "eccentricity": 0.0005,
                        "inclination_deg": 51.62,
                        "raan_deg": 0.05,
                        "arg_pe_deg": 0.0,
                        "true_anomaly_deg": 179.98,
                        "mass": 25.0,
                        "drag_area": 0.15,
                        "cd": 2.2,
                        "color": "#00ffcc",
                        "autonomous_avoidance": True,
                        "avoidance_range_km": 5.0,
                        "avoidance_dv": 15.0,
                    }
                ]
            }
        ]
    }


@app.get("/api/weather/live")
def get_live_weather():
    """Returns simulated / real-time global atmospheric weather metrics and cloud coverage."""
    import time
    t = time.time()
    diurnal_phase = (t % 86400) / 86400.0
    return {
        "timestamp": t,
        "diurnal_phase": diurnal_phase,
        "troposphere_cloud_cover_pct": 62.4 + 4.5 * math.sin(t * 0.001),
        "mean_surface_temp_k": 288.15 + 2.1 * math.sin(diurnal_phase * 2 * math.pi),
        "sea_level_pressure_hpa": 1013.25,
        "geomagnetic_kp_index": 2.3,
        "solar_wind_speed_km_s": 425.0,
        "interplanetary_mag_field_nt": 5.8,
        "aurora_activity": "moderate",
    }


@app.get("/list_saves")
def list_saves():
    files = [f.replace(".json", "") for f in os.listdir(SAVED_SIMS_DIR) if f.endswith(".json")]
    return {"files": files}


@app.get("/moon_track")
def get_moon_track(dt: float = 60.0, n: int = 1440):
    """Return Moon ephemeris positions in meters."""
    if not 0 < dt <= 86400.0:
        return {"error": "dt must be in (0, 86400] seconds"}
    if not 1 <= n <= 20000:
        return {"error": "n must be in [1, 20000]"}
    return [
        {"t": float(i * dt), "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
        for i in range(n)
        for pos in [moon_position(i * dt)]
    ]


class MLTrainRequest(BaseModel):
    epochs: int = Field(300, ge=1, le=50000, description="Training epochs (up to 50,000)")
    hidden: int = Field(32, ge=8, le=256)
    layers: int = Field(2, ge=1, le=8)
    lr: float = Field(1e-3, gt=0.0)
    force: bool = Field(False)
    resume: bool = Field(True, description="Resume training from previous checkpoint weights")
    alt_km: float = Field(420.0, ge=100.0, le=2000.0)
    inc_deg: float = Field(60.0, ge=0.0, le=180.0)


@app.post("/ai/train")
async def api_train_ml(req: MLTrainRequest = MLTrainRequest()):
    """Trigger training of local ML transition surrogate on accumulated simulation data."""
    try:
        from ai.agent import run_once
        res = await asyncio.to_thread(
            run_once,
            epochs=req.epochs,
            hidden=req.hidden,
            layers=req.layers,
            lr=req.lr,
            alt_km=req.alt_km,
            inc_deg=req.inc_deg,
            force=req.force,
            resume=req.resume,
        )
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/progress")
def api_get_ml_progress():
    """Return live training iteration progress."""
    possible_paths = [
        os.path.join(PROJECT_ROOT, "ai", "data", "training_live.json"),
        os.path.join(PROJECT_ROOT, "ai", "data", "combined_dt60", "training_live.json"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except Exception:
                pass
    return {"status": "idle", "epoch": 0, "total_epochs": 0, "progress_pct": 0.0}


@app.get("/ai/history")
def api_get_ml_history():
    """Return historical log of training sessions and prediction accuracy evaluations."""
    eval_history = []
    eval_log_path = os.path.join(PROJECT_ROOT, "ai", "data", "evaluation_history.jsonl")
    if os.path.exists(eval_log_path):
        try:
            with open(eval_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        eval_history.append(json.loads(line))
        except Exception:
            pass

    train_history = []
    train_log_path = os.path.join(PROJECT_ROOT, "ai", "checkpoints", "agent", "training_history.jsonl")
    if not os.path.exists(train_log_path):
        train_log_path = os.path.join(PROJECT_ROOT, "ai", "data", "training_history.jsonl")
    if os.path.exists(train_log_path):
        try:
            with open(train_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        train_history.append(json.loads(line))
        except Exception:
            pass

    return {
        "evaluation_history": eval_history,
        "training_history": train_history,
        "count_evaluations": len(eval_history),
        "count_training_runs": len(train_history)
    }


@app.get("/ai/status")
def api_get_ml_status():
    """Return latest ML agent training and rollout evaluation status."""
    agent_report_path = os.path.join(PROJECT_ROOT, "ai", "data", "agent_report.json")
    if os.path.exists(agent_report_path):
        try:
            with open(agent_report_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    state_path = os.path.join(PROJECT_ROOT, "ai", "data", "agent_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"status": "idle", "message": "No model trained yet. Run simulations and click Train ML Surrogate."}


@app.get("/ai/pairs")
def api_get_ml_pairs():
    """Count total ML training data pairs across all recorded datasets."""
    import numpy as np
    total = 0
    data_dir = os.path.join(PROJECT_ROOT, "ai", "data")
    npz_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".npz"):
                npz_files.append(os.path.join(root, f))
    counted = set()
    for path in npz_files:
        if path in counted:
            continue
        try:
            d = np.load(path, allow_pickle=False)
            # x array holds input state vectors — rows = samples
            if "x" in d:
                total += int(d["x"].shape[0])
                counted.add(path)
        except Exception:
            pass
    # Also count JSONL sim log rows as a secondary metric
    sim_log = os.path.join(data_dir, "sim_log.jsonl")
    log_rows = 0
    if os.path.exists(sim_log):
        try:
            with open(sim_log) as f:
                log_rows = sum(1 for ln in f if ln.strip())
        except Exception:
            pass
    return {"total_pairs": total, "sim_log_rows": log_rows, "npz_files_scanned": len(counted)}


class HFIngestRequest(BaseModel):
    dataset: str = Field("space_weather", description="Dataset key in leo_simulator.hf_data.DATASETS")
    year: int | None = Field(2024, description="Year for tle_history dataset")


class HFValidateRequest(BaseModel):
    norad_id: int = Field(25544, description="NORAD catalog ID (default ISS 25544)")
    year: int = Field(2024, description="Historical TLE year")
    max_pairs: int = Field(5, ge=1, le=50, description="Number of consecutive TLE pairs to evaluate")


@app.get("/api/hf/datasets")
def api_hf_datasets():
    """Return local cache status of Hugging Face astrodynamics datasets."""
    from leo_simulator.hf_data import dataset_status
    return {"datasets": dataset_status()}


@app.post("/api/hf/ingest")
def api_hf_ingest(req: HFIngestRequest):
    """Download/sync a Hugging Face dataset slice to the local cache."""
    from leo_simulator.hf_data import dataset_status, download_dataset
    try:
        path = download_dataset(req.dataset, year=req.year)
        return {"status": "ok", "path": str(path), "datasets": dataset_status()}
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=str(ke))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hf/validate")
def api_hf_validate(req: HFValidateRequest):
    """Validate RK45 and PINN propagators against historical TLE arcs."""
    from scripts.hf_validate_pinn import validate_tle_arc
    try:
        return validate_tle_arc(norad_id=req.norad_id, year=req.year, max_pairs=req.max_pairs)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dual-ESP32 EdgeLink & OTA WebSocket Bridge (/ws/edge) ─────────────────────
import struct as _struct

ORBIT_PACKET_FORMAT = "<H10f"  # uint16_t sat_id + 10 floats = 42 bytes
ORBIT_PACKET_SIZE = _struct.calcsize(ORBIT_PACKET_FORMAT)


class OrbitStatePacketModel(BaseModel):
    sat_id: int = Field(0, ge=0, le=65535)
    t: float = 0.0
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    B_coeff: float = 0.015
    dt: float = 10.0
    t_infer_ms: float = 0.0

    def to_bytes(self) -> bytes:
        return _struct.pack(
            ORBIT_PACKET_FORMAT,
            int(self.sat_id),
            float(self.t),
            float(self.x),
            float(self.y),
            float(self.z),
            float(self.vx),
            float(self.vy),
            float(self.vz),
            float(self.B_coeff),
            float(self.dt),
            float(self.t_infer_ms),
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "OrbitStatePacketModel":
        if len(raw) != ORBIT_PACKET_SIZE:
            raise ValueError(f"Expected {ORBIT_PACKET_SIZE} bytes for OrbitStatePacket, got {len(raw)}")
        vals = _struct.unpack(ORBIT_PACKET_FORMAT, raw)
        return cls(
            sat_id=vals[0],
            t=vals[1],
            x=vals[2],
            y=vals[3],
            z=vals[4],
            vx=vals[5],
            vy=vals[6],
            vz=vals[7],
            B_coeff=vals[8],
            dt=vals[9],
            t_infer_ms=vals[10],
        )


def symplectic_hamiltonian_project(
    r_next: np.ndarray,
    v_nn: np.ndarray,
    e_target: float,
    mu: float = 3.986004418e14,
) -> np.ndarray:
    """Symplectic Hamiltonian Manifold projection engine:
    v_target = sqrt(2 * (E_target + mu / ||r_{k+1}||))
    v_{k+1}  = v_NN * (v_target / ||v_NN||)
    """
    r_norm = float(np.linalg.norm(r_next))
    v_norm = float(np.linalg.norm(v_nn))
    if r_norm <= 1.0 or v_norm <= 1e-9:
        return np.asarray(v_nn, dtype=np.float64)
    two_term = 2.0 * (float(e_target) + (float(mu) / r_norm))
    if two_term <= 0.0:
        return np.asarray(v_nn, dtype=np.float64)
    v_target = math.sqrt(two_term)
    return np.asarray(v_nn, dtype=np.float64) * (v_target / v_norm)


@app.websocket("/ws/edge")
async def websocket_edge_bridge(websocket: WebSocket):
    """Persistent WebSocket bridge for ESP32 Node 1 Gateway (binary 42-byte struct or JSON)."""
    await websocket.accept()
    mu_earth = 3.986004418e14
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                pkt = OrbitStatePacketModel.from_bytes(msg["bytes"])
                r0 = np.array([pkt.x, pkt.y, pkt.z], dtype=np.float64)
                v0 = np.array([pkt.vx, pkt.vy, pkt.vz], dtype=np.float64)
                r0_norm = float(np.linalg.norm(r0))
                e_target = 0.5 * float(np.dot(v0, v0)) - (mu_earth / max(r0_norm, 1.0))

                # Advance one symplectic step + Hamiltonian manifold projection
                a0 = -mu_earth * r0 / (max(r0_norm, 1.0) ** 3)
                r1 = r0 + v0 * pkt.dt + 0.5 * a0 * (pkt.dt ** 2)
                r1_norm = float(np.linalg.norm(r1))
                a1 = -mu_earth * r1 / (max(r1_norm, 1.0) ** 3)
                v_nn = v0 + 0.5 * (a0 + a1) * pkt.dt
                v1 = symplectic_hamiltonian_project(r1, v_nn, e_target, mu=mu_earth)

                out_pkt = OrbitStatePacketModel(
                    sat_id=pkt.sat_id,
                    t=pkt.t + pkt.dt,
                    x=float(r1[0]),
                    y=float(r1[1]),
                    z=float(r1[2]),
                    vx=float(v1[0]),
                    vy=float(v1[1]),
                    vz=float(v1[2]),
                    B_coeff=pkt.B_coeff,
                    dt=pkt.dt,
                    t_infer_ms=pkt.t_infer_ms if pkt.t_infer_ms > 0.0 else 1.85,
                )
                await websocket.send_bytes(out_pkt.to_bytes())
            elif "text" in msg and msg["text"] is not None:
                data = json.loads(msg["text"])
                pkt = OrbitStatePacketModel(**data)
                r0 = np.array([pkt.x, pkt.y, pkt.z], dtype=np.float64)
                v0 = np.array([pkt.vx, pkt.vy, pkt.vz], dtype=np.float64)
                r0_norm = float(np.linalg.norm(r0))
                e_target = 0.5 * float(np.dot(v0, v0)) - (mu_earth / max(r0_norm, 1.0))

                a0 = -mu_earth * r0 / (max(r0_norm, 1.0) ** 3)
                r1 = r0 + v0 * pkt.dt + 0.5 * a0 * (pkt.dt ** 2)
                r1_norm = float(np.linalg.norm(r1))
                a1 = -mu_earth * r1 / (max(r1_norm, 1.0) ** 3)
                v_nn = v0 + 0.5 * (a0 + a1) * pkt.dt
                v1 = symplectic_hamiltonian_project(r1, v_nn, e_target, mu=mu_earth)

                out_pkt = OrbitStatePacketModel(
                    sat_id=pkt.sat_id,
                    t=pkt.t + pkt.dt,
                    x=float(r1[0]),
                    y=float(r1[1]),
                    z=float(r1[2]),
                    vx=float(v1[0]),
                    vy=float(v1[1]),
                    vz=float(v1[2]),
                    B_coeff=pkt.B_coeff,
                    dt=pkt.dt,
                    t_infer_ms=pkt.t_infer_ms if pkt.t_infer_ms > 0.0 else 1.85,
                )
                await websocket.send_text(out_pkt.model_dump_json())
    except WebSocketDisconnect:
        pass


class EdgeWifiProvisionRequest(BaseModel):
    ssid: str = Field(..., min_length=1, description="Target Wi-Fi SSID for ESP32 Gateway")
    password: str = Field("", description="Wi-Fi Password")
    host_ip: str = Field("192.168.1.38", description="Local machine IP running the NanoDynamics simulator")
    port: str = Field("/dev/ttyUSB0", description="Serial port or Wi-Fi host")


class EdgePinnParamsRequest(BaseModel):
    dt_s: float | None = Field(None, ge=0.5, le=600.0)
    b_coeff: float | None = Field(None, ge=0.0001, le=2.0)
    symplectic_enabled: bool | None = None
    transport_mode: str | None = None
    wifi_ssid: str | None = None
    wifi_host_ip: str | None = None


class EdgeBenchmarkRequest(BaseModel):
    steps: int = Field(16, ge=1, le=120, description="Number of live PINN orbital steps to execute")
    altitude_km: float = Field(550.0, ge=150.0, le=36000.0)
    inclination_deg: float = Field(53.0, ge=0.0, le=180.0)
    dt_s: float | None = None
    b_coeff: float | None = None


@app.get("/edge-console")
def read_edge_console():
    """Serve the dedicated Hugging Face + SpaceX-styled ESP32 PINN Telemetry & Wi-Fi Provisioning Web App."""
    return FileResponse(
        os.path.join(BASE_DIR, "static", "edge_console.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"},
    )


@app.get("/api/edge/status")
def api_edge_status():
    """Query live hardware status of connected ESP32 boards, Wi-Fi config, and PINN telemetry summary."""
    from hardware_in_the_loop.esp32_bridge import ESP32HardwareAgent

    gw_port = device_manager.pinn_params.get("gateway_port", "/dev/ttyUSB0")
    edge_port = device_manager.pinn_params.get("edge_port", "/dev/ttyUSB1")
    gw_status = {"role": "offline", "port": gw_port, "usb_present": os.path.exists(gw_port)}
    edge_status = {"role": "offline", "port": edge_port, "usb_present": os.path.exists(edge_port)}

    if os.path.exists(gw_port):
        try:
            agent = ESP32HardwareAgent.get_shared(port=gw_port)
            st = agent.query_status()
            if st:
                gw_status.update(st)
                gw_status["usb_present"] = True
                dev1 = device_manager.get_device("esp32-node-1")
                if dev1:
                    if st.get("mac"):
                        dev1.mac_address = st["mac"]
                    if st.get("sta_ip") and st["sta_ip"] != "0.0.0.0":
                        dev1.ip_address = st["sta_ip"]
                    dev1.status = "online"
        except Exception as e:
            gw_status["error"] = str(e)

    if os.path.exists(edge_port):
        try:
            agent2 = ESP32HardwareAgent.get_shared(port=edge_port)
            st2 = agent2.query_status()
            if st2:
                edge_status.update(st2)
                edge_status["usb_present"] = True
        except Exception as e:
            edge_status["error"] = str(e)

    return {
        "gateway": gw_status,
        "edge_node": edge_status,
        "devices": device_manager.list_devices(),
        "pinn_params": device_manager.pinn_params,
        "telemetry": device_manager.get_telemetry_summary(limit=80),
    }


@app.post("/api/edge/wifi")
def api_edge_set_wifi(req: EdgeWifiProvisionRequest):
    """Push new Wi-Fi SSID, password, and simulator Host IP to the ESP32 Gateway over USB or Wi-Fi."""
    from hardware_in_the_loop.esp32_bridge import ESP32HardwareAgent

    device_manager.update_pinn_params({"wifi_ssid": req.ssid, "wifi_host_ip": req.host_ip})
    agent = ESP32HardwareAgent.get_shared(port=req.port)
    result = agent.set_wifi_credentials(ssid=req.ssid, password=req.password, host_ip=req.host_ip)
    return result


@app.post("/api/edge/params")
def api_edge_update_params(req: EdgePinnParamsRequest):
    """Update live PINN & Symplectic Hamiltonian manifold parameters."""
    updated = device_manager.update_pinn_params(req.model_dump(exclude_none=True))
    return {"status": "ok", "pinn_params": updated}


@app.post("/api/edge/benchmark")
def api_edge_benchmark(req: EdgeBenchmarkRequest):
    """Execute a live burst of PINN orbital propagation steps on the physical ESP32 and stream telemetry."""
    import time
    import torch
    from hardware_in_the_loop.esp32_bridge import ESP32HardwareAgent

    dt_s = float(req.dt_s if req.dt_s is not None else device_manager.pinn_params.get("dt_s", 10.0))
    b_coeff = float(req.b_coeff if req.b_coeff is not None else device_manager.pinn_params.get("b_coeff", 0.015))
    symplectic_on = bool(device_manager.pinn_params.get("symplectic_enabled", True))
    gw_port = device_manager.pinn_params.get("gateway_port", "/dev/ttyUSB0")

    r_mag = 6378137.0 + req.altitude_km * 1000.0
    v_circ = math.sqrt(3.986004418e14 / r_mag)
    inc_rad = math.radians(req.inclination_deg)
    curr_r = np.array([r_mag, 0.0, 0.0], dtype=np.float64)
    curr_v = np.array([0.0, v_circ * math.cos(inc_rad), v_circ * math.sin(inc_rad)], dtype=np.float64)

    pos_scale = 6378137.0
    vel_scale = 8000.0
    sc = np.array([pos_scale] * 3 + [vel_scale] * 3, dtype=np.float64)
    mu_earth = 3.986004418e14
    e0 = 0.5 * float(np.dot(curr_v, curr_v)) - (mu_earth / r_mag)

    agent = ESP32HardwareAgent.get_shared(port=gw_port)
    prop = PINNPropagator(mode="pinn_local")

    results = []
    for step_idx in range(req.steps):
        state_in_norm = (np.concatenate([curr_r, curr_v]) / sc).tolist()
        t_start = time.perf_counter()
        hw_used = False
        engine_label = "pinn_esp32_hil"

        if os.path.exists(gw_port):
            try:
                nxt_norm, t_infer_ms, rtt_ms, mode_str = agent.step_detailed(
                    state_in_norm, dt=dt_s, b_coeff=b_coeff, sat_id=step_idx + 1
                )
                unscaled = nxt_norm.astype(np.float64) * sc
                transit_ms = max(0.05, rtt_ms - t_infer_ms)
                hw_used = True
                engine_label = mode_str
            except Exception:
                hw_used = False

        if not hw_used:
            t0_cpu = time.perf_counter()
            with torch.no_grad():
                inp_t = torch.tensor(state_in_norm, dtype=torch.float32).unsqueeze(0)
                out_t = prop.model(inp_t).squeeze(0).numpy()
            t_infer_ms = (time.perf_counter() - t0_cpu) * 1000.0
            transit_ms = 0.0
            unscaled = out_t.astype(np.float64) * sc
            engine_label = "pinn_cpu_fallback"

        r_nn = unscaled[0:3]
        v_nn = unscaled[3:6]
        r_kin = curr_r + 0.5 * (curr_v + v_nn) * dt_s
        r_next = 0.5 * r_nn + 0.5 * r_kin
        if symplectic_on:
            v_next = symplectic_hamiltonian_project(r_next, v_nn, e0, mu=mu_earth)
        else:
            v_next = v_nn

        r_next_mag = float(np.linalg.norm(r_next))
        e_now = 0.5 * float(np.dot(v_next, v_next)) - (mu_earth / max(r_next_mag, 1.0))
        e_drift = abs(e_now - e0) / max(abs(e0), 1.0)
        tot_ms = (time.perf_counter() - t_start) * 1000.0

        state_out_norm = (np.concatenate([r_next, v_next]) / sc).tolist()
        device_manager.record_telemetry(
            device_id="esp32-node-1",
            engine=engine_label,
            proc_time_ms=t_infer_ms,
            transit_latency_ms=transit_ms,
            total_step_ms=tot_ms,
            step_index=step_idx + 1,
            energy_error=e_drift,
            sat_id=f"HIL-{step_idx + 1:02d}",
        )
        log_entry = device_manager.record_pinn_request(
            device_id="esp32-node-1",
            sat_id=f"HIL-{step_idx + 1:02d}",
            engine=engine_label,
            state_in=state_in_norm,
            state_out=state_out_norm,
            dt_s=dt_s,
            b_coeff=b_coeff,
            t_infer_ms=t_infer_ms,
            transit_latency_ms=transit_ms,
            total_step_ms=tot_ms,
            energy_drift=e_drift,
        )
        results.append(asdict(log_entry))
        curr_r = r_next
        curr_v = v_next

    return {
        "status": "ok",
        "steps_executed": len(results),
        "hardware_active": hw_used,
        "telemetry": device_manager.get_telemetry_summary(limit=80),
    }


