import asyncio
import json
import os
import re
import sys

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
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
    Satellite,
)
from leo_simulator.models.gravity import moon_position

app = FastAPI()


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

ATM = ExponentialAtmosphere(h0=350_000.0, rho0=9.5e-12, scale_height=53_200.0)

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
        "default_chunk_duration": 600.0,
        "dt_eval": 10.0,
    },
    "Sun": {
        "name": "Sun",
        "mu": 1.32712440018e20,
        "radius_m": 696340000.0,
        "j2": 2.0e-7,
        "omega": 2.865e-6,
        "has_atmosphere": False,
        "min_alt_km": 1000000.0,
        "max_alt_km": 1000000000.0,
        "default_alt_km": 149600000.0,
        "default_ecc": 0.0167,
        "default_inc": 0.0,
        "default_raan": 0.0,
        "default_chunk_duration": 86400.0,
        "dt_eval": 1800.0,
    },
}


class SatelliteParams(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Satellite name")
    parent_body: str = Field("Earth", description="Central body: Earth, Moon, or Sun")
    mass: float = Field(..., gt=0.0, le=100000.0, description="Mass in kg")
    drag_area: float = Field(..., ge=0.0, le=10000.0, description="Cross-section area in m^2")
    cd: float = Field(..., ge=0.0, le=20.0, description="Drag coefficient")
    altitude_km: float = Field(..., description="Altitude in km")
    eccentricity: float = Field(..., ge=0.0, lt=1.0, description="Orbital eccentricity")
    inclination_deg: float = Field(..., ge=0.0, le=180.0, description="Inclination in degrees")
    raan_deg: float = Field(..., ge=0.0, lt=360.0, description="RAAN in degrees")
    # Perturbation toggles
    include_j2: bool = Field(True, description="Include J2 oblateness perturbation")
    include_drag: bool = Field(True, description="Include atmospheric drag")
    include_moon: bool = Field(False, description="Include lunar third-body gravity")

    @model_validator(mode="after")
    def validate_orbit_safety(self) -> "SatelliteParams":
        body = BODIES.get(self.parent_body, BODIES["Earth"])
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
    chunk_duration: float = Field(600.0, ge=10.0, le=604800.0,
                                  description="Duration of this chunk in seconds")


class BurnRequest(BaseModel):
    params: SatelliteParams
    t_burn: float = Field(..., description="Time at burn (s)")
    current_state: list[float] = Field(..., min_length=6, max_length=6, description="[x, y, z, vx, vy, vz]")
    dv_prograde: float = Field(0.0, description="Delta-v in prograde/transverse direction (m/s)")
    dv_normal: float = Field(0.0, description="Delta-v in normal/out-of-plane direction (m/s)")
    dv_radial: float = Field(0.0, description="Delta-v in radial-outward direction (m/s)")


class SavePayload(BaseModel):
    filename: str
    satellites: list


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
    atm = ATM if use_drag else None

    return OrbitPropagator(
        satellite=sat,
        atmosphere=atm,
        include_central_gravity=True,
        include_j2=(body["j2"] > 0 and params.include_j2),
        include_drag=use_drag,
        include_earth_rotation=use_drag,
        include_moon=(params.parent_body == "Earth" and params.include_moon),
        mu=body["mu"],
        r_earth=body["radius_m"],
        j2=body["j2"],
        omega_earth=body["omega"],
    )


def _compute_forces(prop: OrbitPropagator, t_arr, r_arr, v_arr):
    """Compute force vectors along a trajectory. Returns dict of lists."""
    forces = {"central": [], "j2": [], "drag": [], "moon": [], "total": []}
    for i in range(len(t_arr)):
        acc = prop.dynamics.compute_accelerations(r_arr[i], v_arr[i], t=float(t_arr[i]))
        for k in forces:
            forces[k].append(acc[k].tolist())
    return forces


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
    """Convert PropagationResult to columnar JSON-serialisable dict."""
    t = res.t.tolist()
    r = res.r
    v = res.v
    altitudes = ((np.linalg.norm(r, axis=1) - radius_m) / 1000.0).tolist()
    out = {
        "t": t,
        "x": r[:, 0].tolist(),
        "y": r[:, 1].tolist(),
        "z": r[:, 2].tolist(),
        "vx": v[:, 0].tolist(),
        "vy": v[:, 1].tolist(),
        "vz": v[:, 2].tolist(),
        "alt_km": altitudes,
        "reentry": bool(res.reentry_detected),
        "reentry_time": float(res.reentry_time) if res.reentry_time is not None else None,
    }
    if include_forces:
        out["forces"] = _compute_forces(prop, res.t, r, v)
    return out


@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/bodies")
def get_bodies():
    """Return parameters and constraints for each central celestial body."""
    return {"bodies": BODIES}


@app.post("/simulate")
async def simulate_satellite(params: SatelliteParams):
    """Bootstrap: propagates the first chunk and returns initial state + trajectory."""
    try:
        body = BODIES.get(params.parent_body, BODIES["Earth"])
        prop = _build_propagator(params)
        
        a_m = body["radius_m"] + params.altitude_km * 1000.0
        orbit = OrbitalElements(
            a=a_m,
            e=params.eccentricity,
            i=np.radians(params.inclination_deg),
            raan=np.radians(params.raan_deg),
            arg_pe=0.0,
            nu=0.0,
        )

        chunk_duration = body["default_chunk_duration"]
        dt_eval = body["dt_eval"]

        res = await asyncio.to_thread(
            prop.propagate, orbit, chunk_duration, 0.0, dt_eval
        )
        chunk = _format_chunk(res, prop, body["radius_m"], include_forces=True)

        # Return final Cartesian state for /stream continuity
        last_state = res.r[-1].tolist() + res.v[-1].tolist()
        period_s = 2.0 * np.pi * np.sqrt((a_m**3) / body["mu"])

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
        prop = _build_propagator(req.params)
        dt_eval = body["dt_eval"]
        res = await asyncio.to_thread(
            _propagate_chunk, prop, req.state, req.t_start, req.chunk_duration, dt_eval
        )
        chunk = _format_chunk(res, prop, body["radius_m"], include_forces=True)
        last_state = res.r[-1].tolist() + res.v[-1].tolist()

        return {
            "trajectory": chunk,
            "last_state": last_state,
            "last_t": float(res.t[-1]),
            "reentry": bool(res.reentry_detected),
            "reentry_time": float(res.reentry_time) if res.reentry_time is not None else None,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/burn")
async def execute_burn(req: BurnRequest):
    """Apply an instantaneous Delta-V in local RTN frame and propagate forward."""
    try:
        body = BODIES.get(req.params.parent_body, BODIES["Earth"])
        r = np.array(req.current_state[:3], dtype=np.float64)
        v = np.array(req.current_state[3:], dtype=np.float64)

        r_norm = np.linalg.norm(r)
        if r_norm <= 0.0:
            raise ValueError("Position vector norm must be positive.")
        u_r = r / r_norm

        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        u_n = h / h_norm if h_norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        u_t = np.cross(u_n, u_r)

        dv_vec = req.dv_radial * u_r + req.dv_prograde * u_t + req.dv_normal * u_n
        v_new = v + dv_vec
        new_state = r.tolist() + v_new.tolist()

        prop = _build_propagator(req.params)
        chunk_duration = body["default_chunk_duration"]
        dt_eval = body["dt_eval"]

        res = await asyncio.to_thread(
            _propagate_chunk, prop, new_state, req.t_burn, chunk_duration, dt_eval
        )
        chunk = _format_chunk(res, prop, body["radius_m"], include_forces=True)
        last_state = res.r[-1].tolist() + res.v[-1].tolist()

        return {
            "status": "success",
            "new_state": new_state,
            "trajectory": chunk,
            "last_state": last_state,
            "last_t": float(res.t[-1]),
            "reentry": bool(res.reentry_detected),
            "reentry_time": float(res.reentry_time) if res.reentry_time is not None else None,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save")
def save_simulation(payload: SavePayload):
    safe_name = sanitize_filename(payload.filename)
    filepath = os.path.join(SAVED_SIMS_DIR, f"{safe_name}.json")
    try:
        with open(filepath, "w") as f:
            json.dump(payload.satellites, f)
        return {"status": "success", "message": f"Saved {safe_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/load/{filename}")
def load_simulation(filename: str):
    safe_name = sanitize_filename(filename)
    filepath = os.path.join(SAVED_SIMS_DIR, f"{safe_name}.json")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return {"satellites": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
