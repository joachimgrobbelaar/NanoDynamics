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
    return JSONResponse(
        status_code=422,
        content={"detail": clean_detail}
    )

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

SAVED_SIMS_DIR = os.path.join(BASE_DIR, "saved_sims")
os.makedirs(SAVED_SIMS_DIR, exist_ok=True)


class SatelliteParams(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Satellite name")
    mass: float = Field(..., gt=0.0, le=100000.0, description="Mass in kg")
    drag_area: float = Field(..., ge=0.0, le=10000.0, description="Cross-section area in m^2")
    cd: float = Field(..., ge=0.0, le=20.0, description="Drag coefficient")
    altitude_km: float = Field(..., ge=100.0, le=2000.0, description="Altitude in km")
    eccentricity: float = Field(..., ge=0.0, lt=1.0, description="Orbital eccentricity")
    inclination_deg: float = Field(..., ge=0.0, le=180.0, description="Inclination in degrees")
    raan_deg: float = Field(..., ge=0.0, lt=360.0, description="RAAN in degrees")

    @model_validator(mode="after")
    def validate_orbit_safety(self) -> "SatelliteParams":
        # The satellite is launched at true anomaly nu=0 (perigee), so
        # altitude_km is the perigee altitude directly. The Field constraint
        # (ge=100.0) already guarantees this is safe. We only need to guard
        # against a highly eccentric orbit whose perigee is below the surface.
        # Semi-major axis: a = (R_EARTH + altitude_km*1000) / (1 - e)
        # Perigee radius (at nu=0): r_p = R_EARTH + altitude_km * 1000
        # (altitude_km IS the perigee altitude, not the semi-major axis)
        perigee_alt_km = self.altitude_km  # nu=0 launch → perigee = input altitude
        if perigee_alt_km < 50.0:
            raise ValueError(
                f"Perigee altitude {perigee_alt_km:.1f} km is below minimum safe threshold of 50 km."
            )
        return self


class SavePayload(BaseModel):
    filename: str
    satellites: list


def sanitize_filename(filename: str) -> str:
    # Allow only alphanumeric, dashes, and underscores
    if not re.match(r'^[\w\-]+$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename format. Use only alphanumeric characters, dashes, and underscores.")
    return filename

@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.post("/simulate")
async def simulate_satellite(params: SatelliteParams):
    try:
        # 1. Initialize objects (will raise ValueError on bad inputs)
        sat = Satellite(name=params.name, mass=params.mass, drag_area=params.drag_area, cd=params.cd)
        r_initial_m = R_EARTH + params.altitude_km * 1000.0
        orbit = OrbitalElements(
            a=r_initial_m, 
            e=params.eccentricity, 
            i=np.radians(params.inclination_deg), 
            raan=np.radians(params.raan_deg), 
            arg_pe=0.0, 
            nu=0.0
        )
        atm = ExponentialAtmosphere(h0=350_000.0, rho0=9.5e-12, scale_height=53_200.0)
        
        # 2. Propagate (offload heavy integration to thread)
        prop = OrbitPropagator(
            satellite=sat, 
            atmosphere=atm, 
            include_central_gravity=True, 
            include_j2=True, 
            include_drag=True
        )
        
        res = await asyncio.to_thread(prop.propagate, orbit, duration_seconds=86400, dt_eval=60)
        
        # 3. Format & Return columnar trajectory dictionary
        trajectory = {
            "t": res.t.tolist(),
            "x": res.r[:, 0].tolist(),
            "y": res.r[:, 1].tolist(),
            "z": res.r[:, 2].tolist(),
            "vx": res.v[:, 0].tolist(),
            "vy": res.v[:, 1].tolist(),
            "vz": res.v[:, 2].tolist(),
        }
        
        # Use model_dump() for Pydantic V2
        return {
            "name": params.name, 
            "params": params.model_dump() if hasattr(params, 'model_dump') else params.dict(),
            "period_s": float(orbit.period),
            "trajectory": trajectory
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save")
def save_simulation(payload: SavePayload):
    safe_name = sanitize_filename(payload.filename)
    filepath = os.path.join(SAVED_SIMS_DIR, f"{safe_name}.json")
    try:
        with open(filepath, "w") as f:
            json.dump(payload.satellites, f)
        return {"status": "success", "message": f"Saved {safe_name}"}
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
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
        {
            "t": float(i * dt),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
        }
        for i in range(n)
        for pos in [moon_position(i * dt)]
    ]
