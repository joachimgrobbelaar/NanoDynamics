import os
import sys
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from leo_simulator.models.gravity import moon_position  # noqa: E402
from leo_simulator import Satellite, OrbitalElements, OrbitPropagator, ExponentialAtmosphere, R_EARTH  # noqa: E402

app = FastAPI()

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

SAVED_SIMS_DIR = os.path.join(BASE_DIR, "saved_sims")
os.makedirs(SAVED_SIMS_DIR, exist_ok=True)


class SatelliteParams(BaseModel):
    name: str
    mass: float
    drag_area: float
    cd: float
    altitude_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float


class SavePayload(BaseModel):
    filename: str
    satellites: list


import re

def sanitize_filename(filename: str) -> str:
    # Allow only alphanumeric, dashes, and underscores
    if not re.match(r'^[\w\-]+$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename format. Use only alphanumeric characters, dashes, and underscores.")
    return filename

@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.post("/simulate")
def simulate_satellite(params: SatelliteParams):
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
        
        # 2. Propagate
        prop = OrbitPropagator(
            satellite=sat, 
            atmosphere=atm, 
            include_central_gravity=True, 
            include_j2=True, 
            include_drag=True
        )
        
        # Shorten duration to 5 hours to prevent blocking the worker on UI interactions
        res = prop.propagate(orbit, duration_seconds=18000, dt_eval=60)
        
        # 3. Format & Return
        trajectory = [
            {
                "t": float(res.t[i]), 
                "x": float(res.r[i,0]), 
                "y": float(res.r[i,1]), 
                "z": float(res.r[i,2]),
                "vx": float(res.v[i,0]),
                "vy": float(res.v[i,1]),
                "vz": float(res.v[i,2])
            } 
            for i in range(len(res.t))
        ]
        
        # Use model_dump() for Pydantic V2
        return {
            "name": params.name, 
            "params": params.model_dump() if hasattr(params, 'model_dump') else params.dict(),
            "trajectory": trajectory
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
        {
            "t": float(i * dt),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
        }
        for i in range(n)
        for pos in [moon_position(i * dt)]
    ]
