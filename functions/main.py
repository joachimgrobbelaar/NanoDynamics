import typing
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from firebase_functions import https_fn
from firebase_admin import initialize_app, firestore

initialize_app()

from leo_simulator.experiment import run_single_simulation

@https_fn.on_call(memory=1024, timeout_sec=120)
def evaluate_sweep_point(req: https_fn.CallableRequest) -> typing.Any:
    """Serverless endpoint to evaluate a single trajectory and return metrics."""
    params = req.data["params"]
    
    # Optional trajectories?
    include_traj = req.data.get("include_trajectory", False)
    
    metrics, traj = run_single_simulation(
        name=params["name"],
        mass=params["mass"],
        drag_area=params["drag_area"],
        cd=params["cd"],
        altitude_km=params["altitude_km"],
        eccentricity=params["eccentricity"],
        inclination_deg=params["inclination_deg"],
        raan_deg=params.get("raan_deg", 0.0),
        atmosphere_type=req.data.get("atmosphere_type", "piecewise"),
        density_scale=req.data.get("density_scale", 1.0),
        include_j2=params.get("include_j2", True),
        include_drag=params.get("include_drag", True),
        include_moon=params.get("include_moon", False),
        max_duration_seconds=req.data.get("max_duration_seconds", 30*86400.0),
        include_trajectory=include_traj
    )
    
    # We can't return numpy floats, so ensure metrics are plain types
    clean_metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
    
    return {"metrics": clean_metrics, "trajectory": traj}
