from fastapi.testclient import TestClient
import pytest
import os
import sys

# Ensure project root is in sys.path
# Ensure project root and visualization directory are in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vis_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(vis_root))

from app import app, SAVED_SIMS_DIR

client = TestClient(app)

def test_simulate_valid_params():
    payload = {
        "name": "TestSat",
        "mass": 4.0,
        "drag_area": 0.03,
        "cd": 2.2,
        "altitude_km": 400.0,
        "eccentricity": 0.001,
        "inclination_deg": 51.6,
        "raan_deg": 45.0
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "TestSat"
    assert "trajectory" in data
    assert len(data["trajectory"]) > 0

def test_simulate_invalid_params_returns_400():
    # Negative altitude should trigger a ValueError in OrbitalElements / Satellite
    payload = {
        "name": "BadSat",
        "mass": 4.0,
        "drag_area": 0.03,
        "cd": 2.2,
        "altitude_km": -100.0,  # Invalid
        "eccentricity": 0.001,
        "inclination_deg": 51.6,
        "raan_deg": 45.0
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 400
    assert "detail" in response.json()

def test_save_load_roundtrip():
    # 1. Save
    save_payload = {
        "filename": "test_sim_roundtrip",
        "satellites": [
            {"name": "Sat1", "trajectory": []}
        ]
    }
    response = client.post("/save", json=save_payload)
    assert response.status_code == 200

    # 2. Load
    response = client.get("/load/test_sim_roundtrip")
    assert response.status_code == 200
    data = response.json()
    assert "satellites" in data
    assert len(data["satellites"]) == 1
    assert data["satellites"][0]["name"] == "Sat1"
    
    # 3. Clean up
    os.remove(os.path.join(SAVED_SIMS_DIR, "test_sim_roundtrip.json"))

def test_path_traversal_save_rejected():
    save_payload = {
        "filename": "../../../bashrc",
        "satellites": []
    }
    response = client.post("/save", json=save_payload)
    assert response.status_code == 400

def test_path_traversal_load_rejected():
    response = client.get("/load/..%2F..%2Fbashrc")
    # FastAPI may reject traversal via 404 before it even hits the endpoint, or our 400 catches it
    assert response.status_code in [400, 404]

def test_list_saves():
    # Ensure at least one save exists for the test
    save_payload = {
        "filename": "test_list_sim",
        "satellites": []
    }
    client.post("/save", json=save_payload)
    
    response = client.get("/list_saves")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert "test_list_sim" in data["files"]
    
    os.remove(os.path.join(SAVED_SIMS_DIR, "test_list_sim.json"))
