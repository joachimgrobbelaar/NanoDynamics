import os
import sys

from fastapi.testclient import TestClient

# Ensure project root is in sys.path
# Ensure project root and visualization directory are in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vis_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(vis_root))

from app import SAVED_SIMS_DIR, app

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
    assert isinstance(data["trajectory"], dict)
    expected_keys = {"t", "x", "y", "z", "vx", "vy", "vz"}
    assert expected_keys.issubset(set(data["trajectory"].keys()))
    n_points = len(data["trajectory"]["t"])
    assert n_points > 0
    for key in expected_keys:
        assert len(data["trajectory"][key]) == n_points

def test_simulate_invalid_params_returns_400():
    # Negative altitude should trigger a validation error
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
    assert response.status_code in (400, 422)
    assert "detail" in response.json()
    assert isinstance(response.json()["detail"], str)

def test_simulate_negative_mass_rejected():
    payload = {
        "name": "TestSat",
        "mass": -1.0,
        "drag_area": 0.03,
        "cd": 2.2,
        "altitude_km": 400.0,
        "eccentricity": 0.001,
        "inclination_deg": 51.6,
        "raan_deg": 45.0
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code in (400, 422)
    assert "mass" in response.json()["detail"]

def test_simulate_eccentricity_bounds_rejected():
    # Eccentricity >= 1.0 (parabolic/hyperbolic) rejected
    payload = {
        "name": "TestSat",
        "mass": 4.0,
        "drag_area": 0.03,
        "cd": 2.2,
        "altitude_km": 400.0,
        "eccentricity": 1.0,
        "inclination_deg": 51.6,
        "raan_deg": 45.0
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code in (400, 422)

    # Negative eccentricity rejected
    payload["eccentricity"] = -0.05
    response = client.post("/simulate", json=payload)
    assert response.status_code in (400, 422)

def test_simulate_perigee_collision_validator():
    # High eccentricity at low altitude causes perigee below safe 50 km threshold
    payload = {
        "name": "CrashingSat",
        "mass": 4.0,
        "drag_area": 0.03,
        "cd": 2.2,
        "altitude_km": 200.0,
        "eccentricity": 0.5,  # perigee would be subterranean
        "inclination_deg": 51.6,
        "raan_deg": 45.0
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code in (400, 422)
    detail = response.json()["detail"].lower()
    assert "perigee" in detail or "collision" in detail or "safe" in detail

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


def test_simulate_bounds_validation():
    base_valid = {
        "name": "TestSat",
        "mass": 4.0,
        "drag_area": 0.03,
        "cd": 2.2,
        "altitude_km": 400.0,
        "eccentricity": 0.001,
        "inclination_deg": 51.6,
        "raan_deg": 45.0,
    }

    # Empty name
    p = {**base_valid, "name": ""}
    res = client.post("/simulate", json=p)
    assert res.status_code in (400, 422)

    # Altitude too high (> 2000 km)
    p = {**base_valid, "altitude_km": 2500.0}
    res = client.post("/simulate", json=p)
    assert res.status_code in (400, 422)

    # CD too high (> 20.0)
    p = {**base_valid, "cd": 25.0}
    res = client.post("/simulate", json=p)
    assert res.status_code in (400, 422)

    # Negative drag area
    p = {**base_valid, "drag_area": -0.1}
    res = client.post("/simulate", json=p)
    assert res.status_code in (400, 422)

    # Inclination > 180 deg
    p = {**base_valid, "inclination_deg": 185.0}
    res = client.post("/simulate", json=p)
    assert res.status_code in (400, 422)

    # RAAN >= 360 deg
    p = {**base_valid, "raan_deg": 360.0}
    res = client.post("/simulate", json=p)
    assert res.status_code in (400, 422)

    # Missing required field
    p = {k: v for k, v in base_valid.items() if k != "mass"}
    res = client.post("/simulate", json=p)
    assert res.status_code in (400, 422)


def test_simulate_clean_error_message_format():
    # Verify that error details are clean human-readable strings, not raw JSON blobs
    res = client.post("/simulate", json={"name": "Sat", "mass": -10.0})
    assert res.status_code in (400, 422)
    data = res.json()
    assert "detail" in data
    assert isinstance(data["detail"], str)
    assert "mass" in data["detail"]


def test_simulate_trajectory_arrays_numeric():
    payload = {
        "name": "NumericSat",
        "mass": 10.0,
        "drag_area": 0.1,
        "cd": 2.2,
        "altitude_km": 500.0,
        "eccentricity": 0.01,
        "inclination_deg": 98.0,
        "raan_deg": 120.0,
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 200
    traj = response.json()["trajectory"]
    for col in ("t", "x", "y", "z", "vx", "vy", "vz"):
        assert col in traj
        assert isinstance(traj[col], list)
        assert len(traj[col]) > 0
        assert all(isinstance(val, (int, float)) for val in traj[col][:10])
