"""
E2E API Test Suite for LEO Satellite Simulation Refactor.

Covers:
- R1: Backend Robustness & Input Validation (Pydantic models, boundary checks, clean 400/422 errors)
- R2: Asynchronous API Offloading (async endpoint definition, non-blocking execution)
- R3: Columnar Payload Compression (structural dictionary of arrays: t, x, y, z, vx, vy, vz)
- Endpoints: /simulate, /moon_track, /save, /load/{filename}, /list_saves, / (root)
- Security: Path traversal rejection, filename sanitization
"""

import inspect
import math
import os
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from leo_simulator import R_EARTH
from visualization_3d.app import SAVED_SIMS_DIR, app, simulate_satellite

client = TestClient(app)

REQUIRED_TRAJECTORY_KEYS = {"t", "x", "y", "z", "vx", "vy", "vz"}


class TestSimulateEndpointValid:
    """Tests asserting HTTP 200 and structural dictionary trajectory format (R3)."""

    def test_simulate_valid_columnar_payload(self):
        """Verify valid simulation returns HTTP 200 and columnar dictionary format."""
        payload = {
            "name": "LEOSat-Standard",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        # Metadata assertions
        assert data["name"] == "LEOSat-Standard"
        assert "params" in data
        assert "period_s" in data
        assert data["period_s"] > 0

        # Structural dictionary format assertion (R3 payload compression)
        assert "trajectory" in data, "Response missing 'trajectory' key"
        trajectory = data["trajectory"]
        assert isinstance(trajectory, dict), (
            f"Expected trajectory to be structural dictionary of arrays, but got {type(trajectory).__name__}"
        )

        # Keys assertion: {"t", "x", "y", "z", "vx", "vy", "vz"}
        assert REQUIRED_TRAJECTORY_KEYS.issubset(trajectory.keys()), (
            f"Trajectory missing required keys. Found: {list(trajectory.keys())}, "
            f"Expected: {REQUIRED_TRAJECTORY_KEYS}"
        )

        # Columnar arrays integrity
        n_points = len(trajectory["t"])
        assert n_points > 0, "Trajectory arrays must not be empty"

        for key in REQUIRED_TRAJECTORY_KEYS:
            arr = trajectory[key]
            assert isinstance(arr, list), f"Expected column '{key}' to be a list, got {type(arr).__name__}"
            assert len(arr) == n_points, f"Column '{key}' length {len(arr)} != t length {n_points}"
            # Verify all entries are finite floating-point numbers
            assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in arr), (
                f"Column '{key}' contains non-finite or non-numeric values"
            )

        # Monotonically increasing time
        t_arr = trajectory["t"]
        assert all(t_arr[i] < t_arr[i + 1] for i in range(len(t_arr) - 1)), "Time array 't' must be strictly increasing"

        # Initial position physical radius sanity check (at true anomaly nu=0, r0 = a * (1 - e))
        r0 = math.sqrt(trajectory["x"][0] ** 2 + trajectory["y"][0] ** 2 + trajectory["z"][0] ** 2)
        expected_r0 = (R_EARTH + 400_000.0) * (1.0 - 0.001)
        assert abs(r0 - expected_r0) < 1000.0, (
            f"Initial orbital radius {r0:.1f} m deviates from expected perigee {expected_r0:.1f} m"
        )

    @pytest.mark.parametrize("altitude_km", [100.0, 500.0, 1000.0, 2000.0])
    def test_simulate_valid_altitude_boundaries(self, altitude_km):
        """Test boundary altitudes within allowed physical range [100.0, 2000.0] km."""
        payload = {
            "name": f"Sat-Alt-{int(altitude_km)}",
            "mass": 10.0,
            "drag_area": 0.05,
            "cd": 2.2,
            "altitude_km": altitude_km,
            "eccentricity": 0.0001,
            "inclination_deg": 45.0,
            "raan_deg": 0.0,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code == 200, f"Failed at altitude_km={altitude_km}: {response.text}"
        data = response.json()
        assert isinstance(data["trajectory"], dict)
        assert len(data["trajectory"]["t"]) > 0

    @pytest.mark.parametrize("eccentricity", [0.0, 0.001, 0.01])
    def test_simulate_valid_eccentricity_boundaries(self, eccentricity):
        """Test circular (0.0) and near-circular orbits within safe perigee."""
        payload = {
            "name": f"Sat-Ecc-{eccentricity}",
            "mass": 5.0,
            "drag_area": 0.02,
            "cd": 2.2,
            "altitude_km": 600.0,
            "eccentricity": eccentricity,
            "inclination_deg": 30.0,
            "raan_deg": 0.0,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert REQUIRED_TRAJECTORY_KEYS.issubset(data["trajectory"].keys())

    @pytest.mark.parametrize("inclination_deg", [0.0, 51.6, 90.0, 98.0, 180.0])
    def test_simulate_valid_inclinations(self, inclination_deg):
        """Test valid inclination boundaries [0.0, 180.0] deg (equatorial, polar, retrograde)."""
        payload = {
            "name": f"Sat-Inc-{int(inclination_deg)}",
            "mass": 3.0,
            "drag_area": 0.01,
            "cd": 2.0,
            "altitude_km": 500.0,
            "eccentricity": 0.0001,
            "inclination_deg": inclination_deg,
            "raan_deg": 120.0,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code == 200


class TestSimulateEndpointInvalid:
    """Tests asserting HTTP 400 or 422 with descriptive details on invalid inputs (R1)."""

    @pytest.mark.parametrize("bad_mass", [-100.0, -5.0, -0.001, 0.0, 100000.1, 1000000.0])
    def test_simulate_invalid_mass(self, bad_mass):
        """Assert mass <= 0 or > 100000 kg returns HTTP 422 with descriptive detail."""
        payload = {
            "name": "BadMassSat",
            "mass": bad_mass,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code in [400, 422], f"Expected 400/422 for mass={bad_mass}, got {response.status_code}"
        detail = response.json().get("detail", "")
        assert detail, "Error response must contain non-empty 'detail' field"
        assert "mass" in str(detail).lower(), f"Detail should mention 'mass', got: {detail}"

    @pytest.mark.parametrize("bad_ecc", [-1.0, -0.1, -0.0001, 1.0, 1.05, 2.5])
    def test_simulate_invalid_eccentricity(self, bad_ecc):
        """Assert eccentricity < 0 or >= 1.0 (unbound/hyperbolic) returns HTTP 422."""
        payload = {
            "name": "BadEccSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": bad_ecc,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code in [400, 422], f"Expected 400/422 for eccentricity={bad_ecc}, got {response.status_code}"
        detail = response.json().get("detail", "")
        assert detail, "Error response must contain 'detail'"
        assert "eccentricity" in str(detail).lower(), f"Detail should mention 'eccentricity', got: {detail}"

    @pytest.mark.parametrize("bad_alt", [-500.0, -100.0, 0.0, 50.0, 99.9, 400000.1, 500000.0, 1000000.0])
    def test_simulate_invalid_altitude(self, bad_alt):
        """Assert altitude_km < 100.0 or > 400000.0 returns HTTP 422."""
        payload = {
            "name": "BadAltSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": bad_alt,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code in [400, 422], f"Expected 400/422 for altitude_km={bad_alt}, got {response.status_code}"
        detail = response.json().get("detail", "")
        assert detail, "Error response must contain 'detail'"
        assert "altitude" in str(detail).lower(), f"Detail should mention 'altitude', got: {detail}"

    @pytest.mark.parametrize("bad_inc", [-10.0, -0.01, 180.1, 270.0])
    def test_simulate_invalid_inclination(self, bad_inc):
        """Assert inclination outside [0.0, 180.0] degrees returns HTTP 422."""
        payload = {
            "name": "BadIncSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": bad_inc,
            "raan_deg": 45.0,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code in [400, 422]
        detail = response.json().get("detail", "")
        assert "inclination" in str(detail).lower()

    @pytest.mark.parametrize("bad_raan", [-1.0, 360.0, 400.0])
    def test_simulate_invalid_raan(self, bad_raan):
        """Assert RAAN outside [0.0, 360.0) degrees returns HTTP 422."""
        payload = {
            "name": "BadRaanSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": bad_raan,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code in [400, 422]
        detail = response.json().get("detail", "")
        assert "raan" in str(detail).lower()

    def test_simulate_perigee_collision_rejection(self):
        """Assert orbit with perigee below safe threshold (50 km) is rejected with HTTP 422."""
        # At altitude 200 km, e=0.05 -> perigee = (6371 + 200) * 0.95 - 6371 = -128.5 km (collision)
        payload = {
            "name": "CollisionSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 200.0,
            "eccentricity": 0.05,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code in [400, 422]
        detail = response.json().get("detail", "")
        assert any(word in str(detail).lower() for word in ["perigee", "collision", "safe", "threshold"]), (
            f"Expected perigee/collision warning, got: {detail}"
        )

    def test_simulate_empty_name_rejected(self):
        """Assert empty name string violates min_length=1."""
        payload = {
            "name": "",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        response = client.post("/simulate", json=payload)
        assert response.status_code in [400, 422]

    @pytest.mark.parametrize("missing_field", ["name", "mass", "drag_area", "cd", "altitude_km", "eccentricity", "inclination_deg", "raan_deg"])
    def test_simulate_missing_fields_rejected(self, missing_field):
        """Assert omitting any required field returns HTTP 422."""
        valid_payload = {
            "name": "FieldTestSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        payload = {k: v for k, v in valid_payload.items() if k != missing_field}
        response = client.post("/simulate", json=payload)
        assert response.status_code in [400, 422]
        detail = response.json().get("detail", "")
        assert missing_field in str(detail)

    @pytest.mark.parametrize("field,bad_val", [
        ("mass", "invalid_string"),
        ("mass", None),
        ("altitude_km", [400.0]),
        ("eccentricity", {"e": 0.1}),
    ])
    def test_simulate_invalid_types_rejected(self, field, bad_val):
        """Assert incorrect JSON data types are rejected with HTTP 422."""
        valid_payload = {
            "name": "TypeTestSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        valid_payload[field] = bad_val
        response = client.post("/simulate", json=valid_payload)
        assert response.status_code in [400, 422]


class TestAsyncOffloading:
    """Tests asserting R2 async offloading implementation."""

    def test_simulate_endpoint_is_async_coroutine(self):
        """Verify simulate_satellite endpoint handler is an async def coroutine (R2)."""
        assert inspect.iscoroutinefunction(simulate_satellite), (
            "simulate_satellite endpoint must be defined as 'async def' for non-blocking execution"
        )


class TestMoonTrackEndpoint:
    """Tests for the /moon_track ephemeris endpoint."""

    def test_moon_track_default_parameters(self):
        """Verify default call returns 1440 ephemeris coordinates with keys {t, x, y, z}."""
        response = client.get("/moon_track")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1440
        first = data[0]
        assert {"t", "x", "y", "z"}.issubset(first.keys())
        assert first["t"] == 0.0
        assert math.isfinite(first["x"]) and math.isfinite(first["y"]) and math.isfinite(first["z"])

    def test_moon_track_custom_query_params(self):
        """Verify custom dt and n query parameters."""
        response = client.get("/moon_track?dt=120.0&n=20")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 20
        assert data[0]["t"] == 0.0
        assert data[1]["t"] == 120.0

    @pytest.mark.parametrize("bad_dt", [0.0, -10.0, 90000.0])
    def test_moon_track_invalid_dt(self, bad_dt):
        """Verify invalid dt outside (0, 86400] returns error indication."""
        response = client.get(f"/moon_track?dt={bad_dt}&n=10")
        assert response.status_code == 200 or response.status_code == 400
        data = response.json()
        assert "error" in data or "detail" in data

    @pytest.mark.parametrize("bad_n", [0, -5, 25000])
    def test_moon_track_invalid_n(self, bad_n):
        """Verify invalid n outside [1, 20000] returns error indication."""
        response = client.get(f"/moon_track?dt=60.0&n={bad_n}")
        assert response.status_code == 200 or response.status_code == 400
        data = response.json()
        assert "error" in data or "detail" in data


class TestSaveLoadRoundtripAndSecurity:
    """Tests for /save, /load, /list_saves, and path traversal security."""

    @pytest.fixture(autouse=True)
    def cleanup_saves(self):
        """Ensure test save files are cleaned up before and after test execution."""
        test_files = ["test_e2e_sim_roundtrip", "test_columnar_save", "test_list_saves_fixture"]
        for tf in test_files:
            fp = os.path.join(SAVED_SIMS_DIR, f"{tf}.json")
            if os.path.exists(fp):
                os.remove(fp)
        yield
        for tf in test_files:
            fp = os.path.join(SAVED_SIMS_DIR, f"{tf}.json")
            if os.path.exists(fp):
                os.remove(fp)

    def test_save_and_load_roundtrip_legacy_format(self):
        """Verify saving and loading simulation payload with satellite array."""
        save_payload = {
            "filename": "test_e2e_sim_roundtrip",
            "satellites": [
                {
                    "name": "Sat-Legacy-1",
                    "params": {"mass": 4.0, "altitude_km": 400.0},
                    "trajectory": [{"t": 0, "x": 6771000.0, "y": 0.0, "z": 0.0}],
                }
            ],
        }
        res_save = client.post("/save", json=save_payload)
        assert res_save.status_code == 200
        assert res_save.json().get("status") == "success"

        res_load = client.get("/load/test_e2e_sim_roundtrip")
        assert res_load.status_code == 200
        data = res_load.json()
        assert "satellites" in data
        assert len(data["satellites"]) == 1
        assert data["satellites"][0]["name"] == "Sat-Legacy-1"

    def test_save_and_load_columnar_format(self):
        """Verify saving and loading simulation payload with new columnar dictionary format (R3)."""
        save_payload = {
            "filename": "test_columnar_save",
            "satellites": [
                {
                    "name": "Sat-Columnar-1",
                    "params": {"mass": 5.0, "altitude_km": 500.0},
                    "trajectory": {
                        "t": [0.0, 60.0],
                        "x": [6871000.0, 6870000.0],
                        "y": [0.0, 1000.0],
                        "z": [0.0, 500.0],
                        "vx": [0.0, -10.0],
                        "vy": [7600.0, 7590.0],
                        "vz": [100.0, 100.0],
                    },
                }
            ],
        }
        res_save = client.post("/save", json=save_payload)
        assert res_save.status_code == 200

        res_load = client.get("/load/test_columnar_save")
        assert res_load.status_code == 200
        loaded_sat = res_load.json()["satellites"][0]
        assert isinstance(loaded_sat["trajectory"], dict)
        assert REQUIRED_TRAJECTORY_KEYS.issubset(loaded_sat["trajectory"].keys())
        assert loaded_sat["trajectory"]["t"] == [0.0, 60.0]

    def test_list_saves(self):
        """Verify /list_saves lists available filenames without .json extension."""
        save_payload = {"filename": "test_list_saves_fixture", "satellites": []}
        client.post("/save", json=save_payload)

        response = client.get("/list_saves")
        assert response.status_code == 200
        files = response.json().get("files", [])
        assert "test_list_saves_fixture" in files
        assert not any(f.endswith(".json") for f in files)

    def test_load_nonexistent_file_returns_404(self):
        """Assert requesting an unknown simulation filename returns HTTP 404."""
        response = client.get("/load/definitely_not_a_real_saved_simulation_file_9999")
        assert response.status_code == 404

    @pytest.mark.parametrize("malicious_filename", [
        "../../../etc/passwd",
        "..\\..\\windows\\system32",
        "sub/dir",
        "file;whoami",
        "file name with spaces",
        "file\x00null",
    ])
    def test_save_path_traversal_rejected(self, malicious_filename):
        """Assert path traversal or illegal characters in save filename are rejected with HTTP 400."""
        payload = {"filename": malicious_filename, "satellites": []}
        response = client.post("/save", json=payload)
        assert response.status_code in [400, 422], (
            f"Expected rejection for malicious save filename {malicious_filename}, got {response.status_code}"
        )

    @pytest.mark.parametrize("malicious_load", [
        "..%2F..%2Fetc%2Fpasswd",
        "../../bashrc",
        "sub/folder",
    ])
    def test_load_path_traversal_rejected(self, malicious_load):
        """Assert path traversal in load filename is rejected (HTTP 400 or 404)."""
        response = client.get(f"/load/{malicious_load}")
        assert response.status_code in [400, 404], (
            f"Expected rejection for traversal load {malicious_load}, got {response.status_code}"
        )


class TestRootEndpoint:
    """Tests for the frontend root / endpoint."""

    def test_root_serves_html(self):
        """Verify GET / serves index.html with HTTP 200 and HTML content."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        content = response.text
        assert "Three.js" in content or "top-nav" in content or "modal-overlay" in content


class TestGeodeticCoordinatesAndBurnEndpoints:
    """Tests for geodetic coordinates (lat, lon) and burn preview/execution."""

    def test_simulate_contains_geodetic_coordinates_and_events(self):
        payload = {
            "name": "Sat-GeoTest",
            "mass": 5.0,
            "drag_area": 0.02,
            "cd": 2.2,
            "altitude_km": 350.0,
            "eccentricity": 0.001,
            "inclination_deg": 45.0,
            "raan_deg": 30.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 200
        data = res.json()
        traj = data["trajectory"]
        assert "lat_deg" in traj
        assert "lon_deg" in traj
        assert len(traj["lat_deg"]) == len(traj["t"])
        assert len(traj["lon_deg"]) == len(traj["t"])
        assert "events" in traj
        assert "initial" in traj["events"]
        assert traj["events"]["initial"]["lat_deg"] is not None

    def test_burn_preview_endpoint(self):
        payload = {
            "params": {
                "name": "Sat-BurnTest",
                "mass": 5.0,
                "drag_area": 0.02,
                "cd": 2.2,
                "altitude_km": 350.0,
                "eccentricity": 0.001,
                "inclination_deg": 45.0,
                "raan_deg": 30.0,
            },
            "t_burn": 0.0,
            "current_state": [6728137.0, 0.0, 0.0, 0.0, 7700.0, 0.0],
            "dv_prograde": 15.0,
            "dv_normal": 0.0,
            "dv_radial": 5.0,
        }
        res = client.post("/burn/preview", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "trajectory" in data
        assert "lat_deg" in data["trajectory"]
        assert "lon_deg" in data["trajectory"]
        assert len(data["trajectory"]["t"]) > 0

