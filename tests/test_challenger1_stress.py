"""
Adversarial Stress Test Suite for LEO Satellite Simulation Refactor.
Challenger 1 Empirical Verification (Milestone M4).

Test Categories:
1. Concurrency & Async Worker Offloading (parallel requests under load, mixed valid/invalid).
2. Physical Boundary Values (minimal mass, maximal mass, boundary altitudes, near-unity eccentricity, collision thresholds).
3. Adversarial Types, Missing Fields, and Malformed JSON (NaN, Infinity, booleans, bad types).
4. Trajectory Columnar Structure & Numerical Integrity (identical lengths, no NaN/inf, monotonic time).
5. Physical Energy Conservation & Monotonic Drag Dissipation (Hamiltonian conservation without drag, energy decay with drag).
6. Security & Auxiliary Endpoints (path traversal, filename sanitization, moon_track boundaries).
"""

import asyncio
import math
import os
import sys

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from leo_simulator.constants import J2_EARTH, MU_EARTH, R_EARTH
from visualization_3d.app import app

client = TestClient(app)
REQUIRED_KEYS = {"t", "x", "y", "z", "vx", "vy", "vz"}


class TestConcurrentLoad:
    """Stress tests verifying asynchronous offloading and stability under concurrent load."""

    @pytest.mark.anyio
    async def test_concurrent_simulate_batch(self):
        """Simultaneously launch 10 propagation tasks to test worker thread concurrency."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
            payloads = [
                {
                    "name": f"ConcurrentSat-{i}",
                    "mass": 4.0 + i * 0.5,
                    "drag_area": 0.03,
                    "cd": 2.2,
                    "altitude_km": 400.0 + i * 20.0,
                    "eccentricity": 0.001,
                    "inclination_deg": 45.0 + i * 2.0,
                    "raan_deg": 10.0 * i,
                }
                for i in range(10)
            ]

            tasks = [async_client.post("/simulate", json=p) for p in payloads]
            responses = await asyncio.gather(*tasks)

            for i, r in enumerate(responses):
                assert r.status_code == 200, f"Task {i} failed: {r.text}"
                data = r.json()
                assert data["name"] == f"ConcurrentSat-{i}"
                assert "trajectory" in data
                traj = data["trajectory"]
                assert REQUIRED_KEYS.issubset(traj.keys())
                n_pts = len(traj["t"])
                assert n_pts > 0
                for k in REQUIRED_KEYS:
                    assert len(traj[k]) == n_pts
                    assert all(math.isfinite(v) for v in traj[k])
                # Check monotonic time
                assert all(traj["t"][idx] < traj["t"][idx + 1] for idx in range(n_pts - 1))

    @pytest.mark.anyio
    async def test_concurrent_mixed_valid_and_invalid(self):
        """Simultaneously send 10 valid and 10 invalid requests to verify failure isolation."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
            tasks = []
            expected_valid = []

            for i in range(10):
                # Valid payload
                tasks.append(
                    async_client.post(
                        "/simulate",
                        json={
                            "name": f"MixedSat-Valid-{i}",
                            "mass": 5.0,
                            "drag_area": 0.02,
                            "cd": 2.2,
                            "altitude_km": 500.0,
                            "eccentricity": 0.001,
                            "inclination_deg": 51.6,
                            "raan_deg": 30.0,
                        },
                    )
                )
                expected_valid.append(True)

                # Invalid payload (alternating invalid field)
                bad_payload = {
                    "name": f"MixedSat-Invalid-{i}",
                    "mass": -1.0 if i % 2 == 0 else 5.0,
                    "drag_area": 0.02,
                    "cd": 2.2,
                    "altitude_km": 50.0 if i % 2 != 0 else 500.0,
                    "eccentricity": 0.9999 if i % 3 == 0 else 0.001,
                    "inclination_deg": 51.6,
                    "raan_deg": 30.0,
                }
                tasks.append(async_client.post("/simulate", json=bad_payload))
                expected_valid.append(False)

            responses = await asyncio.gather(*tasks)

            for i, (resp, is_val) in enumerate(zip(responses, expected_valid, strict=True)):
                if is_val:
                    assert resp.status_code == 200, f"Expected 200 for task {i}, got {resp.status_code}"
                    assert "trajectory" in resp.json()
                else:
                    assert resp.status_code in (400, 422), f"Expected 400/422 for invalid task {i}, got {resp.status_code}"
                    assert "detail" in resp.json()


class TestBoundaryValues:
    """Rigorous boundary and extreme value testing across physical parameter limits."""

    def test_minimal_mass_nanosat_reentry(self):
        """Mass at physical lower boundary 1e-6 kg (nanogram/chipsat) with re-entry."""
        payload = {
            "name": "MicroNanosat",
            "mass": 1e-6,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 100.0001,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        traj = data["trajectory"]
        n_pts = len(traj["t"])
        assert n_pts > 0
        for k in REQUIRED_KEYS:
            assert len(traj[k]) == n_pts
            assert all(math.isfinite(x) for x in traj[k])
        # Re-entry occurs quickly before full 86400s
        assert traj["t"][-1] < 86400.0

    def test_maximal_mass_space_station(self):
        """Mass at physical upper boundary 100000.0 kg (100-ton station)."""
        payload = {
            "name": "HeavyStation",
            "mass": 100000.0,
            "drag_area": 50.0,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 0.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 200
        data = res.json()
        traj = data["trajectory"]
        assert len(traj["t"]) == 1441
        for k in REQUIRED_KEYS:
            assert all(math.isfinite(x) for x in traj[k])

    @pytest.mark.parametrize("bad_mass", [0.0, -1e-6, -100.0, 100000.0001, 1e7])
    def test_mass_outside_bounds_rejected(self, bad_mass):
        """Mass <= 0 or > 100000 must return HTTP 422 with descriptive detail."""
        payload = {
            "name": "BadMassSat",
            "mass": bad_mass,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 0.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 422
        assert "mass" in res.json().get("detail", "").lower()

    def test_altitude_minimum_boundary(self):
        """Altitude exactly at minimum boundary 100.0 km."""
        payload = {
            "name": "MinAltSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 100.0,
            "eccentricity": 0.0,
            "inclination_deg": 51.6,
            "raan_deg": 0.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 200
        traj = res.json()["trajectory"]
        assert len(traj["t"]) > 0
        for k in REQUIRED_KEYS:
            assert all(math.isfinite(x) for x in traj[k])

    def test_altitude_maximum_boundary(self):
        """Altitude exactly at maximum boundary 2000.0 km."""
        payload = {
            "name": "MaxAltSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 2000.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 0.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 200
        traj = res.json()["trajectory"]
        assert len(traj["t"]) == 1441

    @pytest.mark.parametrize("bad_alt", [99.9999, -10.0, 2000.0001, 36000.0])
    def test_altitude_outside_bounds_rejected(self, bad_alt):
        """Altitude < 100 or > 2000 km must return HTTP 422."""
        payload = {
            "name": "BadAltSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": bad_alt,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 0.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 422
        assert "altitude_km" in res.json().get("detail", "").lower()

    def test_eccentricity_near_unity_collision_rejection(self):
        """Eccentricity near 1.0 (e=0.999999) must be rejected due to collision risk."""
        payload = {
            "name": "NearParabolicSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 1000.0,
            "eccentricity": 0.999999,
            "inclination_deg": 51.6,
            "raan_deg": 0.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 422
        detail = res.json().get("detail", "").lower()
        assert "perigee" in detail or "collision" in detail

    def test_eccentricity_perigee_threshold_knife_edge(self):
        """At altitude 2000km, e=0.23 has perigee > 50km (valid), e=0.24 has perigee < 50km (invalid)."""
        # e = 0.23 -> rp = 8371 * 0.77 = 6445.67 km > 6421 km (valid)
        res_valid = client.post(
            "/simulate",
            json={
                "name": "SafeHighEcc",
                "mass": 10.0,
                "drag_area": 0.05,
                "cd": 2.2,
                "altitude_km": 2000.0,
                "eccentricity": 0.23,
                "inclination_deg": 30.0,
                "raan_deg": 0.0,
            },
        )
        assert res_valid.status_code == 200
        assert len(res_valid.json()["trajectory"]["t"]) == 1441

        # e = 0.24 -> rp = 8371 * 0.76 = 6361.96 km < 6421 km (invalid, collision)
        res_invalid = client.post(
            "/simulate",
            json={
                "name": "UnsafeHighEcc",
                "mass": 10.0,
                "drag_area": 0.05,
                "cd": 2.2,
                "altitude_km": 2000.0,
                "eccentricity": 0.24,
                "inclination_deg": 30.0,
                "raan_deg": 0.0,
            },
        )
        assert res_invalid.status_code == 422
        assert "collision" in res_invalid.json().get("detail", "").lower()

    @pytest.mark.parametrize("inc", [0.0, 180.0])
    def test_inclination_boundaries(self, inc):
        """Inclination at exact physical extremes 0.0 and 180.0."""
        payload = {
            "name": f"IncSat-{int(inc)}",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": inc,
            "raan_deg": 0.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 200

    @pytest.mark.parametrize("bad_inc", [-0.0001, 180.0001])
    def test_inclination_outside_bounds(self, bad_inc):
        """Inclination < 0.0 or > 180.0 rejected with 422."""
        payload = {
            "name": "BadIncSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": bad_inc,
            "raan_deg": 0.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 422

    @pytest.mark.parametrize("raan", [0.0, 359.9999])
    def test_raan_boundaries(self, raan):
        """RAAN at valid boundaries [0.0, 360.0)."""
        payload = {
            "name": "RaanSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": raan,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 200

    @pytest.mark.parametrize("bad_raan", [-0.0001, 360.0, 360.0001])
    def test_raan_outside_bounds(self, bad_raan):
        """RAAN < 0.0 or >= 360.0 rejected with 422."""
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
        res = client.post("/simulate", json=payload)
        assert res.status_code == 422

    def test_name_length_boundaries(self):
        """Name min_length=1 and max_length=100 enforcement."""
        base_payload = {
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 0.0,
        }
        # 1 char: OK
        res1 = client.post("/simulate", json={**base_payload, "name": "X"})
        assert res1.status_code == 200

        # 100 chars: OK
        res100 = client.post("/simulate", json={**base_payload, "name": "A" * 100})
        assert res100.status_code == 200

        # 0 chars: 422
        res0 = client.post("/simulate", json={**base_payload, "name": ""})
        assert res0.status_code == 422

        # 101 chars: 422
        res101 = client.post("/simulate", json={**base_payload, "name": "A" * 101})
        assert res101.status_code == 422


class TestAdversarialTypesAndMissingFields:
    """Type confusion, injection, and serialization boundary checks."""

    @pytest.mark.parametrize(
        "bad_val",
        ["NaN", "Infinity", "-Infinity", "not_a_number"],
    )
    def test_special_strings_for_float_rejected(self, bad_val):
        """Assert NaN, Infinity, and invalid string floats are rejected with 422."""
        payload = {
            "name": "AdversarialFloat",
            "mass": bad_val,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 422
        assert "detail" in res.json()

    def test_raw_nan_in_json_body_rejected(self):
        """Assert non-standard raw NaN in JSON body is rejected with 422."""
        raw_body = b'{"name": "NaNPayload", "mass": NaN, "drag_area": 0.03, "cd": 2.2, "altitude_km": 400.0, "eccentricity": 0.001, "inclination_deg": 51.6, "raan_deg": 45.0}'
        res = client.post("/simulate", content=raw_body, headers={"content-type": "application/json"})
        assert res.status_code == 422

    def test_boolean_false_for_mass_rejected(self):
        """Assert boolean False (coerced to 0.0 in Python) violates mass > 0.0 constraint."""
        payload = {
            "name": "BoolSat",
            "mass": False,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 400.0,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 45.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 422
        assert "mass" in res.json().get("detail", "").lower()

    def test_malformed_json_syntax_returns_422(self):
        """Assert completely broken JSON syntax returns 422 with decode error detail."""
        res = client.post("/simulate", content=b"{{{invalid_json", headers={"content-type": "application/json"})
        assert res.status_code == 422
        detail = res.json().get("detail", "").lower()
        assert "json" in detail or "decode" in detail


class TestPhysicalEnergyAndNumericalSanity:
    """Numerical stability, energy conservation, and monotonic dissipation checks."""

    def test_hamiltonian_energy_conservation_without_drag(self):
        """In absence of non-conservative drag (cd=0), specific mechanical energy + J2 potential is conserved."""
        payload = {
            "name": "EnergyConservationSat",
            "mass": 10.0,
            "drag_area": 0.05,
            "cd": 0.0,
            "altitude_km": 500.0,
            "eccentricity": 0.001,
            "inclination_deg": 45.0,
            "raan_deg": 0.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 200
        traj = res.json()["trajectory"]

        x, y, z = np.array(traj["x"]), np.array(traj["y"]), np.array(traj["z"])
        vx, vy, vz = np.array(traj["vx"]), np.array(traj["vy"]), np.array(traj["vz"])

        r_norms = np.sqrt(x**2 + y**2 + z**2)
        v_norms = np.sqrt(vx**2 + vy**2 + vz**2)

        # Hamiltonian: E = 0.5 * v^2 - mu / r + phi_J2
        phi_j2 = (MU_EARTH * J2_EARTH * (R_EARTH**2) / (2.0 * r_norms**3)) * (3.0 * (z / r_norms) ** 2 - 1.0)
        E_tot = 0.5 * v_norms**2 - MU_EARTH / r_norms + phi_j2

        # Assert energy variation is negligible (< 1e-8 relative deviation)
        rel_deviation = (np.max(E_tot) - np.min(E_tot)) / abs(E_tot[0])
        assert rel_deviation < 1e-8, f"Energy drift too large: rel_deviation = {rel_deviation}"

    def test_monotonic_energy_dissipation_with_drag(self):
        """With atmospheric drag active (cd=2.2, h=300km), total energy strictly dissipates over 24h."""
        payload = {
            "name": "DragDissipationSat",
            "mass": 4.0,
            "drag_area": 0.03,
            "cd": 2.2,
            "altitude_km": 300.0,
            "eccentricity": 0.001,
            "inclination_deg": 45.0,
            "raan_deg": 0.0,
        }
        res = client.post("/simulate", json=payload)
        assert res.status_code == 200
        traj = res.json()["trajectory"]

        x, y, z = np.array(traj["x"]), np.array(traj["y"]), np.array(traj["z"])
        vx, vy, vz = np.array(traj["vx"]), np.array(traj["vy"]), np.array(traj["vz"])

        r_norms = np.sqrt(x**2 + y**2 + z**2)
        v_norms = np.sqrt(vx**2 + vy**2 + vz**2)

        phi_j2 = (MU_EARTH * J2_EARTH * (R_EARTH**2) / (2.0 * r_norms**3)) * (3.0 * (z / r_norms) ** 2 - 1.0)
        E_tot = 0.5 * v_norms**2 - MU_EARTH / r_norms + phi_j2

        # Net mechanical energy must decrease (non-conservative dissipation)
        energy_loss = E_tot[0] - E_tot[-1]
        assert energy_loss > 1000.0, f"Expected significant drag energy loss, got {energy_loss} J/kg"


class TestSecurityAndAuxiliaryEndpoints:
    """Security and auxiliary endpoint adversarial edge cases."""

    @pytest.mark.parametrize("bad_name", ["..%2F..%2Fetc%2Fpasswd", "test%00evil", "invalid/slash", "has space", "foo..bar"])
    def test_path_traversal_and_bad_filename_rejected(self, bad_name):
        """Assert /save and /load reject directory traversal and illegal characters with HTTP 400."""
        res_save = client.post("/save", json={"filename": bad_name, "satellites": []})
        assert res_save.status_code == 400

        res_load = client.get(f"/load/{bad_name}")
        assert res_load.status_code in (400, 404)

    @pytest.mark.parametrize(
        "dt,n,expected_ok",
        [
            (0.001, 1, True),
            (86400.0, 20000, True),
            (0.0, 10, False),
            (-5.0, 10, False),
            (86400.1, 10, False),
            (60.0, 0, False),
            (60.0, -1, False),
            (60.0, 20001, False),
        ],
    )
    def test_moon_track_parameter_boundaries(self, dt, n, expected_ok):
        """Assert /moon_track enforces dt in (0, 86400] and n in [1, 20000]."""
        res = client.get(f"/moon_track?dt={dt}&n={n}")
        if expected_ok:
            assert res.status_code == 200
            data = res.json()
            assert isinstance(data, list)
            assert len(data) == n
        else:
            data = res.json()
            assert "error" in data or res.status_code == 400
