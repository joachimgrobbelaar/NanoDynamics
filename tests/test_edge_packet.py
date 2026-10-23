"""Programmatic verification for Track 3: Dual-ESP32 EdgeLink Packet Contract & /ws/edge Bridge.

Verifies:
1. 42-byte packed binary serialization of OrbitStatePacket (<H10f).
2. Symplectic Hamiltonian Manifold projection conservation of target specific mechanical energy.
3. Bidirectional binary and JSON streaming over /ws/edge FastAPI WebSocket endpoint.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from leo_simulator.constants import MU_EARTH, R_EARTH
from visualization_3d.app import (
    ORBIT_PACKET_SIZE,
    OrbitStatePacketModel,
    app,
    symplectic_hamiltonian_project,
)


def test_orbit_state_packet_binary_contract():
    """Assert packed C-struct OrbitStatePacket is exactly 42 bytes and round-trips losslessly."""
    assert ORBIT_PACKET_SIZE == 42
    pkt = OrbitStatePacketModel(
        sat_id=42,
        t=120.0,
        x=6778137.0,
        y=0.0,
        z=0.0,
        vx=0.0,
        vy=7668.6,
        vz=0.0,
        B_coeff=0.0165,
        dt=10.0,
        t_infer_ms=1.72,
    )
    raw = pkt.to_bytes()
    assert len(raw) == 42

    unpacked = OrbitStatePacketModel.from_bytes(raw)
    assert unpacked.sat_id == 42
    assert unpacked.t == pytest.approx(120.0, rel=1e-6)
    assert unpacked.x == pytest.approx(6778137.0, rel=1e-6)
    assert unpacked.vy == pytest.approx(7668.6, rel=1e-5)
    assert unpacked.t_infer_ms == pytest.approx(1.72, rel=1e-5)


def test_symplectic_hamiltonian_manifold_projection():
    """Assert symplectic Hamiltonian projection restores exact target mechanical energy E_target."""
    r = np.array([R_EARTH + 400_000.0, 100_000.0, -50_000.0], dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    v_circ = np.sqrt(MU_EARTH / r_norm)
    v_true = np.array([0.0, v_circ, 120.0], dtype=np.float64)
    e_target = 0.5 * float(np.dot(v_true, v_true)) - (MU_EARTH / r_norm)

    # Perturb neural network velocity prediction by +3.5% energy drift
    v_nn_perturbed = v_true * 1.035
    v_proj = symplectic_hamiltonian_project(r, v_nn_perturbed, e_target, mu=MU_EARTH)

    e_after = 0.5 * float(np.dot(v_proj, v_proj)) - (MU_EARTH / r_norm)
    assert abs((e_after - e_target) / e_target) < 1e-12


def test_ws_edge_websocket_binary_and_json_roundtrip():
    """Verify /ws/edge handles both 42-byte binary ESP-NOW packets and JSON payloads."""
    client = TestClient(app)
    r0 = R_EARTH + 500_000.0
    v0 = float(np.sqrt(MU_EARTH / r0))
    e0 = 0.5 * (v0 ** 2) - (MU_EARTH / r0)

    pkt_in = OrbitStatePacketModel(
        sat_id=7,
        t=0.0,
        x=r0,
        y=0.0,
        z=0.0,
        vx=0.0,
        vy=v0,
        vz=0.0,
        B_coeff=0.015,
        dt=10.0,
        t_infer_ms=1.64,
    )

    with client.websocket_connect("/ws/edge") as ws:
        # 1. Binary packet test
        ws.send_bytes(pkt_in.to_bytes())
        resp_bytes = ws.receive_bytes()
        assert len(resp_bytes) == 42
        pkt_out = OrbitStatePacketModel.from_bytes(resp_bytes)
        assert pkt_out.sat_id == 7
        assert pkt_out.t == pytest.approx(10.0)
        r1 = np.linalg.norm([pkt_out.x, pkt_out.y, pkt_out.z])
        v1_sq = pkt_out.vx**2 + pkt_out.vy**2 + pkt_out.vz**2
        e1 = 0.5 * v1_sq - (MU_EARTH / r1)
        # Float32 serialization preserves energy within ~1e-6
        assert abs((e1 - e0) / e0) < 1e-5

        # 2. JSON packet test
        ws.send_text(pkt_in.model_dump_json())
        resp_json = ws.receive_json()
        assert resp_json["sat_id"] == 7
        assert resp_json["t"] == pytest.approx(10.0)
