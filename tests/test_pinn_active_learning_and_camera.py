"""Unit tests for active learning hard-negative mining, multi-orbit rollout metrics, and camera WebSocket streaming."""

import json
import os
from fastapi.testclient import TestClient
import numpy as np
import pytest

from ai.evaluate import evaluate_multi_orbit_rollout, mine_hard_negatives
from visualization_3d.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_multi_orbit_rollout_metrics():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "ai", "checkpoints", "agent", "transition.pt")
    if not os.path.exists(ckpt_path):
        pytest.skip("transition.pt checkpoint not present")

    res = evaluate_multi_orbit_rollout(ckpt_path, orbit_counts=(0.2, 0.5), alt_km=420.0, inc_deg=51.6)
    assert 0.2 in res
    assert 0.5 in res
    assert "final_drift_pct" in res[0.2]
    assert "mean_drift_pct" in res[0.2]
    assert res[0.2]["max_drift_pct"] >= 0.0


def test_mine_hard_negatives(tmp_path):
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "ai", "checkpoints", "agent", "transition.pt")
    if not os.path.exists(ckpt_path):
        pytest.skip("transition.pt checkpoint not present")

    out_file = str(tmp_path / "hard_negatives.npz")
    # Set threshold low so that failing steps are captured and mined
    res = mine_hard_negatives(ckpt_path, n_candidates=2, threshold_pct=0.01, out_path=out_file)

    assert "mined_count" in res
    assert "total_evaluated" in res
    assert res["total_evaluated"] == 2
    if res["mined_count"] > 0:
        assert os.path.exists(out_file)
        data = np.load(out_file)
        assert "x" in data and "y" in data
        assert data["x"].shape[1] == 6


def test_camera_feed_websocket_broadcast(client):
    with client.websocket_connect("/ws/camera-feed") as ws_sender:
        with client.websocket_connect("/ws/camera-feed") as ws_receiver:
            payload = {
                "type": "camera_frame",
                "sat_id": "sat-demo",
                "mode": "nadir",
                "frame": "data:image/jpeg;base64,sample_frame",
                "timestamp": 12345678,
            }
            ws_sender.send_text(json.dumps(payload))
            received_raw = ws_receiver.receive_text()
            received_json = json.loads(received_raw)
            assert received_json["type"] == "camera_frame"
            assert received_json["sat_id"] == "sat-demo"
            assert received_json["mode"] == "nadir"
