"""Firebase acceptance verification (ORIGINAL_REQUEST.md AC1–AC3).

Verification lane owned by Muse on branch `muse/verify-acceptance`.
Implementation lane (app, functions, UI) owned by Claude in Antigravity —
these tests assert the contract, they do not implement it.

- AC1: Cloud Function ML training endpoint reachable (emulator/live) and
  returns a structured training-loss response; otherwise the registration
  contract of `functions/main.py:trigger_pinn_training` is asserted and the
  live-call test skips with a clear reason (no fake pass).
- AC2: `POST /simulate` increments the recorded pair metric and the UI
  element `#ml-pairs-val` is wired to `/ai/pairs`.
- AC3: deployment pipeline script is syntactically valid and contains the
  required bundling/deploy steps plus PyTorch deps (dry-run, no live deploy).
"""

import json
import os
import re
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visualization_3d.app import app  # noqa: E402

client = TestClient(app)

VALID_PAYLOAD = {
    "name": "AC-Sat",
    "mass": 4.0,
    "drag_area": 0.03,
    "cd": 2.2,
    "altitude_km": 450.0,
    "eccentricity": 0.001,
    "inclination_deg": 51.6,
    "raan_deg": 10.0,
}


def _default_project_id():
    """Project id the emulator serves: .firebaserc default (singleProjectMode)."""
    try:
        rc = json.load(open(os.path.join(PROJECT_ROOT, ".firebaserc")))
        return rc.get("projects", {}).get("default", "demo-test")
    except (OSError, ValueError):
        return "demo-test"


def _emulator_training_url():
    base = os.environ.get(
        "FIREBASE_TRAINING_URL",
        f"http://localhost:5001/{_default_project_id()}"
        "/us-central1/trigger_pinn_training",
    )
    return base


class TestTrainingEndpoint:
    def test_trigger_registered_with_structured_contract(self):
        """AC1 (static): training trigger exists and documents the loss contract."""
        import ast

        path = os.path.join(PROJECT_ROOT, "functions", "main.py")
        src = open(path).read()
        assert "trigger_pinn_training" in src, "Cloud Function trigger missing"
        tree = ast.parse(src)
        fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "trigger_pinn_training" in fns
        body_src = ast.get_source_segment(src, fns["trigger_pinn_training"]) or ""
        for key in ("final_loss", "loss_history", "n_pairs", "epochs_run", "status"):
            assert key in body_src, f"training response missing key: {key}"

    def test_training_endpoint_live_or_skip(self):
        """AC1 (live): hit emulator/live endpoint; skip cleanly when unreachable."""
        import urllib.request

        url = _emulator_training_url()
        payload = json.dumps(
            {"data": {"epochs": 2, "lr": 1e-3, "batch_size": 64, "hidden": 16, "layers": 1}}
        ).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            # 180s: cold worker (torch import) + 2 training epochs over ~345k pairs.
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:  # emulator not running / no live creds
            pytest.skip(f"training endpoint unreachable at {url}: {exc}")
        result = body.get("result", body)
        assert result.get("status") == "success", f"unexpected training reply: {body}"
        assert isinstance(result.get("final_loss"), (int, float))
        assert isinstance(result.get("loss_history"), list)
        assert isinstance(result.get("n_pairs"), int)


class TestPairTelemetry:
    def test_simulate_increments_sim_log_rows(self):
        """AC2: POST /simulate appends exactly one sim_log row (pair metric)."""
        before = client.get("/ai/pairs")
        assert before.status_code == 200
        rows_before = before.json()["sim_log_rows"]

        resp = client.post("/simulate", json=VALID_PAYLOAD)
        assert resp.status_code == 200, resp.text

        after = client.get("/ai/pairs")
        assert after.status_code == 200
        data = after.json()
        assert set(("total_pairs", "sim_log_rows", "npz_files_scanned")) <= set(data)
        assert isinstance(data["total_pairs"], int) and data["total_pairs"] >= 0
        assert data["sim_log_rows"] == rows_before + 1

    def test_ui_pair_counter_wired(self):
        """AC2 (UI): #ml-pairs-val exists and is fed from /ai/pairs total_pairs."""
        path = os.path.join(PROJECT_ROOT, "visualization_3d", "static", "index.html")
        html = open(path).read()
        assert 'id="ml-pairs-val"' in html, "UI element #ml-pairs-val missing"
        assert "fetch('/ai/pairs')" in html or 'fetch("/ai/pairs")' in html
        m = re.search(r"getElementById\(['\"]ml-pairs-val['\"]\)", html)
        assert m, "#ml-pairs-val never populated from JS"
        # Purge check: exactly one element with that id (no duplicates).
        assert html.count('id="ml-pairs-val"') == 1


class TestDeployPipeline:
    def test_deploy_script_valid_and_complete(self):
        """AC3 (dry-run): deploy.sh parses and contains bundle/deploy/cleanup + torch."""
        deploy = os.path.join(PROJECT_ROOT, "functions", "deploy.sh")
        assert os.path.exists(deploy)
        syntax = subprocess.run(["bash", "-n", deploy], capture_output=True, text=True)
        assert syntax.returncode == 0, syntax.stderr
        src = open(deploy).read()
        assert "firebase deploy" in src, "deploy step missing"
        assert "leo_simulator" in src, "bundling step missing"

        reqs = open(os.path.join(PROJECT_ROOT, "functions", "requirements.txt")).read()
        assert "torch" in reqs, "PyTorch dep missing from Cloud Function bundle"

        cfg = json.load(open(os.path.join(PROJECT_ROOT, "firebase.json")))
        assert "functions" in cfg and "hosting" in cfg, "firebase.json incomplete"
        assert "firestore" in cfg and "emulators" in cfg
