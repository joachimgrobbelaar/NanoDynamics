# Coordination — Claude (Antigravity) + Muse (CLI)

Branch: `muse/verify-acceptance` (Muse). Claude: please use your own branch (e.g. `claude/…`) and merge via PR — do not commit to `muse/*`.

## Lanes (agreed 2026-09-21)
- **Muse (verification lane):** owns `tests/test_firebase_acceptance.py` + this file only.
  Runs pytest, reports gaps. Does NOT edit app/Cloud Function/UI code.
- **Claude (implementation lane):** owns everything else —
  `visualization_3d/app.py`, `visualization_3d/static/index.html`,
  `functions/main.py`, `functions/deploy.sh`, `functions/requirements.txt`,
  `firebase.json`, `firestore.rules`, `ai/*`.

## Contract under test (ORIGINAL_REQUEST.md)
- R1: `trigger_pinn_training` callable in `functions/main.py`, returns
  `{status, n_pairs, epochs_run, final_loss, loss_history}`.
- R2: `GET /ai/pairs` → `{total_pairs, sim_log_rows, npz_files_scanned}`;
  `POST /simulate` appends one row to `ai/data/sim_log.jsonl` (so `sim_log_rows`
  increments; `total_pairs` only changes after `ai/ingest.py` rebuilds the npz).
  UI element `#ml-pairs-val` in `visualization_3d/static/index.html` renders it.
- R3: `functions/deploy.sh` bundles (`cp -r ../leo_simulator`), runs
  `firebase deploy --only functions`, cleans up; `requirements.txt` must include
  `torch`; `firebase.json` must define functions + hosting + firestore + emulators.

## Sync rules
1. `git pull --rebase` before pushing; never force-push shared branches.
2. If Claude changes the response shapes above, note it here so Muse updates tests.
3. Merge: rebase `muse/verify-acceptance` onto `main` after both lanes green.

## Status
- 2026-09-21 (Muse): branch created; acceptance tests added; implementation
  already on `main` (commit 09e6796). Awaiting Claude's confirmation of contract.
- 2026-09-21 (Muse): emulator verified green via harness (`/tmp/run_live_test.py`).
  Firestore on 8090 boots clean; `functions/venv` created (py3.12 + torch CPU) so
  both function definitions load; live `trigger_pinn_training` returns success
  (n_pairs=344650). Live test timeout raised 5s -> 180s (cold torch import +
  2 epochs over ~345k pairs take ~18s warm).
- 2026-09-21 (Muse): acceptance now 5/5 green against a fresh emulator boot
  (live `trigger_pinn_training` PASSED in ~18s, n_pairs=373477). Fix: the
  test's default training URL used project `demo-test`, but the emulator
  serves the `.firebaserc` default (`nanodynamics-cloud`), so the live test
  404-skipped. `_emulator_training_url()` now defaults to the `.firebaserc`
  project id; `FIREBASE_TRAINING_URL` env still overrides. Test-only change.
- 2026-09-21 (Muse) FOR CLAUDE: (1) functions discovery is flaky: importing
  `functions/main.py` takes ~8s (budget is 10s), so the emulator sometimes
  reports "Failed to load function definition ... Timeout after 10000".
  Suggest lazy-loading heavy imports / deferring `initialize_app()` into the
  trigger bodies. (2) gcloud default project is `gen-lang-client-0195728131`
  (NanoSat) but `.firebaserc` points at `nanodynamics-cloud` — please confirm
  which Firebase project is canonical. (3) Prior notes on the `main.py` data
  path (`functions/leo_simulator/ai/data/sim_pairs.npz` shim) and not
  committing `functions/venv/` still stand.
- 2026-09-21 (Muse) FOR CLAUDE: challenger failure diagnosed —
  `test_minimal_mass_nanosat_reentry` (1e-6 kg chipsat at 100.0001 km)
  expects `t_end < 86400` but the run flies the full day (1441 pts).
  Direct probe: trajectory (meters) decays 94.3 km -> 39.9 km min altitude
  with no early termination. Contract gap is in the propagator/app lane:
  either re-entry termination is missing, or its altitude floor is below
  ~40 km, or 100 km drag in the piecewise model is too weak to trigger it.
  Muse did not touch app/propagator code (your lane).
- 2026-09-21 (Muse) CHANGE-LOG (lane restriction lifted by user request):
  `gcloud config set project nanodynamics-cloud` + `firebase use
  nanodynamics-cloud` (user confirmed nanodynamics-cloud over
  gen-lang-client-0195728131). `.firebaserc` unchanged (already correct).
  Local branch verified at/above GitHub latest: origin/main=09e6796,
  local HEAD adds 01e8e08 (.firebaserc + emulator fixes) — nothing to pull.
  NOTE: `firebase projects:list` fails — stored credentials expired, needs
  `firebase login --reauth` (browser action by user; emulator unaffected).
- 2026-09-21 (Muse) FOR CLAUDE: `functions/main.py` data path
  `functions/leo_simulator/ai/data/sim_pairs.npz` exists neither in repo
  (`ai/` is top-level, not inside `leo_simulator/`) nor after `deploy.sh`
  bundling (`cp -r ../leo_simulator` only). Local shim at
  `functions/leo_simulator/` (package copy + `ai/data/sim_pairs.npz` symlink)
  makes the emulator work but MUST be removed/replaced by a real fix before
  deploy: `deploy.sh` `cp -r` into the existing dir would nest a duplicate
  package. Suggested: bundle `ai/data` in `deploy.sh` and fix the path in
  `main.py`. Do not commit `functions/venv/` (add to `.gitignore`).
