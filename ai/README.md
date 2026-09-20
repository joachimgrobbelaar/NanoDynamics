# AI surrogate pipeline: simulator → dataset → NN → ESP32

Ground truth always comes from `leo_simulator` (solve_ivp). The networks
learn from it and are scored against it — never the other way round.

## 1. Collect data — two feeds

**A. Your simulations (automatic).** Every `POST /simulate` appends one
line to `ai/data/sim_log.jsonl` (best-effort, never breaks the response);
`Save` in the web UI writes `visualization_3d/saved_sims/*.json`.
`ai/ingest.py` unifies all of it — both saved-sim formats,
`trajectory.json`, the sim log, plus `--csv` exports:

```bash
python3 ai/ingest.py --csv ~/Downloads/trajectory_export.csv
```

This validates each run (finite values, above-surface, monotonic time),
splits glued/non-uniform tracks into uniform-dt segments, dedups by
content hash, and writes `ai/data/sim_pairs.npz` + `sim_manifest.json`
with per-run physics provenance (atmosphere, density_scale, J2/drag/moon
flags). Only segments at `--dt` (default 60 s) become pairs; the rest are
reported, not silently dropped. Rebuilds are deterministic — re-running
never doubles data.

**B. Synthetic sweep (backfill).** Covers orbit space your clicks haven't:

```bash
python3 ai/generate_dataset.py --mode one_step --n-traj 16 --orbits 1 --dt 60 --seed 1
python3 ai/generate_dataset.py --mode trajectory --n-traj 20 --orbits 1 --dt 60  # whole-orbit PINN variant
```

## 2. Local ML agent (recommended)

One command runs the whole loop — ingest → retrain on new data only →
rollout vs simulation → `ai/data/agent_report.json`:

```bash
python3 ai/agent.py --once
python3 ai/agent.py --watch --interval 300   # keep running: picks up new sims as you fly them
```

Checkpoints land in `ai/checkpoints/agent/` (never clobbers manual runs).
The agent trains Earth-compatible runs only — explicit non-Earth bodies
(e.g. lunar orbits, different mu) are excluded and listed, never mixed.
Tune with `--epochs/--hidden/--layers/--min-pairs/--alt-km/--inc-deg`.

Manual equivalents: `ai/train_transition.py` (small residual net),
`ai/train.py` (whole-trajectory PINN), `ai/evaluate.py --model
transition --alt-km 420 --inc-deg 60` (drift % vs solve_ivp).

## 3. ESP32 (TinyML) path — after local works

The transition net is the deployable one: fixed 6-float input, no
autograd, ~36 KB fp32 (~9 KB int8). Only start this once the local
rollout passes the 5% check above.

1. `pip install onnx tf2onnx tensorflow` (build machine, not ESP32).
2. `torch.onnx.export` the traced net → `onnx2tf` → TFLite, then
   post-training **int8 quantization** with a calibration sample from
   `dataset_onestep.npz` (fully-quantized int8 runs fastest on ESP32).
3. Deploy with **TensorFlow Lite Micro** (ESP-IDF component or Arduino
   `Arduino_TensorFlowLite`): one inference per tick advances the orbit
   by `dt` seconds; compare on-device predictions against a downlinked
   `solve_ivp` reference exactly like `evaluate.py` does.
4. Budgets to respect: 520 KB SRAM total (keep arena < 200 KB),
   float32 quant drift re-check after conversion, and keep `dt` fixed —
   the net learned one specific step size (`stats_onestep.json`).

Rule of thumb from this repo: whole-trajectory PINNs need lots of orbits
to generalize; the residual one-step map learns orbital flow (including
J2 + drag) from far less data and fits the microcontroller.
