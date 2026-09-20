# AI surrogate pipeline: simulator → dataset → NN → ESP32

Ground truth always comes from `leo_simulator` (solve_ivp). The networks
learn from it and are scored against it — never the other way round.

## 1. Collect data (automated sweep)

```bash
python3 ai/generate_dataset.py --mode one_step --n-traj 16 --orbits 1 --dt 60 --seed 1
python3 ai/generate_dataset.py --mode trajectory --n-traj 20 --orbits 1 --dt 60  # whole-orbit PINN variant
```

Writes `ai/data/dataset[_onestep].npz` + `stats[_onestep].json` (scales,
seed, physics config). Re-runnable and seeded for reproducibility.

## 2. Train locally

```bash
python3 ai/train_transition.py --epochs 600 --hidden 64 --layers 3   # recommended: small residual net
python3 ai/train.py --epochs 400                                     # whole-trajectory PINN variant
```

Checkpoints (+ TorchScript trace for the transition net) land in
`ai/checkpoints/`. The 64×3 transition net is 9,158 params (~36 KB fp32).

## 3. Compare against simulation

```bash
python3 ai/evaluate.py --model transition --alt-km 420 --inc-deg 60
```

Rolls the net step-by-step on a held-out orbit and reports drift % vs
solve_ivp. Same harness covers the PINN (`--model pinn`).

## 4. ESP32 (TinyML) path

The transition net is the deployable one: fixed 6-float input, no
autograd, ~36 KB fp32 (~9 KB int8). Recipe:

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
