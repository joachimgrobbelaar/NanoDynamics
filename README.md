# Nanodynamics-v3: Empirical Flight Telemetry, Multi-Body Astrodynamics & Dual-ESP32 Symplectic PINN Framework

[![TRL 1-9 Modular](https://img.shields.io/badge/NASA%2FESA-TRL%201--9%20Modular-00FF66)](#)
[![Astrodynamics](https://img.shields.io/badge/Physics-J2--J4%20%7C%20NRLMSISE--00%20%7C%20SRP%20%7C%20CR3BP-00E5FF)](#)
[![Hardware-in-the-Loop](https://img.shields.io/badge/Edge%20AI-Dual--ESP32%20TFLM%20INT8-FFD21E)](#)

`Nanodynamics-v3` is an open-source, flight-grade framework bridging high-precision empirical astrodynamics (**ESA Copernicus Sentinel `.EOF`**, **NASA CDDIS `.SP3`**, **NAIF SPICE `.bsp`**) with real-time **Symplectic Physics-Informed Neural Network (PINN)** surrogates deployed on **Dual-ESP32 FreeRTOS** microcontrollers.

---

## 1. Repository Architecture

```text
Nanodynamics-v3/
├── .github/workflows/          # Automated CI.yml & numerical-vs-PINN benchmark.yml
├── config/                     # Mission regimes, Space Weather (F10.7/Kp), & Dual-ESP32 YAML configs
├── data/
│   ├── raw/                    # Bundled NASA SP3 (Swarm-A/GRACE-FO), ESA Sentinel-1A .EOF & SPICE kernels
│   ├── processed/              # Canonical non-dimensional [r*, v*, p_k] training arrays
│   └── space_weather/          # Historical NOAA SWPC / CelesTrak solar & geomagnetic indices
├── docs/                       # System architecture, LaTeX mathematical formulation & HIL guide
├── firmware/
│   ├── node1_gateway/          # ESP32 Node 1: Wi-Fi AP+STA, WebSocket (/ws/edge), NVS portal & ESP-NOW
│   └── node2_inference/        # ESP32 Node 2: TFLite Micro INT8 runtime + Symplectic Hamiltonian projector
├── src/
│   ├── ingestion/              # sp3_parser.py, eof_parser.py, spice_loader.py
│   ├── astrodynamics/          # force_models.py (J2-J4, NRLMSISE-00, SRP, CR3BP), propagators.py, frames.py
│   ├── ml/                     # dataset_generator.py, pinn_loss.py, trainer.py
│   └── export/                 # tflite_quantizer.py (INT8 quantizer & C-header emitter)
├── tests/                      # test_ingestion.py, test_propagator.py, test_hamiltonian.py
└── web_console/                # FastAPI WebSocket server + Three.js Rayleigh/Mie 3D Telemetry Cockpit
```

---

## 2. Quickstart & Verification

### Run the Automated Verification Suite
```bash
PYTHONPATH=. pytest -v tests/
```

### Launch the 3D WebGL & Dual-ESP32 Flight-Director Console
```bash
PYTHONPATH=. uvicorn web_console.server:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` to inspect live **ESA Sentinel-1A / NASA Swarm-A** empirical arcs, **DOP853 vs. Symplectic PINN** 3D trajectory ribbons, **CR3BP Cislunar** trajectories, and **Dual-ESP32 HIL** latency sparklines.
