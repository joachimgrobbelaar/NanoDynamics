# Nanodynamics-v3 System Architecture & TRL 1–9 Progression

## 1. End-to-End Data & Computation Pipeline

```mermaid
flowchart LR
    subgraph Empirical["1. Empirical Telemetry (ESA / NASA)"]
        SP3["NASA CDDIS .SP3\n(GRACE-FO / GNSS)"]
        EOF["ESA Sentinel .EOF\n(AUX_POEORB / Swarm)"]
        SPICE["NAIF SPICE .bsp\n(DE440s Ephemeris)"]
        SW["NOAA / CelesTrak\n(F10.7, Kp, Ap)"]
    end

    subgraph Astro["2. Astrodynamics & Frames (src/astrodynamics)"]
        FRAMES["IAU 2006/2000A\nITRF <-> GCRF + Coriolis"]
        FORCES["Force Models\nJ2-J4 + NRLMSISE-00 + SRP + CR3BP"]
        PROP["Ground-Truth Propagators\nDOP853 / TudatPy / Basilisk"]
    end

    subgraph ML["3. Physics-Informed ML & Quantization (src/ml & src/export)"]
        CANON["Canonical Scaling\n(R_E, v_scale, tau_scale)"]
        PINN["Autograd Residual MLP\n+ Hamiltonian Loss"]
        TFLM["INT8 Quantizer\ntransition_model_data.h"]
    end

    subgraph Edge["4. Dual-ESP32 FreeRTOS HIL Mesh (firmware/)"]
        GW["ESP32 Node 1 Gateway\nCore 0: Wi-Fi/WS/OTA\nCore 1: ESP-NOW MAC"]
        INF["ESP32 Node 2 Inference\nCore 1: TFLM INT8\nCore 0: Symplectic Projection"]
    end

    SP3 & EOF & SPICE --> FRAMES --> PROP
    SW --> FORCES --> PROP
    PROP --> CANON --> PINN --> TFLM --> INF
    GW <-->|"ESP-NOW Ch1 (<2ms)"| INF
```

## 2. NASA / ESA TRL 1–9 Modularity Matrix
- **TRL 1–3**: Analytical force models (`src/astrodynamics/force_models.py`), IAU frame transformations (`src/astrodynamics/frames.py`), and Hamiltonian manifold projection.
- **TRL 4–5**: Breadboard validation against empirical ESA Copernicus `.EOF` and NASA `.SP3` flight arcs (`src/ingestion/`).
- **TRL 6**: Hardware-in-the-Loop (HIL) execution on Dual-ESP32 microcontrollers over deterministic 42-byte `OrbitStatePacket` frames.
- **TRL 7–9**: Pluggable avionics bus interface (`CCSDS 133.0-B-2 Space Packet`, UART/RS-422, SPI, CAN) with static arena allocation (`32 KB`, zero heap allocation) and automated FDIR fallback to symplectic Verlet integration if $|\Delta \mathcal{H} / \mathcal{H}_0| > 10^{-5}$.
