# Dual-ESP32 Hardware-in-the-Loop (HIL) Deployment Guide

This guide documents the hardware architecture, wiring specifications, power consumption profile, and flashing procedures for the Nanodynamics v3 dual-ESP32 orbital state prediction and telemetry testbed.

---

## 1. System Architecture Overview

```
                          FASTAPI GROUND SERVER (:8000)
                                       │
                      Wi-Fi / WebSocket (:80 /ws/hil-worker)
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ ESP32 NODE 1: NETWORK GATEWAY & TELEMETRY BROKER (Port: /dev/ttyUSB0)│
    │  - Core 0: Wi-Fi STA/AP + WebSocket Server + ArduinoOTA + WDT (4s) │
    │  - Core 1: ESP-NOW Transceiver locked to Wi-Fi Channel 1            │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │
                         ESP-NOW 2.4 GHz (Channel 1)
                       Packed 64-byte C-Struct (<2 ms)
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ ESP32 NODE 2: DEDICATED PINN INFERENCE ENGINE (Port: /dev/ttyUSB1)  │
    │  - Wi-Fi/Bluetooth OFF upfront to preserve 40 KB Tensor Arena       │
    │  - Core 1: TFLM INT8 TransitionMLP Forward Inference (~1.85 ms)     │
    │  - Core 0: Symplectic Hamiltonian Manifold Projection (Energy Cons) │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pinout & Hardware Wiring Diagram

Both nodes utilize standard ESP32-WROOM-32 dev boards (38-pin or 30-pin variants).

### Pin Mapping

| Function | Node 1 (Gateway) | Node 2 (Inference) | Description |
| :--- | :--- | :--- | :--- |
| **Status LED** | `GPIO 2` | `GPIO 2` | Active-high onboard blue LED (blinks on packet infer/tx) |
| **UART TXD** | `GPIO 1 (TX0)` | `GPIO 1 (TX0)` | Primary serial transmit to host PC (`115200 baud`) |
| **UART RXD** | `GPIO 3 (RX0)` | `GPIO 3 (RX0)` | Primary serial receive from host PC (`115200 baud`) |
| **Logic Supply** | `3V3` | `3V3` | Regulated 3.3V DC power rail |
| **Ground** | `GND` | `GND` | Common ground reference across USB hosts |

### Physical Benchtop Interconnect

```
         HOST WORKSTATION / JETSON
     ┌───────────────────────────────┐
     │  USB Port 0      USB Port 1   │
     └───┬───────────────────┬───────┘
         │ (/dev/ttyUSB0)    │ (/dev/ttyUSB1)
         ▼                   ▼
    ┌─────────────┐     ┌─────────────┐
    │ ESP32 NODE 1│     │ ESP32 NODE 2│
    │   Gateway   │     │  Inference  │
    │             │     │             │
    │      GND ───┴─────┴─── GND      │  <-- Common ground loop
    └─────────────┘     └─────────────┘
          ) ) )               ( ( (
          ESP-NOW Channel 1 (2.4 GHz)
```

---

## 3. Power Consumption & Energy Profile (Sub-milliJoule Execution)

The architecture is explicitly optimized for ultra-low power orbital edge computing (< 1.0 mJ per propagation step).

### Power Measurements at 3.3V DC

| State | Node 1 (Gateway) | Node 2 (Inference) | Combined Current |
| :--- | :--- | :--- | :--- |
| **Idle / Listening** | 65 mA (Wi-Fi beacon) | 22 mA (Wi-Fi OFF, BT OFF) | 87 mA (287 mW) |
| **Peak Active Inference** | 82 mA (ESP-NOW TX) | 78 mA (Dual Core 240 MHz) | 160 mA (528 mW) |
| **Cycle Duration** | 0.40 ms (TX/RX frame) | 1.85 ms (TFLM + Symplectic) | **1.85 ms total** |

### Per-Cycle Energy Calculation

$$\text{Energy per Step} = V_{\text{bus}} \times I_{\text{active}} \times \Delta t_{\text{active}}$$

For Node 2 dedicated inference:
$$E_{\text{node2}} = 3.3\,\text{V} \times 78\,\text{mA} \times 1.85\,\text{ms} = 0.476\,\text{mJ}$$

For Node 1 ESP-NOW framing:
$$E_{\text{node1}} = 3.3\,\text{V} \times 82\,\text{mA} \times 0.40\,\text{ms} = 0.108\,\text{mJ}$$

$$\mathbf{E}_{\text{total}} = 0.476\,\text{mJ} + 0.108\,\text{mJ} = \mathbf{0.584\,\text{mJ}} \quad (< 1.0\,\text{mJ})$$

---

## 4. Flashing & Deployment Instructions

### Prerequisites

Install `platformio` CLI or use `esptool.py`:
```bash
pip install platformio esptool
```

### Option A: Flashing via PlatformIO (Recommended)

1. **Flash Node 1 (Gateway):**
   Connect Node 1 to `/dev/ttyUSB0`:
   ```bash
   cd firmware/node1_gateway
   pio run --target upload --upload-port /dev/ttyUSB0
   ```

2. **Flash Node 2 (Inference Engine):**
   Connect Node 2 to `/dev/ttyUSB1`:
   ```bash
   cd firmware/node2_inference
   pio run --target upload --upload-port /dev/ttyUSB1
   ```

### Option B: Flashing via Arduino IDE

1. In Arduino IDE `Preferences`, add the ESP32 board URL:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
2. Install board package `esp32 by Espressif Systems` (version `>= 2.0.14`).
3. Install library `tflite-micro` (v1.0.3) in the Library Manager.
4. Select `ESP32 Dev Module`, CPU Frequency: `240MHz`, Flash Frequency: `80MHz`.
5. Open `firmware/node1_gateway/node1_gateway.ino`, select `/dev/ttyUSB0`, and click **Upload**.
6. Open `firmware/node2_inference/node2_inference.ino`, select `/dev/ttyUSB1`, and click **Upload**.

---

## 5. Verification & Telemetry Health Check

Verify the hardware link from the command line:

```bash
# Query Node 1 Gateway status over serial
python3 -c "
import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=1.0)
s.write(b'STATUS?\n')
print('Gateway:', s.readline().decode().strip())
s.write(b'INFER:6878137,0,0,0,7612,0,60,0.015\n')
print('Inference:', s.readline().decode().strip())
s.close()
"
```

Expected output:
```json
Gateway: {"role":"gateway","ch":1,"tx":42,"rx":42,"rtt_ms":1.82}
Inference: RESULT:6878120.450120,456720.128400,0.000000,-506.120000,7595.230000,0.000000,1.850
```
