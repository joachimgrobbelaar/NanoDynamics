# Dual-ESP32 Hardware Deployment & TFLM Integration Guide

## 1. Hardware Architecture
`Nanodynamics-v3` partitions communication and neural inference across two ESP32 microcontrollers connected via raw 802.11 ESP-NOW frames on Channel 1:

| Node | Firmware Sketch | Core 0 Task | Core 1 Task | SRAM / Flash Budget |
| :--- | :--- | :--- | :--- | :--- |
| **Node 1 (Gateway)** | `firmware/node1_gateway/node1_gateway.ino` | Wi-Fi SoftAP + STA, WebSocket (`/ws/edge`), HTTP NVS Provisioning (`:80`), ArduinoOTA | ESP-NOW Raw MAC Transceiver (`< 2 ms` RTT) | 49 KB RAM / 843 KB Flash |
| **Node 2 (Inference)** | `firmware/node2_inference/node2_inference.ino` | Symplectic Hamiltonian Manifold Projector (`hamiltonian.h`) + FDIR | TFLite Micro INT8 Interpreter (`transition_model_data.h`, 32 KB Arena) | 85 KB RAM / 807 KB Flash |

## 2. Quantizing & Exporting a New PINN Checkpoint
```bash
PYTHONPATH=. python3 -m src.export.tflite_quantizer \
  --output firmware/node2_inference/transition_model_data.h
```

## 3. Flashing Firmware & Provisioning Wi-Fi Credentials
1. Flash **Node 1 Gateway** (`/dev/ttyUSB0`) and **Node 2 Inference** (`/dev/ttyUSB1`).
2. Connect to SoftAP `NANODYNAMICS-ESP32-GW` (`192.168.4.1`) or use the Web Console (`http://localhost:8000`) to push your local Wi-Fi SSID, password, and simulator Host IP to NVS flash via `WIFI_SET:<ssid>|<pass>|<host_ip>`.
