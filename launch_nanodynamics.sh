#!/usr/bin/env bash
# NanoDynamics Mission Control & ESP32 PINN HIL Portal Launcher
# Starts the FastAPI/Uvicorn backend (if not already running on :8000) and opens both:
#   1. 3D SpaceX-Style Orbital Simulator (http://localhost:8000/)
#   2. Real-Time ESP32 PINN Telemetry & Wi-Fi Provisioning Web App (http://localhost:8000/edge-console)

set -e

SIM_DIR="/home/m/Projects/misc/teamwork_projects/leo_simulator_v2"
PORT=8000

echo "=================================================================="
echo "  NANODYNAMICS // 3D ASTRODYNAMICS & DUAL-ESP32 PINN HIL PORTAL"
echo "=================================================================="

if ! ss -tln | grep -q ":${PORT} "; then
    echo "[+] Starting NanoDynamics FastAPI + ESP32 HIL Server on port ${PORT}..."
    cd "${SIM_DIR}"
    PYTHONPATH="${SIM_DIR}" nohup python3 -m uvicorn visualization_3d.app:app \
        --host 0.0.0.0 --port "${PORT}" > /tmp/nanodynamics_server.log 2>&1 &
    for _ in $(seq 1 30); do
        if curl -s "http://127.0.0.1:${PORT}/api/edge/status" >/dev/null 2>&1; then
            break
        fi
        sleep 0.3
    done
else
    echo "[✓] NanoDynamics server already active on port ${PORT}."
fi

echo "[+] Launching 3D Orbital Simulator & ESP32 PINN Settings Portal..."
if command -v google-chrome >/dev/null 2>&1; then
    google-chrome --new-window "http://localhost:${PORT}/" "http://localhost:${PORT}/edge-console" >/dev/null 2>&1 &
elif command -v chromium-browser >/dev/null 2>&1; then
    chromium-browser --new-window "http://localhost:${PORT}/" "http://localhost:${PORT}/edge-console" >/dev/null 2>&1 &
elif command -v firefox >/dev/null 2>&1; then
    firefox "http://localhost:${PORT}/" "http://localhost:${PORT}/edge-console" >/dev/null 2>&1 &
else
    xdg-open "http://localhost:${PORT}/" >/dev/null 2>&1 &
    sleep 0.5
    xdg-open "http://localhost:${PORT}/edge-console" >/dev/null 2>&1 &
fi
