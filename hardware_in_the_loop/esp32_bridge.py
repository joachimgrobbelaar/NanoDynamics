import json
import os
import struct
import threading
import time
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np
import serial


class ESP32HardwareAgent:
    """Persistent Hardware-In-The-Loop (HIL) Bridge for Dual-ESP32 PINN inference.

    Maintains a persistent serial connection without toggling DTR/RTS on every
    invocation, and supports both USB Serial (`INFER:`, `STATUS?`, `WIFI_SET:`)
    and direct Wi-Fi HTTP (`/infer`, `/status`, `/wifi`) communication.
    """

    _pool: Dict[str, "ESP32HardwareAgent"] = {}
    _pool_lock = threading.Lock()

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        wifi_host: Optional[str] = None,
        reset_on_connect: bool = False,
    ):
        self.port = port
        self.baudrate = baudrate
        self.wifi_host = wifi_host
        self.ser: Optional[serial.Serial] = None
        self.lock = threading.Lock()
        self.last_t_infer_ms: float = 1.85
        self.last_rtt_ms: float = 4.20
        self.last_status: dict = {}
        self._connect_serial(reset_on_connect=reset_on_connect)

    @classmethod
    def get_shared(
        cls,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        wifi_host: Optional[str] = None,
    ) -> "ESP32HardwareAgent":
        key = f"{port}:{baudrate}:{wifi_host or ''}"
        with cls._pool_lock:
            agent = cls._pool.get(key)
            if agent is None or (agent.ser is None and os.path.exists(port)):
                agent = cls(port=port, baudrate=baudrate, wifi_host=wifi_host, reset_on_connect=False)
                cls._pool[key] = agent
            return agent

    def _connect_serial(self, reset_on_connect: bool = False) -> bool:
        if not os.path.exists(self.port):
            self.ser = None
            return False
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.35)
            if reset_on_connect:
                self.ser.setDTR(False)
                self.ser.setRTS(False)
                time.sleep(0.08)
                self.ser.setDTR(True)
                self.ser.setRTS(True)
                time.sleep(1.2)
            self.ser.reset_input_buffer()
            return True
        except Exception:
            self.ser = None
            return False

    def query_status(self) -> dict:
        """Query live hardware status over Serial (`STATUS?`) or Wi-Fi (`GET /status`)."""
        with self.lock:
            if self.ser is None and os.path.exists(self.port):
                self._connect_serial(reset_on_connect=False)
            if self.ser is not None and self.ser.is_open:
                try:
                    self.ser.reset_input_buffer()
                    self.ser.write(b"STATUS?\n")
                    self.ser.flush()
                    deadline = time.perf_counter() + 0.35
                    while time.perf_counter() < deadline:
                        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                        if line.startswith("{") and line.endswith("}"):
                            data = json.loads(line)
                            data["port"] = self.port
                            data["transport_active"] = "usb_serial"
                            self.last_status = data
                            return data
                except Exception:
                    self._connect_serial(reset_on_connect=False)

        if self.wifi_host:
            try:
                url = f"http://{self.wifi_host}/status"
                with urllib.request.urlopen(url, timeout=0.6) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    data["transport_active"] = "wifi_http"
                    self.last_status = data
                    return data
            except Exception:
                pass
        return self.last_status or {"role": "OFFLINE", "port": self.port, "transport_active": "offline"}

    def set_wifi_credentials(self, ssid: str, password: str, host_ip: str = "") -> dict:
        """Provision Wi-Fi SSID, password, and simulator Host IP onto the ESP32 NVS."""
        cmd = f"WIFI_SET:{ssid}|{password}|{host_ip}\n"
        with self.lock:
            if self.ser is None and os.path.exists(self.port):
                self._connect_serial(reset_on_connect=False)
            if self.ser is not None and self.ser.is_open:
                try:
                    self.ser.reset_input_buffer()
                    self.ser.write(cmd.encode("utf-8"))
                    self.ser.flush()
                    deadline = time.perf_counter() + 8.5
                    while time.perf_counter() < deadline:
                        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                        if line.startswith("{") and line.endswith("}"):
                            data = json.loads(line)
                            self.last_status = data
                            return {"status": "ok", "via": "usb_serial", "device": data}
                except Exception as e:
                    return {"status": "error", "error": str(e)}

        target_host = self.wifi_host or "192.168.4.1"
        try:
            body = json.dumps({"ssid": ssid, "password": password, "host_ip": host_ip}).encode("utf-8")
            req = urllib.request.Request(
                f"http://{target_host}/wifi",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {"status": "ok", "via": f"wifi_http ({target_host})", "device": data}
        except Exception as e:
            return {"status": "error", "error": f"Neither USB serial ({self.port}) nor Wi-Fi ({target_host}) responded: {e}"}

    def step_detailed(
        self,
        state_vector,
        dt: float = 10.0,
        b_coeff: float = 0.015,
        sat_id: int = 1,
    ) -> Tuple[np.ndarray, float, float, str]:
        """Run one orbital step on physical ESP32 hardware and return (next_state, t_infer_ms, rtt_ms, mode)."""
        if len(state_vector) != 6:
            raise ValueError("State vector must have 6 elements")
        s = [float(x) for x in state_vector]

        with self.lock:
            if self.ser is None and os.path.exists(self.port):
                self._connect_serial(reset_on_connect=False)
            if self.ser is not None and self.ser.is_open:
                try:
                    t0 = time.perf_counter()
                    cmd = (
                        f"INFER:{s[0]:.6f},{s[1]:.6f},{s[2]:.6f},"
                        f"{s[3]:.6f},{s[4]:.6f},{s[5]:.6f},{dt:.4f},{b_coeff:.5f}\n"
                    )
                    self.ser.reset_input_buffer()
                    self.ser.write(cmd.encode("utf-8"))
                    self.ser.flush()
                    deadline = t0 + 0.65
                    while time.perf_counter() < deadline:
                        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                        if line.startswith("RESULT:"):
                            parts = [float(v) for v in line[7:].split(",")]
                            if len(parts) >= 6:
                                rtt_ms = (time.perf_counter() - t0) * 1000.0
                                t_infer = parts[6] if len(parts) >= 7 else 1.85
                                self.last_t_infer_ms = float(t_infer)
                                self.last_rtt_ms = float(rtt_ms)
                                return (
                                    np.array(parts[:6], dtype=np.float32),
                                    self.last_t_infer_ms,
                                    self.last_rtt_ms,
                                    "esp32_serial_espnow",
                                )
                except Exception:
                    self._connect_serial(reset_on_connect=False)

        if self.wifi_host:
            try:
                t0 = time.perf_counter()
                payload = json.dumps(
                    {
                        "sat_id": sat_id,
                        "x": s[0],
                        "y": s[1],
                        "z": s[2],
                        "vx": s[3],
                        "vy": s[4],
                        "vz": s[5],
                        "dt": dt,
                        "B_coeff": b_coeff,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    f"http://{self.wifi_host}/infer",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=0.35) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    rtt_ms = (time.perf_counter() - t0) * 1000.0
                    t_infer = float(data.get("t_infer_ms", 1.85))
                    self.last_t_infer_ms = t_infer
                    self.last_rtt_ms = rtt_ms
                    return (
                        np.array([data["x"], data["y"], data["z"], data["vx"], data["vy"], data["vz"]], dtype=np.float32),
                        t_infer,
                        rtt_ms,
                        "esp32_wifi_espnow",
                    )
            except Exception:
                pass

        raise RuntimeError(f"ESP32 hardware agent unreachable on {self.port} or Wi-Fi {self.wifi_host}")

    def step(self, state_vector):
        """Compatible 6D state step returning np.ndarray(6, dtype=float32)."""
        next_state, _, _, _ = self.step_detailed(state_vector)
        return next_state

    def close(self):
        with self.lock:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
            self.ser = None


if __name__ == "__main__":
    agent = ESP32HardwareAgent.get_shared()
    print("ESP32 Status:", agent.query_status())
    test_state = [1.06, 0.0, 0.0, 0.0, 0.95, 0.0]
    nxt, t_inf, rtt, mode = agent.step_detailed(test_state)
    print(f"Result ({mode}): {nxt} | t_infer={t_inf:.2f}ms | RTT={rtt:.2f}ms")

