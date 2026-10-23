"""Linux Native Raw 802.11 Monitor Mode Transceiver for ESP-NOW.

Enables direct peer-to-peer ESP-NOW communication from a Linux host without
requiring a USB bridge or dongle, by crafting and capturing 802.11 Action Frames
with Espressif's vendor OUI (0x18, 0xFE, 0x34).
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import time
from typing import Optional, Tuple


class LinuxESPNowMonitor:
    """Raw 802.11 socket engine for native ESP-NOW packet injection/sniffing."""

    ESPRESSIF_OUI = b"\x18\xfe\x34"
    ESPNOW_CATEGORY = 0x04

    def __init__(self, interface: str = "wlp2s0", channel: int = 1) -> None:
        self.interface = interface
        self.channel = channel
        self.raw_sock: Optional[socket.socket] = None
        self.is_active = False

    def init_monitor_interface(self, sudo_password: str = "xxx") -> bool:
        """Configures virtual monitor interface using iw/ip."""
        try:
            # Check if mon0 exists
            check = subprocess.run(["ip", "link", "show", "mon0"], capture_output=True, text=True)
            if "mon0" not in check.stdout:
                # Create monitor virtual interface on top of physical wifi device
                cmd = f"echo '{sudo_password}' | sudo -S iw dev {self.interface} interface add mon0 type monitor"
                subprocess.run(cmd, shell=True, capture_output=True)
                cmd_up = f"echo '{sudo_password}' | sudo -S ip link set mon0 up"
                subprocess.run(cmd_up, shell=True, capture_output=True)

            self.interface = "mon0"
            self.raw_sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
            self.raw_sock.bind((self.interface, 0x0003))
            self.raw_sock.settimeout(0.1)
            self.is_active = True
            return True
        except Exception as e:
            print(f"[LinuxESPNowMonitor] Failed to open raw 802.11 monitor socket: {e}")
            self.is_active = False
            return False

    def craft_espnow_frame(self, dest_mac: bytes, src_mac: bytes, payload: bytes) -> bytes:
        """Builds an IEEE 802.11 vendor-specific action frame carrying ESP-NOW payload."""
        # Minimal Radiotap Header (8 bytes)
        radiotap = b"\x00\x00\x08\x00\x00\x00\x00\x00"

        # IEEE 802.11 Action Frame Header (24 bytes)
        # Frame Control: 0x00d0 (Action Frame)
        fc = b"\xd0\x00"
        duration = b"\x00\x00"
        seq_ctrl = b"\x00\x00"
        bssid = b"\xff\xff\xff\xff\xff\xff"  # Broadcast BSSID for ESP-NOW

        header = fc + duration + dest_mac + src_mac + bssid + seq_ctrl

        # Action category + Espressif OUI (4 bytes)
        vendor_header = bytes([self.ESPNOW_CATEGORY]) + self.ESPRESSIF_OUI
        # ESP-NOW vendor-specific content
        return radiotap + header + vendor_header + payload

    def send_state_vector(
        self,
        dest_mac_str: str,
        state: list[float],
        seq_id: int = 1,
    ) -> bool:
        """Sends a 6D state vector payload to the target ESP32 MAC address."""
        if not self.raw_sock or not self.is_active:
            return False

        try:
            dest_mac = bytes.fromhex(dest_mac_str.replace(":", "").replace("-", ""))
            src_mac = b"\x00\x11\x22\x33\x44\x55"  # Host interface MAC or virtual
            payload = struct.pack("<I6f", seq_id, *state)
            frame = self.craft_espnow_frame(dest_mac, src_mac, payload)
            self.raw_sock.send(frame)
            return True
        except Exception as e:
            print(f"[LinuxESPNowMonitor] Transmit error: {e}")
            return False

    def receive_prediction(self, timeout_s: float = 0.05) -> Optional[Tuple[int, list[float], float]]:
        """Listens for an ESP-NOW response packet and extracts predicted state + on-chip compute time."""
        if not self.raw_sock or not self.is_active:
            return None

        t_end = time.perf_counter() + timeout_s
        while time.perf_counter() < t_end:
            try:
                pkt, _ = self.raw_sock.recvfrom(2048)
                idx = pkt.find(self.ESPRESSIF_OUI)
                if idx != -1 and len(pkt) >= idx + 3 + struct.calcsize("<I6fI"):
                    data = pkt[idx + 3 :]
                    seq_id, x, y, z, vx, vy, vz, proc_us = struct.unpack("<I6fI", data[:32])
                    proc_ms = proc_us / 1000.0
                    return seq_id, [x, y, z, vx, vy, vz], proc_ms
            except socket.timeout:
                break
            except Exception:
                break
        return None
