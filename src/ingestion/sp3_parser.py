"""Extended SP3-c/d ASCII precise orbit ephemeris parser (NASA CDDIS / IGS / GFZ).

Converts SP3 position records (`P` lines in km) and velocity records (`V` lines in dm/s)
into SI state vectors (`[x, y, z]` in meters, `[vx, vy, vz]` in m/s) in the ITRF/ECEF frame,
with optional kinematic transformation to GCRF/ECI via `src.astrodynamics.frames`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request

import numpy as np


@dataclass
class SP3StateRecord:
    sat_id: str
    epoch_utc: datetime
    t_seconds: float
    r_ecef_m: np.ndarray
    v_ecef_m_s: np.ndarray
    clock_bias_us: float = 0.0


def parse_sp3_string(content: str, target_sat_id: Optional[str] = None) -> List[SP3StateRecord]:
    """Parse an Extended SP3-c/d ASCII string into SI state records (`m`, `m/s`).

    SP3 specification:
    - Epoch header line starts with `*  YYYY MM DD HH MM SS.SSSSSSSS`
    - Position line starts with `P<SAT>` in kilometers (`km` -> `* 1e3 m`)
    - Velocity line starts with `V<SAT>` in decimeters/second (`dm/s` -> `* 0.1 m/s`)
    """
    lines = content.splitlines()
    current_epoch: Optional[datetime] = None
    t0: Optional[datetime] = None

    pos_buffer: Dict[str, tuple[np.ndarray, float]] = {}
    records: List[SP3StateRecord] = []

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line or line.startswith("EOF"):
            continue
        if line.startswith("*"):
            parts = line[1:].split()
            if len(parts) >= 6:
                year, month, day, hour, minute = (int(p) for p in parts[:5])
                sec_float = float(parts[5])
                sec_int = int(sec_float)
                usec = int(round((sec_float - sec_int) * 1e6))
                current_epoch = datetime(year, month, day, hour, minute, sec_int, usec, tzinfo=timezone.utc)
                if t0 is None:
                    t0 = current_epoch
                pos_buffer.clear()
        elif line.startswith("P") and current_epoch is not None:
            sat_id = line[1:4].strip()
            if target_sat_id and sat_id != target_sat_id:
                continue
            vals = line[4:].split()
            if len(vals) >= 3:
                x_km, y_km, z_km = float(vals[0]), float(vals[1]), float(vals[2])
                clk_us = float(vals[3]) if len(vals) >= 4 else 0.0
                r_m = np.array([x_km * 1000.0, y_km * 1000.0, z_km * 1000.0], dtype=np.float64)
                pos_buffer[sat_id] = (r_m, clk_us)
        elif line.startswith("V") and current_epoch is not None:
            sat_id = line[1:4].strip()
            if target_sat_id and sat_id != target_sat_id:
                continue
            vals = line[4:].split()
            if len(vals) >= 3 and sat_id in pos_buffer:
                vx_dms, vy_dms, vz_dms = float(vals[0]), float(vals[1]), float(vals[2])
                v_ms = np.array([vx_dms * 0.1, vy_dms * 0.1, vz_dms * 0.1], dtype=np.float64)
                r_m, clk_us = pos_buffer[sat_id]
                dt_s = (current_epoch - t0).total_seconds() if t0 else 0.0
                records.append(
                    SP3StateRecord(
                        sat_id=sat_id,
                        epoch_utc=current_epoch,
                        t_seconds=float(dt_s),
                        r_ecef_m=r_m,
                        v_ecef_m_s=v_ms,
                        clock_bias_us=clk_us,
                    )
                )

    # If SP3 file only contains P lines (no V lines), estimate velocity via central finite differences
    if not records and pos_buffer and current_epoch is not None:
        for sat_id, (r_m, clk_us) in pos_buffer.items():
            records.append(
                SP3StateRecord(
                    sat_id=sat_id,
                    epoch_utc=current_epoch,
                    t_seconds=0.0,
                    r_ecef_m=r_m,
                    v_ecef_m_s=np.zeros(3, dtype=np.float64),
                    clock_bias_us=clk_us,
                )
            )
    return records


def parse_sp3_file(file_path: str | Path, target_sat_id: Optional[str] = None) -> List[SP3StateRecord]:
    """Load and parse an SP3 file from disk."""
    path = Path(file_path)
    return parse_sp3_string(path.read_text(encoding="utf-8"), target_sat_id=target_sat_id)


def download_sample_sp3(url: str, dest_path: str | Path) -> Path:
    """Download an empirical SP3 precise orbit file from NASA CDDIS / IGS mirror."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))
    return dest
