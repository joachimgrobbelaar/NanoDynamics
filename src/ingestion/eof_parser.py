"""ESA Copernicus Sentinel AUX_POEORB / Earth Explorer XML (.EOF) ephemeris parser.

Parses `<OSV>` (Orbit State Vector) blocks from ESA GOCE, Swarm, and Sentinel-1/2/3/6
Precise Orbit Determination (POD) XML files into SI state vectors (`m`, `m/s`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class EOFStateRecord:
    mission: str
    epoch_utc: datetime
    t_seconds: float
    orbit_number: int
    r_ecef_m: np.ndarray
    v_ecef_m_s: np.ndarray
    quality: str = "NOMINAL"


def _parse_utc_tag(utc_str: str) -> datetime:
    cleaned = utc_str.strip()
    if cleaned.startswith("UTC="):
        cleaned = cleaned[4:]
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_eof_xml_string(xml_content: str) -> List[EOFStateRecord]:
    """Parse an ESA Earth Observation File (.EOF) XML string into `EOFStateRecord` objects."""
    root = ET.fromstring(xml_content)
    mission_el = root.find(".//Fixed_Header/Mission")
    mission_name = mission_el.text.strip() if (mission_el is not None and mission_el.text) else "Sentinel"

    osv_elements = root.findall(".//OSV")
    records: List[EOFStateRecord] = []
    t0: datetime | None = None

    for osv in osv_elements:
        utc_text = osv.findtext("UTC", default="UTC=2026-10-16T00:00:00.000000")
        epoch = _parse_utc_tag(utc_text)
        if t0 is None:
            t0 = epoch
        dt_s = (epoch - t0).total_seconds()

        orbit_num = int(osv.findtext("Absolute_Orbit", default="0"))
        x = float(osv.findtext("X", default="0.0"))
        y = float(osv.findtext("Y", default="0.0"))
        z = float(osv.findtext("Z", default="0.0"))
        vx = float(osv.findtext("VX", default="0.0"))
        vy = float(osv.findtext("VY", default="0.0"))
        vz = float(osv.findtext("VZ", default="0.0"))
        quality = osv.findtext("Quality", default="NOMINAL").strip()

        records.append(
            EOFStateRecord(
                mission=mission_name,
                epoch_utc=epoch,
                t_seconds=float(dt_s),
                orbit_number=orbit_num,
                r_ecef_m=np.array([x, y, z], dtype=np.float64),
                v_ecef_m_s=np.array([vx, vy, vz], dtype=np.float64),
                quality=quality,
            )
        )
    return records


def parse_eof_file(file_path: str | Path) -> List[EOFStateRecord]:
    """Parse an ESA `.EOF` XML file from disk."""
    path = Path(file_path)
    return parse_eof_xml_string(path.read_text(encoding="utf-8"))
