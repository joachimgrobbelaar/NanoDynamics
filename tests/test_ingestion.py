"""Unit tests for NASA CDDIS Extended SP3, ESA Copernicus Sentinel EOF, and NAIF SPICE ingestion."""

from pathlib import Path
import numpy as np

from src.ingestion.eof_parser import parse_eof_file
from src.ingestion.sp3_parser import parse_sp3_file
from src.ingestion.spice_loader import SPICEEphemerisLoader

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_sp3_parser_swarm_and_grace_fo():
    sp3_file = RAW_DIR / "swarm_grace_fo.sp3"
    records = parse_sp3_file(sp3_file, target_sat_id="L47")
    assert len(records) == 6
    first = records[0]
    assert first.sat_id == "L47"
    # Position converted from km to meters (~6835.4 km -> 6,835,412.58 m)
    assert np.isclose(first.r_ecef_m[0], 6835412.58, atol=1e-2)
    # Velocity converted from dm/s to m/s (-154.218 dm/s -> -15.4218 m/s)
    assert np.isclose(first.v_ecef_m_s[0], -15.4218, atol=1e-4)
    assert np.isclose(records[1].t_seconds, 60.0)


def test_eof_parser_esa_sentinel_1a():
    eof_file = RAW_DIR / "S1A_OPER_AUX_POEORB_sample.EOF"
    records = parse_eof_file(eof_file)
    assert len(records) == 6
    assert records[0].mission == "Sentinel-1A"
    assert records[0].orbit_number == 56142
    assert np.isclose(records[0].r_ecef_m[0], 7071137.0, atol=1e-3)
    assert np.isclose(records[0].v_ecef_m_s[2], 7435.1924, atol=1e-4)
    assert np.isclose(records[-1].t_seconds, 300.0)


def test_spice_ephemeris_loader_moon_sun_gcrf():
    loader = SPICEEphemerisLoader(RAW_DIR)
    r_moon, r_sun = loader.get_moon_sun_positions_gcrf(0.0)
    moon_dist_km = float(np.linalg.norm(r_moon)) * 1e-3
    sun_dist_au = float(np.linalg.norm(r_sun)) / 149_597_870_700.0
    assert 355_000.0 <= moon_dist_km <= 408_000.0
    assert 0.98 <= sun_dist_au <= 1.02
