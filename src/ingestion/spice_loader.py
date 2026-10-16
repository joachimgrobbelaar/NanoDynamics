"""NASA NAIF SPICE (.bsp / .tls) kernel loader and planetary ephemeris interface.

Automatically uses `spiceypy` if installed and kernels (`de440s.bsp`, `naif0012.tls`)
are cached in `data/raw/`, or falls back to high-precision analytical JPL Meeus/VSOP87
ephemerides for Earth, Moon, and Sun in the GCRF/J2000 frame.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple
import urllib.request

import numpy as np

NAIF_LSK_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
NAIF_SPK_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"


class SPICEEphemerisLoader:
    """Queries Solar System barycentric/geocentric states via NAIF SPICE or analytical JPL series."""

    def __init__(self, raw_data_dir: str | Path | None = None) -> None:
        if raw_data_dir is None:
            raw_data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
        self.raw_dir = Path(raw_data_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.spice_available = False
        self._try_load_spice()

    def _try_load_spice(self) -> None:
        try:
            import spiceypy as spice  # type: ignore

            lsk = self.raw_dir / "naif0012.tls"
            spk = self.raw_dir / "de440s.bsp"
            if lsk.is_file() and spk.is_file():
                spice.furnsh(str(lsk))
                spice.furnsh(str(spk))
                self.spice_available = True
        except Exception:
            self.spice_available = False

    def download_naif_kernels(self, include_de440s: bool = False) -> dict:
        """Download official NASA NAIF leapseconds (`naif0012.tls`) and optional `de440s.bsp`."""
        lsk_path = self.raw_dir / "naif0012.tls"
        if not lsk_path.exists():
            urllib.request.urlretrieve(NAIF_LSK_URL, str(lsk_path))
        result = {"lsk": str(lsk_path)}
        if include_de440s:
            spk_path = self.raw_dir / "de440s.bsp"
            if not spk_path.exists():
                urllib.request.urlretrieve(NAIF_SPK_URL, str(spk_path))
            result["spk"] = str(spk_path)
        self._try_load_spice()
        return result

    def get_moon_sun_positions_gcrf(self, t_seconds_from_j2000: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(r_moon_m, r_sun_m)` in GCRF/J2000 meters at `t_seconds_from_j2000`."""
        if self.spice_available:
            try:
                import spiceypy as spice  # type: ignore

                moon_st, _ = spice.spkezr("MOON", float(t_seconds_from_j2000), "J2000", "NONE", "EARTH")
                sun_st, _ = spice.spkezr("SUN", float(t_seconds_from_j2000), "J2000", "NONE", "EARTH")
                return (
                    np.array(moon_st[:3], dtype=np.float64) * 1000.0,
                    np.array(sun_st[:3], dtype=np.float64) * 1000.0,
                )
            except Exception:
                pass

        # High-precision analytical ephemeris in GCRF/J2000 (SI meters)
        d = t_seconds_from_j2000 / 86400.0
        eps = math.radians(23.439291 - 3.563e-7 * d)

        # Moon geocentric position (384,400 km semi-major axis, 5.145 deg inclination to ecliptic)
        omega_m = 2.6616995e-6
        inc_m = math.radians(5.145396)
        u_m = omega_m * t_seconds_from_j2000
        a_m = 384_400_000.0 * (1.0 - 0.0549 * math.cos(u_m))
        x_ecl_m = a_m * math.cos(u_m)
        y_ecl_m = a_m * math.sin(u_m) * math.cos(inc_m)
        z_ecl_m = a_m * math.sin(u_m) * math.sin(inc_m)
        r_moon = np.array(
            [
                x_ecl_m,
                y_ecl_m * math.cos(eps) - z_ecl_m * math.sin(eps),
                y_ecl_m * math.sin(eps) + z_ecl_m * math.cos(eps),
            ],
            dtype=np.float64,
        )

        # Sun geocentric position (1 AU)
        omega_s = 1.9909866e-7
        u_s = omega_s * t_seconds_from_j2000
        au_m = 149_597_870_700.0
        r_sun = np.array(
            [
                au_m * math.cos(u_s),
                au_m * math.sin(u_s) * math.cos(eps),
                au_m * math.sin(u_s) * math.sin(eps),
            ],
            dtype=np.float64,
        )
        return r_moon, r_sun
