"""TLE / OMM ephemeris ingestion and Keplerian state derivation client."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from orbitguard.physics.constants import MU_EARTH
from orbitguard.physics.elements import OrbitalElements, coe_to_rv


@dataclass
class ParsedTLE:
    """Structured representation of a NORAD Two-Line Element set."""

    name: str
    norad_cat_id: int
    intl_designator: str
    epoch_utc: datetime.datetime
    bstar: float
    inclination_rad: float
    raan_rad: float
    eccentricity: float
    arg_perigee_rad: float
    mean_anomaly_rad: float
    mean_motion_rev_day: float
    revolution_number: int
    raw_line1: str
    raw_line2: str

    def to_coe_and_state(self) -> Tuple[OrbitalElements, np.ndarray, np.ndarray]:
        """Solve Kepler's equation and derive Cartesian ECI position and velocity."""
        # Mean motion in rad/s
        n_rad_s = self.mean_motion_rev_day * (2.0 * math.pi / 86400.0)
        a = (MU_EARTH / (n_rad_s**2)) ** (1.0 / 3.0)

        # Solve Kepler's equation M = E - e * sin(E) for Eccentric Anomaly E
        M = self.mean_anomaly_rad
        e = self.eccentricity
        E = M
        for _ in range(15):
            f = E - e * math.sin(E) - M
            f_prime = 1.0 - e * math.cos(E)
            delta = f / f_prime
            E -= delta
            if abs(delta) < 1e-10:
                break

        # Compute true anomaly nu
        sin_half_nu = math.sqrt(max(0.0, 1.0 + e)) * math.sin(E / 2.0)
        cos_half_nu = math.sqrt(max(0.0, 1.0 - e)) * math.cos(E / 2.0)
        nu = 2.0 * math.atan2(sin_half_nu, cos_half_nu)
        if nu < 0:
            nu += 2.0 * math.pi

        coe = OrbitalElements(
            a=a,
            e=e,
            i=self.inclination_rad,
            raan=self.raan_rad,
            arg_pe=self.arg_perigee_rad,
            true_anomaly=nu,
            epoch=self.epoch_utc.timestamp(),
        )

        r_eci, v_eci = coe_to_rv(coe, mu=MU_EARTH)
        return coe, r_eci, v_eci


class TLEClient:
    """Ingests and parses Two-Line Element (TLE) ephemeris records."""

    # Built-in High-Fidelity Catalog for local offline autonomy (Standby Protocol)
    BUILTIN_CATALOG: List[Dict[str, str]] = [
        {
            "name": "AEROCUBE-3U (PRIMARY)",
            "line1": "1 43015U 17071F   26305.12500000  .00001234  00000-0  54321-4 0  9991",
            "line2": "2 43015  51.6420 120.4500 0008500  45.1200 315.0200 15.48500000123456",
            "type": "primary_cubesat",
        },
        {
            "name": "COSMOS-2251 DEB (THREAT 1)",
            "line1": "1 34123U 93036AZK 26305.12500000  .00002500  00000-0  98765-4 0  9998",
            "line2": "2 34123  51.6435 120.4520 0008600  45.1100 315.0180 15.48510000234567",
            "type": "debris_threat",
        },
        {
            "name": "FENGYUN-1C DEB (THREAT 2)",
            "line1": "1 31112U 99025AKL 26305.12500000  .00000800  00000-0  34567-4 0  9993",
            "line2": "2 31112  98.6500 240.1200 0012000  70.2000 290.1500 14.82000000345678",
            "type": "debris_safe",
        },
        {
            "name": "SL-08 R/B (THREAT 3)",
            "line1": "1 28990U 05030B   26305.12500000  .00001500  00000-0  67890-4 0  9995",
            "line2": "2 28990  51.6400 120.4480 0008400  45.1300 315.0250 15.48490000456789",
            "type": "debris_threat",
        },
    ]

    @staticmethod
    def parse_tle(name: str, line1: str, line2: str) -> ParsedTLE:
        """Parse standard two-line element strings into a ParsedTLE object."""
        norad_id = int(line1[2:7])
        intl_desig = line1[9:17].strip()

        # Parse epoch year and fractional day
        epoch_yr_str = line1[18:20]
        epoch_day_str = line1[20:32]
        yr = int(epoch_yr_str)
        full_year = 2000 + yr if yr < 57 else 1900 + yr
        fractional_day = float(epoch_day_str)

        # Convert to datetime
        start_of_year = datetime.datetime(full_year, 1, 1, tzinfo=datetime.timezone.utc)
        epoch_utc = start_of_year + datetime.timedelta(days=fractional_day - 1.0)

        # Parse B* drag term: format is +/-XXXXX-X => +/-0.XXXXX * 10^(+/-X)
        bstar_mantissa = line1[53:59].strip()
        bstar_exp = line1[59:61].strip()
        try:
            bstar = float(f"{bstar_mantissa}e{bstar_exp}") * 1e-5
        except Exception:
            bstar = 0.0

        # Line 2 components
        inc_deg = float(line2[8:16])
        raan_deg = float(line2[17:25])
        ecc_str = "0." + line2[26:33].strip()
        ecc = float(ecc_str)
        arg_pe_deg = float(line2[34:42])
        mean_anom_deg = float(line2[43:51])
        mean_motion = float(line2[52:63])
        rev_num = int(line2[63:68]) if line2[63:68].strip().isdigit() else 0

        return ParsedTLE(
            name=name.strip(),
            norad_cat_id=norad_id,
            intl_designator=intl_desig,
            epoch_utc=epoch_utc,
            bstar=bstar,
            inclination_rad=math.radians(inc_deg),
            raan_rad=math.radians(raan_deg),
            eccentricity=ecc,
            arg_perigee_rad=math.radians(arg_pe_deg),
            mean_anomaly_rad=math.radians(mean_anom_deg),
            mean_motion_rev_day=mean_motion,
            revolution_number=rev_num,
            raw_line1=line1.strip(),
            raw_line2=line2.strip(),
        )

    def load_catalog(self) -> List[ParsedTLE]:
        """Load registered satellite and debris objects from local catalog."""
        records = []
        for item in self.BUILTIN_CATALOG:
            parsed = self.parse_tle(item["name"], item["line1"], item["line2"])
            records.append(parsed)
        return records
