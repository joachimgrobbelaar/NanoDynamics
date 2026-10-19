"""Dynamic Space Weather & Ionospheric TEC Engine.

Provides real-time ingest from NOAA Space Weather Prediction Center (SWPC)
with offline cached fallback, dynamic atmospheric drag scaling, and ionospheric
Total Electron Content (TEC) radio dispersion modeling.
"""

from dataclasses import dataclass
import json
import math
import os
import time
import urllib.request
from typing import Any, Dict, Optional

NOAA_F107_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
NOAA_KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index.json"
SPEED_OF_LIGHT = 299792458.0


@dataclass
class SpaceWeatherIndices:
    f107: float
    kp: float
    ap: float
    density_scale: float
    source: str
    timestamp: float


class SpaceWeatherManager:
    """Manages solar flux (F10.7) and geomagnetic index (Kp) tracking."""

    _instance: Optional["SpaceWeatherManager"] = None

    def __init__(self, cache_ttl_sec: float = 3600.0, cache_file: Optional[str] = None):
        self.cache_ttl_sec = cache_ttl_sec
        self.cache_file = cache_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", ".space_weather_cache.json"
        )
        self._current_indices: Optional[SpaceWeatherIndices] = None
        self._last_fetch_time: float = 0.0

    @classmethod
    def get_instance(cls) -> "SpaceWeatherManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def compute_density_multiplier(f107: float, kp: float) -> float:
        """Calculate dynamic upper-atmosphere density scale factor.

        Follows Jacchia-Roberts / NRLMSISE empirical scaling:
        rho_eff = rho_0 * (1.0 + 0.008 * max(0, F10.7 - 70) + 0.15 * Kp)
        """
        delta_f = max(0.0, float(f107) - 70.0)
        kp_val = max(0.0, min(9.0, float(kp)))
        mult = 1.0 + (0.008 * delta_f) + (0.15 * kp_val)
        return float(mult)

    @staticmethod
    def compute_ionospheric_tec(
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
        f107: float = 135.0,
    ) -> Dict[str, float]:
        """Calculate vertical Total Electron Content (TEC) and frequency-dependent RF delays.

        TEC in TECU (1 TECU = 10^16 e-/m^2).
        RF group delay: Delta_tau = (40.3 * TEC * 1e16) / (c * f^2).
        """
        # Peak ionosphere F2 layer around 350 km
        f2_peak_alt = 350.0
        alt_diff = abs(alt_km - f2_peak_alt)
        base_tec = 25.0 * math.exp(-((alt_diff / 150.0) ** 2))

        # Equatorial Ionospheric Anomaly (EIA) peaks near +/- 15 deg geomagnetic lat
        eia_factor = 1.0 + 0.6 * math.exp(-((abs(lat_deg) - 15.0) / 10.0) ** 2)

        # Solar activity scaling factor
        solar_factor = (max(60.0, float(f107)) / 70.0) ** 0.8
        tec_tecu = max(2.0, base_tec * eia_factor * solar_factor)

        # Delays at common frequencies
        def rf_delay_ns(freq_hz: float) -> float:
            dt_s = (40.3 * (tec_tecu * 1e16)) / (SPEED_OF_LIGHT * (freq_hz ** 2))
            return dt_s * 1e9

        return {
            "tec_tecu": float(tec_tecu),
            "iono_delay_uhf_ns": float(rf_delay_ns(4.35e8)),       # 435 MHz CubeSat UHF
            "iono_delay_l1_ns": float(rf_delay_ns(1.57542e9)),     # 1.575 GHz GPS L1
            "iono_delay_s_band_ns": float(rf_delay_ns(2.4e9)),     # 2.4 GHz S-band
            "f107_applied": float(f107),
        }

    def _fetch_noaa_json(self, url: str, timeout: float = 2.5) -> Optional[Any]:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LEOSimulator/2.0 (SpaceWeatherEngine)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
        return None

    def fetch_indices(self, force_refresh: bool = False) -> SpaceWeatherIndices:
        """Fetch current F10.7 and Kp from NOAA SWPC or fallback to cache/default."""
        now = time.time()
        if (
            not force_refresh
            and self._current_indices is not None
            and (now - self._last_fetch_time) < self.cache_ttl_sec
        ):
            return self._current_indices

        # Try live NOAA SWPC
        f107_val = None
        kp_val = None
        ap_val = 9.0

        f107_data = self._fetch_noaa_json(NOAA_F107_URL)
        if isinstance(f107_data, list) and len(f107_data) > 0:
            latest = f107_data[-1]
            if "flux" in latest and latest["flux"] is not None:
                f107_val = float(latest["flux"])

        kp_data = self._fetch_noaa_json(NOAA_KP_URL)
        if isinstance(kp_data, list) and len(kp_data) > 0:
            latest_kp = kp_data[-1]
            if "kp_index" in latest_kp and latest_kp["kp_index"] is not None:
                kp_val = float(latest_kp["kp_index"])

        if f107_val is not None and kp_val is not None:
            source = "noaa_swpc_live"
        else:
            # Fallback to local cache if present
            cached = self._load_disk_cache()
            if cached:
                f107_val = f107_val if f107_val is not None else cached.get("f107", 135.0)
                kp_val = kp_val if kp_val is not None else cached.get("kp", 2.5)
                ap_val = cached.get("ap", 9.0)
                source = "disk_cache"
            else:
                f107_val = f107_val if f107_val is not None else 135.0
                kp_val = kp_val if kp_val is not None else 2.5
                ap_val = 9.0
                source = "default_climatology"

        density_scale = self.compute_density_multiplier(f107_val, kp_val)
        indices = SpaceWeatherIndices(
            f107=float(f107_val),
            kp=float(kp_val),
            ap=float(ap_val),
            density_scale=float(density_scale),
            source=source,
            timestamp=now,
        )

        self._current_indices = indices
        self._last_fetch_time = now

        if source == "noaa_swpc_live":
            self._save_disk_cache(indices)

        return indices

    def _load_disk_cache(self) -> Optional[Dict[str, Any]]:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _save_disk_cache(self, indices: SpaceWeatherIndices) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(
                    {
                        "f107": indices.f107,
                        "kp": indices.kp,
                        "ap": indices.ap,
                        "density_scale": indices.density_scale,
                        "timestamp": indices.timestamp,
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    def get_historical_indices(self, epoch_s: float) -> SpaceWeatherIndices:
        """Look up historical F10.7 and ap/Kp from cached HF space-weather-indices dataset."""
        try:
            from leo_simulator import hf_data

            if not hasattr(self, "_hf_table") or self._hf_table is None:
                self._hf_table = hf_data.load_space_weather()
            row = hf_data.space_weather_lookup(epoch_s, self._hf_table)
            f107_val = float(row["f107"])
            ap_val = float(row["ap"])
            kp_val = min(9.0, max(0.0, math.log(max(1.0, ap_val)) * 1.85))
            density_scale = self.compute_density_multiplier(f107_val, kp_val)
            return SpaceWeatherIndices(
                f107=f107_val,
                kp=kp_val,
                ap=ap_val,
                density_scale=density_scale,
                source="hf_space_weather_indices",
                timestamp=float(epoch_s),
            )
        except Exception:
            return self.fetch_indices()

