"""Hugging Face dataset loaders (TLE history, constellation TLEs, space weather, conjunctions).

Data is cached outside the repo (default ``~/.cache/leo_simulator/hf``; override with
``LEO_HF_CACHE``). Large parquet files are read with predicate pushdown so only the
requested rows are loaded into memory.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

CACHE_DIR = Path(os.environ.get("LEO_HF_CACHE", Path.home() / ".cache" / "leo_simulator" / "hf"))

DATASETS: Dict[str, Dict[str, str]] = {
    "tle_history": {"repo": "juliensimon/space-track-tle-history", "pattern": "data/tle_{year}.parquet"},
    "starlink": {"repo": "juliensimon/starlink-tle-latest", "pattern": "data/starlink.parquet"},
    "space_weather": {"repo": "juliensimon/space-weather-indices", "pattern": "data/space_weather_indices.parquet"},
    "conjunctions": {"repo": "juliensimon/satellite-conjunctions", "pattern": "data/conjunctions.parquet"},
}

MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0


def _local_path(key: str, year: Optional[int] = None) -> Path:
    spec = DATASETS[key]
    rel = spec["pattern"].format(year=year) if year is not None else spec["pattern"]
    return CACHE_DIR / spec["repo"].split("/")[1] / rel


def dataset_status() -> Dict[str, Any]:
    """Report which datasets are cached locally (with sizes in MB)."""
    out: Dict[str, Any] = {}
    for key, spec in DATASETS.items():
        base = CACHE_DIR / spec["repo"].split("/")[1] / "data"
        files = list(base.glob("*.parquet")) if base.exists() else []
        out[key] = {
            "repo": spec["repo"],
            "cached_files": sorted(f.name for f in files),
            "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
        }
    return out


def download_dataset(key: str, year: Optional[int] = None, timeout: float = 1800.0) -> Path:
    """Download one parquet file via the ``hf`` CLI. Returns the local path."""
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset '{key}'. Options: {sorted(DATASETS)}")
    spec = DATASETS[key]
    rel = spec["pattern"].format(year=year) if year is not None else spec["pattern"]
    local_dir = CACHE_DIR / spec["repo"].split("/")[1]
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["hf", "download", spec["repo"], rel, "--type", "dataset", "--local-dir", str(local_dir)],
        check=True, capture_output=True, timeout=timeout,
    )
    return local_dir / rel


def _read_parquet(path: Path, filters: Optional[list] = None, columns: Optional[List[str]] = None):
    import pyarrow.parquet as pq

    if not path.exists():
        raise FileNotFoundError(f"{path} not cached; call download_dataset() first")
    return pq.read_table(path, filters=filters, columns=columns).to_pandas()


def mean_motion_to_sma_m(mean_motion_rev_per_day: float) -> float:
    """Semi-major axis (m) from TLE mean motion (rev/day)."""
    n = mean_motion_rev_per_day * 2.0 * np.pi / 86400.0
    return float((MU_EARTH / (n * n)) ** (1.0 / 3.0))


def load_tle_history(norad_id: int, year: int) -> List[Dict[str, Any]]:
    """Time-ordered element sets for one NORAD object in one year (predicate pushdown)."""
    df = _read_parquet(_local_path("tle_history", year), filters=[("norad_id", "=", int(norad_id))])
    df = df.sort_values("epoch")
    recs = []
    for r in df.itertuples(index=False):
        recs.append({
            "epoch": r.epoch.timestamp(),
            "a_m": mean_motion_to_sma_m(r.mean_motion),
            "e": float(r.eccentricity),
            "inc_deg": float(r.inclination),
            "raan_deg": float(r.raan),
            "arg_pe_deg": float(r.arg_perigee),
            "mean_anomaly_deg": float(r.mean_anomaly),
            "bstar": float(r.bstar),
            "mean_motion_dot": float(r.mean_motion_dot),
        })
    return recs


def load_starlink_tles(limit: Optional[int] = None) -> List[Dict[str, str]]:
    df = _read_parquet(_local_path("starlink"))
    if limit:
        df = df.head(limit)
    return df.to_dict("records")


def load_space_weather(start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Daily F10.7 (observed, 81-day centered mean) and mean ap for a date window."""
    df = _read_parquet(_local_path("space_weather"),
                       columns=["date", "f107_obs", "f107_obs_center81", "ap_avg"])
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    return {
        "epoch": df["date"].astype("datetime64[ns]").astype("int64").to_numpy() / 1e9,
        "f107": df["f107_obs"].to_numpy(dtype=float),
        "f107_avg": df["f107_obs_center81"].to_numpy(dtype=float),
        "ap": df["ap_avg"].to_numpy(dtype=float),
    }


def space_weather_lookup(epoch_s: float, table: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Nearest-day F10.7/ap for a unix epoch; NaN-safe with sensible defaults."""
    if len(table["epoch"]) == 0:
        return {"f107": 150.0, "f107_avg": 150.0, "ap": 15.0}
    i = int(np.argmin(np.abs(table["epoch"] - epoch_s)))
    def _v(arr, default):
        x = float(arr[i])
        return x if np.isfinite(x) else default
    return {"f107": _v(table["f107"], 150.0), "f107_avg": _v(table["f107_avg"], 150.0), "ap": _v(table["ap"], 15.0)}


def load_conjunctions(max_range_km: float = 5.0) -> List[Dict[str, Any]]:
    df = _read_parquet(_local_path("conjunctions"), filters=[("min_range_km", "<=", float(max_range_km))])
    return df.to_dict("records")
