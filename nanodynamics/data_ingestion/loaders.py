"""
nanodynamics/data_ingestion/loaders.py

Unified data ingestion, streaming, and validation engine for Hugging Face
astrodynamics and space systems datasets.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, IterableDataset, load_dataset
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nanodynamics.loaders")


# ============================================================================
# 1. Configuration & Constants
# ============================================================================

CACHE_DIR_DEFAULT = Path(os.getenv("NANODYNAMICS_CACHE_DIR", Path.home() / ".cache" / "nanodynamics"))
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN", None))

if not HF_TOKEN:
    logger.warning("No Hugging Face token detected in environment. Private/gated datasets may fail to load.")


class DataCategory(str, Enum):
    EPHEMERIS = "ephemeris_catalogs"
    ORBITAL = "orbital_mechanics"
    SPACE_WEATHER = "space_weather_atmos"
    PLANETARY = "planetary_exploration"
    SPACE_OPS = "space_ops_infrastructure"
    COMPUTER_VISION = "computer_vision_sensors"
    INSTRUCTION_BENCH = "llm_instruction_bench"


@dataclass(frozen=True)
class DatasetMeta:
    hf_id: str
    category: DataCategory
    is_large: bool = False  # If True, defaults to streaming=True
    split: str = "train"
    subset: Optional[str] = None


# Central registry mapping short keys to Hugging Face dataset endpoints
DATASET_REGISTRY: Dict[str, DatasetMeta] = {
    # Pillar 1: Ephemeris & Small Bodies
    "jpl_small_bodies": DatasetMeta("juliensimon/jpl-small-body-database", DataCategory.EPHEMERIS, is_large=True),
    "sdss_asteroid_taxonomy": DatasetMeta("juliensimon/sdss-asteroid-taxonomy", DataCategory.EPHEMERIS),
    "comet_catalog": DatasetMeta("juliensimon/comet-catalog", DataCategory.EPHEMERIS),
    "solar_system_metadata": DatasetMeta("krt1k/wikipedia_solar_system_dataset", DataCategory.EPHEMERIS),
    "solar_system_moons": DatasetMeta("juliensimon/solar-system-moons", DataCategory.EPHEMERIS),
    "gaia_dr3_sso": DatasetMeta("juliensimon/gaia-dr3-solar-system-objects", DataCategory.EPHEMERIS, is_large=True),
    "solar_3d": DatasetMeta("erisdataworks/solar-3d", DataCategory.EPHEMERIS),
    "globular_clusters": DatasetMeta("juliensimon/globular-star-clusters", DataCategory.EPHEMERIS),
    "hypervelocity_stars": DatasetMeta("juliensimon/hypervelocity-stars-li2023", DataCategory.EPHEMERIS),

    # Pillar 2: Orbital Mechanics & Space Domain Awareness
    "starlink_tle_latest": DatasetMeta("juliensimon/starlink-tle-latest", DataCategory.ORBITAL),
    "satcat_catalog": DatasetMeta("juliensimon/space-track-satcat", DataCategory.ORBITAL),
    "satellite_conjunctions": DatasetMeta("juliensimon/satellite-conjunctions", DataCategory.ORBITAL),
    "orbital_fragmentations": DatasetMeta("juliensimon/orbital-fragmentation-events", DataCategory.ORBITAL),
    "reentry_events": DatasetMeta("juliensimon/reentry-events", DataCategory.ORBITAL),
    "constellation_census": DatasetMeta("juliensimon/constellation-census", DataCategory.ORBITAL),
    "starlink_fleet": DatasetMeta("juliensimon/starlink-fleet-data", DataCategory.ORBITAL),
    "oneweb_fleet": DatasetMeta("juliensimon/oneweb-fleet-data", DataCategory.ORBITAL),
    "kuiper_fleet": DatasetMeta("juliensimon/kuiper-fleet-data", DataCategory.ORBITAL),
    "globalstar_fleet": DatasetMeta("juliensimon/globalstar-fleet-data", DataCategory.ORBITAL),
    "ucs_satellites": DatasetMeta("juliensimon/ucs-satellite-database", DataCategory.ORBITAL),
    "leomsat": DatasetMeta("StefanBaudier/LEOMSAT", DataCategory.ORBITAL),
    "het_transfer_efficiency": DatasetMeta("Taylor658/HET_Transfer_Orbit_Efficiency", DataCategory.ORBITAL),
    "hektor_hohmann_transfer": DatasetMeta("Taylor658/624-hektor-hohmann-transfer-orbit", DataCategory.ORBITAL),
    "orbit_planner": DatasetMeta("warriorLZJ/Orbit_Planner", DataCategory.ORBITAL),
    "orbitbench": DatasetMeta("WorldAgents-c/orbitbench", DataCategory.ORBITAL),
    "orbitwren_compute_tracker": DatasetMeta("CyberMax-tools/orbitwren-orbital-compute-tracker", DataCategory.ORBITAL),
    "orbital_chaos_ssc": DatasetMeta("datamatters24/orbital-chaos-nasa-ssc", DataCategory.ORBITAL),

    # Pillar 3: Space Weather & Atmospheric Drag
    "space_weather_indices": DatasetMeta("juliensimon/space-weather-indices", DataCategory.SPACE_WEATHER),
    "celestrak_space_weather": DatasetMeta("juliensimon/celestrak-space-weather", DataCategory.SPACE_WEATHER),
    "f107_solar_flux": DatasetMeta("juliensimon/f107-solar-flux", DataCategory.SPACE_WEATHER),
    "iers_eop": DatasetMeta("juliensimon/iers-earth-orientation", DataCategory.SPACE_WEATHER),
    "ionosphere_radar": DatasetMeta("mstz/ionosphere", DataCategory.SPACE_WEATHER),
    "iri_iono_maps": DatasetMeta("arodland/IRI-iono-maps", DataCategory.SPACE_WEATHER),
    "cesm_atm": DatasetMeta("sdrbench/cesm-atm", DataCategory.SPACE_WEATHER, is_large=True),
    "strata_scream": DatasetMeta("nvidia/STRATA-SCREAM-sdecadal", DataCategory.SPACE_WEATHER, is_large=True),
    "wrf_severe_weather": DatasetMeta("deepguess/wrf-250m-severe-weather", DataCategory.SPACE_WEATHER, is_large=True),
    "wrf_weather_overlays": DatasetMeta("deepguess/wrf-250m-severe-weather-overlays", DataCategory.SPACE_WEATHER),
    "cloud_temp_data": DatasetMeta("DebasishDhal99/cloud-vs-temperature-data", DataCategory.SPACE_WEATHER),

    # Pillar 4: Planetary & Deep Space Exploration
    "maven_insitu": DatasetMeta("juliensimon/nasa-maven-kp-insitu", DataCategory.PLANETARY),
    "openmars_reanalysis": DatasetMeta("ananyo01/ARCO-OpenMars", DataCategory.PLANETARY, is_large=True),
    "venus_express_obs": DatasetMeta("juliensimon/esa-venus-express-observations", DataCategory.PLANETARY),
    "artemis_ii": DatasetMeta("juliensimon/artemis-ii", DataCategory.PLANETARY),

    # Pillar 5: Space Operations & Infrastructure
    "space_mission_intelligence": DatasetMeta("JuanCastillo29/space-mission-intelligence-data", DataCategory.SPACE_OPS),
    "space_agencies": DatasetMeta("juliensimon/space-agency-database", DataCategory.SPACE_OPS),
    "space_launch_log": DatasetMeta("juliensimon/space-launch-log", DataCategory.SPACE_OPS),
    "spacex_launches": DatasetMeta("windsurfer1511/spacex-launches", DataCategory.SPACE_OPS),
    "rocket_lab_launches": DatasetMeta("juliensimon/rocket-lab-launches", DataCategory.SPACE_OPS),
    "blue_origin_launches": DatasetMeta("juliensimon/blue-origin-launches", DataCategory.SPACE_OPS),
    "launch_vehicles": DatasetMeta("juliensimon/launch-vehicles", DataCategory.SPACE_OPS),
    "starlink_ground_stations": DatasetMeta("juliensimon/starlink-ground-stations", DataCategory.SPACE_OPS),
    "satnogs_transmitters": DatasetMeta("juliensimon/satnogs-transmitters", DataCategory.SPACE_OPS),

    # Pillar 6: Computer Vision & Sensor Kinetics
    "spacecraft_pose_estimation": DatasetMeta("lorenzobottelli/multi-target-spacecraft-pose-estimation", DataCategory.COMPUTER_VISION),
    "downstream_3dclouds": DatasetMeta("nasa-cisto-data-science-group/downstream-3dclouds-subset", DataCategory.COMPUTER_VISION, is_large=True),
    "city_satellite": DatasetMeta("jkellogg123/city_satellite", DataCategory.COMPUTER_VISION),
    "temporal_cityline": DatasetMeta("SharpShots/temporal-aerial-cityline-construction-sample", DataCategory.COMPUTER_VISION),
    "timewarp": DatasetMeta("microsoft/timewarp", DataCategory.COMPUTER_VISION),

    # Pillar 7: LLM Instruction & QA Tuning
    "orbital_mechanics_1": DatasetMeta("vanta-research/orbital-mechanics-1", DataCategory.INSTRUCTION_BENCH),
    "orbital_mechanics_instruct_32": DatasetMeta("abdohisham12/orbital-mechanics-instruct-32", DataCategory.INSTRUCTION_BENCH),
    "ssao_space_instruct": DatasetMeta("fabsssss/ssao-space-instruct", DataCategory.INSTRUCTION_BENCH),
    "momo_pretraining": DatasetMeta("Mirali33/MOMO-pretraining-data", DataCategory.INSTRUCTION_BENCH),
}


# ============================================================================
# 2. Schemas & Type Validation (Pydantic V2)
# ============================================================================

class KeplerianElementRecord(BaseModel):
    """Osculating orbital elements for small bodies and satellites."""
    object_id: str
    semi_major_axis_au: Optional[float] = Field(default=None, alias="a")
    eccentricity: float = Field(..., alias="e", ge=0.0)
    inclination_deg: float = Field(..., alias="i", ge=0.0, le=180.0)
    longitude_asc_node_deg: Optional[float] = Field(default=0.0, alias="om")
    arg_perihelion_deg: Optional[float] = Field(default=0.0, alias="w")
    mean_anomaly_deg: Optional[float] = Field(default=0.0, alias="ma")
    epoch_jd: Optional[float] = None


class SpaceWeatherRecord(BaseModel):
    """Solar and geomagnetic activity parameters."""
    timestamp_utc: str
    f107_flux: Optional[float] = Field(default=None, description="Solar radio flux F10.7 in sfu")
    kp_index: Optional[float] = Field(default=None, ge=0.0, le=9.0)
    ap_index: Optional[float] = Field(default=None, ge=0.0)
    dst_index_nt: Optional[float] = Field(default=None, description="Dst in nanotesla")


class ConjunctionEventRecord(BaseModel):
    """Close approach and conjunction assessment."""
    tca_utc: str = Field(..., description="Time of Closest Approach")
    primary_norad_id: int
    secondary_norad_id: int
    miss_distance_m: float = Field(..., ge=0.0)
    collision_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ============================================================================
# 3. Core Loader & Local Cache Engine
# ============================================================================

class NanodynamicsDataLoader:
    """Central interface for streaming and caching Hugging Face datasets."""

    def __init__(self, cache_dir: Path = CACHE_DIR_DEFAULT):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str, split: str) -> Path:
        meta = DATASET_REGISTRY[key]
        cat_dir = self.cache_dir / meta.category.value
        cat_dir.mkdir(parents=True, exist_ok=True)
        return cat_dir / f"{key}_{split}.parquet"

    def load(
        self,
        dataset_key: str,
        split: Optional[str] = None,
        force_stream: Optional[bool] = None,
        use_cache: bool = True,
    ) -> Union[Dataset, IterableDataset]:
        """
        Loads a registered dataset. If large or forced, returns an IterableDataset.
        Otherwise loads into memory, caching to disk as a Parquet file.
        """
        if dataset_key not in DATASET_REGISTRY:
            raise KeyError(
                f"Dataset '{dataset_key}' is not registered. Choose from: {list(DATASET_REGISTRY.keys())}"
            )

        meta = DATASET_REGISTRY[dataset_key]
        target_split = split or meta.split
        cache_path = self._get_cache_path(dataset_key, target_split)
        should_stream = meta.is_large if force_stream is None else force_stream

        # 1. Check local Parquet cache
        if use_cache and not should_stream and cache_path.exists():
            logger.info("Loading cached Parquet table for '%s' from %s", dataset_key, cache_path)
            return Dataset.from_parquet(str(cache_path))

        # 2. Ingest via Hugging Face
        logger.info(
            "Fetching dataset '%s' (%s) [streaming=%s, split=%s]",
            dataset_key, meta.hf_id, should_stream, target_split
        )

        try:
            ds = load_dataset(
                meta.hf_id,
                name=meta.subset,
                split=target_split,
                streaming=should_stream,
                token=HF_TOKEN,
            )
        except Exception as err:
            logger.error("Failed to load dataset '%s' from Hugging Face: %s", meta.hf_id, err)
            raise

        # 3. Cache static datasets
        if use_cache and not should_stream and isinstance(ds, Dataset):
            try:
                ds.to_parquet(str(cache_path))
                logger.info("Saved local Parquet cache for '%s' at %s", dataset_key, cache_path)
            except Exception as err:
                logger.warning("Could not cache Parquet for '%s': %s", dataset_key, err)

        return ds

    def stream_batches(
        self,
        dataset_key: str,
        batch_size: int = 1000,
        split: Optional[str] = None,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Streams records in manageable memory chunks for large-scale processing."""
        ds = self.load(dataset_key, split=split, force_stream=True)
        batch: List[Dict[str, Any]] = []

        for record in ds:
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def validate_stream(
        self,
        dataset_key: str,
        schema_cls: type[BaseModel],
        max_samples: int = 50,
    ) -> Dict[str, Any]:
        """Validates incoming records against a Pydantic schema model."""
        ds = self.load(dataset_key, force_stream=True)
        valid_count = 0
        errors: List[str] = []

        for idx, row in enumerate(ds):
            if idx >= max_samples:
                break
            try:
                schema_cls.model_validate(row)
                valid_count += 1
            except ValidationError as e:
                errors.append(f"Row {idx} failed validation: {e.errors(include_url=False)}")

        return {
            "dataset_key": dataset_key,
            "checked_samples": idx + 1 if 'idx' in locals() else 0,
            "valid_samples": valid_count,
            "error_count": len(errors),
            "errors": errors[:5],  # Sample first 5 errors
        }


# ============================================================================
# 4. Global Singleton Accessor
# ============================================================================

loader = NanodynamicsDataLoader()


def get_dataset(name: str, **kwargs) -> Union[Dataset, IterableDataset]:
    """Top-level functional interface to fetch any registered dataset."""
    return loader.load(name, **kwargs)