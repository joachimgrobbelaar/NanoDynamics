"""
nanodynamics – Unified astrodynamics data ingestion & analytics platform.

Provides streaming/batch access to 57+ Hugging Face space & astrodynamics
datasets organized across 7 functional pillars.
"""

__version__ = "0.2.0"

from nanodynamics.data_ingestion.loaders import (
    get_dataset,
    loader,
    DATASET_REGISTRY,
    DataCategory,
    DatasetMeta,
    NanodynamicsDataLoader,
    KeplerianElementRecord,
    SpaceWeatherRecord,
    ConjunctionEventRecord,
)

__all__ = [
    "get_dataset",
    "loader",
    "DATASET_REGISTRY",
    "DataCategory",
    "DatasetMeta",
    "NanodynamicsDataLoader",
    "KeplerianElementRecord",
    "SpaceWeatherRecord",
    "ConjunctionEventRecord",
]
