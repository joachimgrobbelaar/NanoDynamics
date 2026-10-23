"""
nanodynamics/datasets.py

Unified high-level registry for accessing all nanodynamics datasets
and analytical functions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, IterableDataset

from nanodynamics.data_ingestion.loaders import (
    DATASET_REGISTRY,
    DataCategory,
    DatasetMeta,
    NanodynamicsDataLoader,
    get_dataset,
    loader,
)


def list_datasets(category: Optional[str] = None) -> List[str]:
    """List all registered dataset keys, optionally filtered by category.

    Args:
        category: One of the DataCategory values (e.g. 'ephemeris_catalogs')
    """
    if category is None:
        return sorted(DATASET_REGISTRY.keys())
    return sorted(
        k for k, v in DATASET_REGISTRY.items()
        if v.category.value == category
    )


def dataset_info(key: str) -> Dict[str, Any]:
    """Return metadata dict for a registered dataset."""
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {key}")
    meta = DATASET_REGISTRY[key]
    return {
        "key": key,
        "hf_id": meta.hf_id,
        "category": meta.category.value,
        "is_large": meta.is_large,
        "default_split": meta.split,
        "subset": meta.subset,
    }


def list_categories() -> List[str]:
    """Return all available pillar category names."""
    return [c.value for c in DataCategory]


__all__ = [
    "get_dataset",
    "list_datasets",
    "dataset_info",
    "list_categories",
    "loader",
]
