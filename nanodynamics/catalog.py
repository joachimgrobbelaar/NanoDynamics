"""
nanodynamics/catalog.py

Human-readable catalog of all available datasets and analytical functions.
"""

from __future__ import annotations

from nanodynamics.data_ingestion.loaders import DATASET_REGISTRY, DataCategory


def print_catalog() -> str:
    """Return a formatted catalog string of all registered datasets."""
    lines = ["=" * 72, "NanoDynamics Dataset Catalog", "=" * 72, ""]

    for cat in DataCategory:
        entries = [
            (k, v) for k, v in DATASET_REGISTRY.items()
            if v.category == cat
        ]
        if not entries:
            continue
        lines.append(f"── {cat.value.upper().replace('_', ' ')} ──")
        for key, meta in sorted(entries):
            stream_tag = " [STREAMING]" if meta.is_large else ""
            lines.append(f"  {key:40s} {meta.hf_id}{stream_tag}")
        lines.append("")

    catalog_text = "\n".join(lines)
    return catalog_text


__all__ = ["print_catalog"]
