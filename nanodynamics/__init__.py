"""NanoDynamics astrodynamics library.

Public physics API, with no web framework imported from these modules:

* ``nanodynamics.elements`` — GP/OMM validation, TLE checksums, Alpha-5
* ``nanodynamics.frames`` — GMST and the TEME display transform
* ``nanodynamics.propagation`` — SGP4 for mean elements, and the numerical
  integrator for labelled osculating Cartesian states

Dataset helpers stay in ``nanodynamics.data_ingestion`` and are loaded only
when requested, because they need the optional ``data`` extra.
"""

__version__ = "1.1.0"

_DATA_EXPORTS = {
    "get_dataset": "get_dataset",
    "loader": "loader",
    "DATASET_REGISTRY": "DATASET_REGISTRY",
    "DataCategory": "DataCategory",
    "DatasetMeta": "DatasetMeta",
    "NanodynamicsDataLoader": "NanodynamicsDataLoader",
    "KeplerianElementRecord": "KeplerianElementRecord",
    "SpaceWeatherRecord": "SpaceWeatherRecord",
    "ConjunctionEventRecord": "ConjunctionEventRecord",
}

__all__ = ["__version__", *list(_DATA_EXPORTS)]


def __getattr__(name: str):
    if name in _DATA_EXPORTS:
        from nanodynamics.data_ingestion import loaders

        return getattr(loaders, _DATA_EXPORTS[name])
    # Submodule imports (nanodynamics.data_ingestion, nanodynamics.propagation, …)
    # go through this hook once it exists. Load a real submodule, and leave
    # every other name missing.
    import importlib

    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__():
    return sorted(set(globals()) | set(_DATA_EXPORTS))
