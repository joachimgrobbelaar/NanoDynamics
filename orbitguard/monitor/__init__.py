"""OrbitGuard LEO monitoring workspace.

This package is the monitoring component beside NanoDynamics. It ingests
public general-perturbations data, propagates those mean elements with
SGP4, and reuses ``leo_simulator.OrbitPropagator`` only for Cartesian
states that the caller declares to be osculating.

It does not compute a collision probability or an expected collision count.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
