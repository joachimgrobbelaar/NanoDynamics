"""Astrodynamics core: IAU reference frames, multi-body force models, and TRL 1-9 propagators."""

from src.astrodynamics.force_models import (
    EARTH_J2,
    EARTH_J3,
    EARTH_J4,
    MU_EARTH,
    MU_MOON,
    MU_SUN,
    OMEGA_EARTH,
    R_EARTH,
    acceleration_cr3bp_synodic,
    acceleration_drag_nrlmsise00,
    acceleration_j2_j4,
    acceleration_srp,
    acceleration_third_body_ems,
    compute_jacobi_constant,
    total_gcrf_acceleration,
)
from src.astrodynamics.frames import ecef_to_gcrf, gcrf_to_ecef, rv_to_osculating_coe
from src.astrodynamics.propagators import NumericalPropagator, PropagationTrajectory

__all__ = [
    "MU_EARTH",
    "MU_MOON",
    "MU_SUN",
    "R_EARTH",
    "OMEGA_EARTH",
    "EARTH_J2",
    "EARTH_J3",
    "EARTH_J4",
    "acceleration_j2_j4",
    "acceleration_drag_nrlmsise00",
    "acceleration_srp",
    "acceleration_third_body_ems",
    "acceleration_cr3bp_synodic",
    "compute_jacobi_constant",
    "total_gcrf_acceleration",
    "ecef_to_gcrf",
    "gcrf_to_ecef",
    "rv_to_osculating_coe",
    "NumericalPropagator",
    "PropagationTrajectory",
]
