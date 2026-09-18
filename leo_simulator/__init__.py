"""LEO Nanosatellite Orbital Dynamics Simulator.

Stage 1 baseline numerical orbit propagator supporting central gravity,
Earth J2 oblateness, and aerodynamic drag.
"""

from leo_simulator.constants import (
    J2_EARTH,
    MU_EARTH,
    MU_MOON,
    OMEGA_EARTH,
    R_EARTH,
    R_MOON_ORBIT,
    T_MOON_ORBIT,
)
from leo_simulator.models.drag import (
    BaseAtmosphere,
    ExponentialAtmosphere,
    PiecewiseExponentialAtmosphere,
    aerodynamic_drag_acceleration,
    relative_velocity_vector,
)
from leo_simulator.models.dynamics import OrbitalDynamics
from leo_simulator.models.gravity import (
    central_gravity_acceleration,
    j2_perturbation_acceleration,
    moon_position,
    third_body_acceleration,
    total_gravity_acceleration,
)
from leo_simulator.orbit.elements import (
    OrbitalElements,
    analytical_j2_raan_rate,
    circular_velocity,
    coe_to_rv,
    rv_to_coe,
)
from leo_simulator.orbit.satellite import Satellite
from leo_simulator.propagator import OrbitPropagator, PropagationResult

__version__ = "0.1.0"

__all__ = [
    "J2_EARTH",
    "MU_EARTH",
    "MU_MOON",
    "OMEGA_EARTH",
    "R_EARTH",
    "R_MOON_ORBIT",
    "T_MOON_ORBIT",
    "BaseAtmosphere",
    "ExponentialAtmosphere",
    "OrbitPropagator",
    "OrbitalDynamics",
    "OrbitalElements",
    "PiecewiseExponentialAtmosphere",
    "PropagationResult",
    "Satellite",
    "aerodynamic_drag_acceleration",
    "analytical_j2_raan_rate",
    "central_gravity_acceleration",
    "circular_velocity",
    "coe_to_rv",
    "j2_perturbation_acceleration",
    "moon_position",
    "relative_velocity_vector",
    "rv_to_coe",
    "third_body_acceleration",
    "total_gravity_acceleration",
]
