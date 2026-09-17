"""Satellite configuration and physical properties data structures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Satellite:
    """Nanosatellite physical specifications for orbital propagation.

    Attributes:
        name: Spacecraft name or identifier.
        mass: Satellite mass in kg.
        drag_area: Cross-sectional drag area in m^2.
        cd: Aerodynamic drag coefficient (typically 2.0 - 2.5 for LEO).
    """

    name: str = "Nanosat-3U"
    mass: float = 4.0        # kg
    drag_area: float = 0.03   # m^2 (0.1 m x 0.3 m)
    cd: float = 2.2

    def __post_init__(self) -> None:
        if not (self.mass > 0.0):
            raise ValueError(f"Satellite mass must be positive and finite, got {self.mass}")
        if not (self.drag_area >= 0.0):
            raise ValueError(f"Drag area cannot be negative, got {self.drag_area}")
        if not (self.cd >= 0.0):
            raise ValueError(f"Drag coefficient cannot be negative, got {self.cd}")

    @property
    def ballistic_coefficient(self) -> float:
        """Compute ballistic coefficient B = mass / (cd * area) in kg / m^2.

        Returns infinity if drag area or cd is zero.
        """
        denominator = self.cd * self.drag_area
        if denominator == 0.0:
            return float("inf")
        return self.mass / denominator

    @classmethod
    def cubesat_1u(cls, name: str = "CubeSat-1U", mass: float = 1.33, cd: float = 2.2) -> "Satellite":
        """Factory for a standard 1U CubeSat (0.1m x 0.1m cross-section)."""
        return cls(name=name, mass=mass, drag_area=0.01, cd=cd)

    @classmethod
    def cubesat_3u(cls, name: str = "CubeSat-3U", mass: float = 4.0, cd: float = 2.2) -> "Satellite":
        """Factory for a standard 3U CubeSat (0.1m x 0.3m cross-section)."""
        return cls(name=name, mass=mass, drag_area=0.03, cd=cd)

    @classmethod
    def cubesat_6u(cls, name: str = "CubeSat-6U", mass: float = 8.0, cd: float = 2.2) -> "Satellite":
        """Factory for a standard 6U CubeSat (0.2m x 0.3m cross-section)."""
        return cls(name=name, mass=mass, drag_area=0.06, cd=cd)
