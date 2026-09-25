"""Physics-Informed Neural Network (PINN) surrogate orbit propagator.

Propagates orbital states autoregressively using a trained PyTorch residual
transition model (S_t -> S_{t+dt}), enabling ultra-fast real-time inference
without numerical ODE integration.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from leo_simulator.constants import MU_EARTH, OMEGA_EARTH, R_EARTH
from leo_simulator.models.drag import (
    ExponentialAtmosphere,
    aerodynamic_drag_acceleration,
)
from leo_simulator.models.dynamics import OrbitalDynamics
from leo_simulator.orbit.elements import OrbitalElements, coe_to_rv, rv_to_coe
from leo_simulator.orbit.satellite import Satellite
from leo_simulator.propagator import PropagationResult

if TYPE_CHECKING:
    from torch import nn


class _TransitionMLP(torch.nn.Module):
    """Local MLP definition matching ai.transition_model.TransitionMLP."""

    def __init__(self, hidden_dim: int = 32, hidden_layers: int = 2):
        super().__init__()
        layers = []
        dim = 6
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(dim, hidden_dim))
            layers.append(torch.nn.Tanh())
            dim = hidden_dim
        layers.append(torch.nn.Linear(dim, 6))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PINNPropagator:
    """Surrogate orbit propagator powered by a trained transition neural network."""

    _cached_model: Optional[_TransitionMLP] = None
    _cached_stats: Optional[dict] = None
    _cached_ckpt_path: Optional[str] = None

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        satellite: Optional[Satellite] = None,
        mu: float = MU_EARTH,
        r_earth: float = R_EARTH,
        omega_earth: float = OMEGA_EARTH,
        min_altitude_reentry: float = 50_000.0,
    ) -> None:
        self.mu = mu
        self.r_earth = r_earth
        self.omega_earth = omega_earth
        self.min_altitude_reentry = min_altitude_reentry
        self.satellite = satellite if satellite is not None else Satellite.cubesat_3u()

        # Dynamics instance for physical force vector computation along surrogate path
        self.dynamics = OrbitalDynamics(
            cd=self.satellite.cd,
            area=self.satellite.drag_area,
            mass=self.satellite.mass,
            atmosphere=ExponentialAtmosphere(),
            mu=self.mu,
            r_earth=self.r_earth,
            j2=1.08263e-3,
            omega_earth=self.omega_earth,
            include_central_gravity=True,
            include_j2=True,
            include_drag=True,
            include_earth_rotation=True,
            include_moon=False,
        )

        self.checkpoint_path = checkpoint_path or self._resolve_default_checkpoint()
        self._load_model()

    @staticmethod
    def _resolve_default_checkpoint() -> str:
        # Search relative to repo root
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates = [
            os.path.join(root, "ai", "checkpoints", "agent", "transition.pt"),
            os.path.join(root, "ai", "checkpoints", "transition.pt"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    def _load_model(self) -> None:
        if (
            PINNPropagator._cached_model is not None
            and PINNPropagator._cached_ckpt_path == self.checkpoint_path
        ):
            self.model = PINNPropagator._cached_model
            self.stats = PINNPropagator._cached_stats
            return

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"PINN transition model checkpoint not found at: {self.checkpoint_path}"
            )

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {"hidden_dim": 32, "hidden_layers": 2})
        stats = ckpt.get("stats", {"pos_scale": 6378137.0, "vel_scale": 8000.0, "dt": 60.0})

        model = _TransitionMLP(
            hidden_dim=cfg.get("hidden_dim", 32),
            hidden_layers=cfg.get("hidden_layers", 2),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        PINNPropagator._cached_model = model
        PINNPropagator._cached_stats = stats
        PINNPropagator._cached_ckpt_path = self.checkpoint_path

        self.model = model
        self.stats = stats

    def propagate(
        self,
        orbit: Optional[Union[OrbitalElements, np.ndarray, list]] = None,
        duration_seconds: float = 86400.0,
        t_start: float = 0.0,
        dt_eval: float = 60.0,
        initial_state: Optional[Union[np.ndarray, list]] = None,
    ) -> PropagationResult:
        """Autoregressively roll out trajectory using the PINN transition model."""
        target = initial_state if initial_state is not None else orbit
        if target is None:
            raise ValueError("Either orbit or initial_state must be provided.")
        if isinstance(target, OrbitalElements):
            r0, v0 = coe_to_rv(target, mu=self.mu)
            state0 = np.concatenate([r0, v0]).astype(np.float64)
        else:
            state0 = np.asarray(target, dtype=np.float64).flatten()
            if len(state0) != 6:
                raise ValueError(f"State vector must have length 6, got {len(state0)}")

        dt_model = float(self.stats.get("dt", 60.0))
        # Use model's native dt if dt_eval <= 0
        step_dt = dt_eval if dt_eval > 0 else dt_model

        pos_scale = float(self.stats.get("pos_scale", 6378137.0))
        vel_scale = float(self.stats.get("vel_scale", 8000.0))
        sc = torch.tensor(
            [pos_scale] * 3 + [vel_scale] * 3, dtype=torch.float32
        )

        n_steps = max(1, int(round(duration_seconds / step_dt)))
        
        t_list = [float(t_start)]
        r_list = [state0[0:3].copy()]
        v_list = [state0[3:6].copy()]
        alt_list = [float(np.linalg.norm(state0[0:3]) - self.r_earth)]
        speed_list = [float(np.linalg.norm(state0[3:6]))]

        reentry_detected = False
        reentry_time = None

        curr_r = state0[0:3].copy()
        curr_v = state0[3:6].copy()
        atm = ExponentialAtmosphere()

        with torch.no_grad():
            curr_t = float(t_start)
            for _ in range(n_steps):
                r_mag = float(np.linalg.norm(curr_r))
                v_mag = float(np.linalg.norm(curr_v))
                alt_curr = r_mag - self.r_earth

                E_k = (v_mag**2) / 2.0 - self.mu / r_mag
                if alt_curr < 1_000_000.0:
                    a_drag = aerodynamic_drag_acceleration(
                        curr_r,
                        curr_v,
                        mass=self.satellite.mass,
                        area=self.satellite.drag_area,
                        cd=self.satellite.cd,
                        atmosphere_model=atm,
                        r_earth=self.r_earth,
                        omega_earth=self.omega_earth,
                    )
                    dE = float(np.dot(curr_v, a_drag)) * step_dt
                else:
                    dE = 0.0
                E_target = E_k + dE

                state_norm = (
                    torch.tensor(np.concatenate([curr_r, curr_v]), dtype=torch.float32) / sc
                ).unsqueeze(0)
                next_norm = self.model(state_norm).squeeze(0)
                unscaled = (next_norm * sc).numpy().astype(np.float64)

                r_nn = unscaled[0:3]
                v_nn = unscaled[3:6]

                r_kin = curr_r + 0.5 * (curr_v + v_nn) * step_dt
                r_next = 0.5 * r_nn + 0.5 * r_kin
                r_next_mag = float(np.linalg.norm(r_next))

                v_sq_target = 2.0 * (E_target + self.mu / r_next_mag)
                if v_sq_target > 0.0:
                    v_nn_norm = float(np.linalg.norm(v_nn))
                    v_next = v_nn * (np.sqrt(v_sq_target) / v_nn_norm) if v_nn_norm > 0 else v_nn
                else:
                    v_next = v_nn

                curr_t += step_dt
                curr_r = r_next
                curr_v = v_next
                alt_k = r_next_mag - self.r_earth

                t_list.append(curr_t)
                r_list.append(curr_r.copy())
                v_list.append(curr_v.copy())
                alt_list.append(alt_k)
                speed_list.append(float(np.linalg.norm(curr_v)))

                if alt_k <= self.min_altitude_reentry:
                    reentry_detected = True
                    reentry_time = curr_t
                    break

        r_arr = np.array(r_list, dtype=np.float64)
        v_arr = np.array(v_list, dtype=np.float64)
        t_arr = np.array(t_list, dtype=np.float64)
        alt_arr = np.array(alt_list, dtype=np.float64)
        speed_arr = np.array(speed_list, dtype=np.float64)

        # Approximate orbital elements for container completeness
        sma_arr = r_arr[:, 0] * 0.0 + (float(np.linalg.norm(r_arr[0])) if len(r_arr) > 0 else self.r_earth)
        ecc_arr = np.zeros_like(sma_arr)
        inc_arr = np.zeros_like(sma_arr)
        raan_arr = np.zeros_like(sma_arr)
        arg_pe_arr = np.zeros_like(sma_arr)
        nu_arr = np.zeros_like(sma_arr)

        return PropagationResult(
            t=t_arr,
            r=r_arr,
            v=v_arr,
            altitudes=alt_arr,
            speeds=speed_arr,
            semi_major_axes=sma_arr,
            eccentricities=ecc_arr,
            inclinations=inc_arr,
            raans=raan_arr,
            arg_pes=arg_pe_arr,
            true_anomalies=nu_arr,
            success=True,
            status=0,
            message="PINN rollout successful",
            reentry_detected=reentry_detected,
            reentry_time=reentry_time,
        )
