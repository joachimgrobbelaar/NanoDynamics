"""Validation of Symplectic Hamiltonian Manifold projection, CR3BP Jacobi conservation, and INT8 export."""

from pathlib import Path
import numpy as np

from src.astrodynamics.force_models import MU_EARTH, R_EARTH
from src.astrodynamics.propagators import NumericalPropagator
from src.export.tflite_quantizer import quantize_and_export_header
from src.ml.pinn_loss import hamiltonian_manifold_project
from src.ml.trainer import train_pinn_surrogate


def test_symplectic_hamiltonian_manifold_projection_machine_precision():
    r_next = np.array([R_EARTH + 550_000.0, 120_000.0, -80_000.0], dtype=np.float64)
    r_mag = float(np.linalg.norm(r_next))
    v_circ = np.sqrt(MU_EARTH / r_mag)
    e_target = 0.5 * (v_circ**2) - MU_EARTH / r_mag

    # Perturbed neural network output with 3.5% velocity magnitude error
    v_nn = np.array([0.0, v_circ * 1.035, 150.0], dtype=np.float64)
    v_proj = hamiltonian_manifold_project(r_next, v_nn, e_target_j_kg=e_target, mu=MU_EARTH)

    e_proj = 0.5 * float(np.dot(v_proj, v_proj)) - MU_EARTH / r_mag
    rel_energy_error = abs(e_proj - e_target) / abs(e_target)
    assert rel_energy_error < 1e-14


def test_cr3bp_jacobi_constant_conservation():
    prop = NumericalPropagator()
    state0_nd = np.array([0.85, 0.0, 0.05, 0.0, 0.22, 0.0], dtype=np.float64)
    res = prop.propagate_cr3bp(state0_nd, duration_nd=2.5, n_points=120)
    assert res["max_rel_jacobi_drift"] < 1e-9


def test_pinn_training_and_int8_header_generation(tmp_path: Path):
    model, losses = train_pinn_surrogate(epochs=12, lr=5e-3)
    assert losses[-1] < losses[0]

    header_out = tmp_path / "transition_model_data.h"
    info = quantize_and_export_header(model, header_out)
    assert header_out.is_file()
    assert info["bytes_written"] > 1000
    text = header_out.read_text(encoding="utf-8")
    assert "alignas(16) const unsigned char transition_tflite[]" in text
