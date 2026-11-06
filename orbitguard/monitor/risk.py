"""Geometric conjunction screening. Pc and Nc are not computed.

Hall (2021) defines Nc as an expected number of collisions. For an isolated
encounter the paper states that Nc equals Pc under its assumptions; for
repeated encounters Nc can exceed Pc. This module does not evaluate either
quantity and does not relabel one as the other. A missing covariance is
not reported as zero and not reported as safe.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

HALL_DOI = "https://doi.org/10.2514/1.A34919"
NC_NOTE = (
    "Nc is the expected number of collisions in Hall, Journal of Spacecraft "
    "and Rockets (2021), doi:10.2514/1.A34919. The abstract states that Nc "
    "equals Pc for an isolated encounter under that paper's assumptions, and "
    "that Nc can exceed Pc when encounters repeat. OrbitGuard does not compute "
    "Nc, does not convert Nc into a probability, and does not send a failed "
    "2D-Pc case to 3D Nc. 3D Nc also does not repair an invalid or "
    "non-Gaussian covariance."
)
PC_NOTE = (
    "2D collision probability is not computed. Public GP data include no "
    "calibrated covariance, and this version does not implement a Pc "
    "integral. A missing or invalid covariance is not displayed as 0 and "
    "is not displayed as safe."
)


def assess_covariance(
    matrix: Any,
    *,
    state_epoch: Optional[str],
    covariance_epoch: Optional[str],
    state_frame: Optional[str],
    covariance_frame: Optional[str],
) -> dict[str, Any]:
    """Check shape, symmetry, epoch, frame, and positive-semidefiniteness.

    A matrix that passes these checks is still not propagated: there is no
    state-transition matrix and no process-noise model in this version.
    Pc remains not computable either way.
    """
    base = {
        "pc": None,
        "pc_status": "not_computable",
        "pc_reason": PC_NOTE,
        "nc": None,
        "nc_status": "not_implemented",
        "nc_reason": NC_NOTE,
        "displayed_as_safe": False,
        "propagated": False,
    }
    if matrix is None:
        return {
            **base,
            "status": "absent",
            "reason": "No covariance was supplied. Public GP/OMM has none.",
            "positive_semidefinite": None,
        }
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] not in (3, 6):
        return {**base, "status": "invalid", "reason": "Covariance must be a finite 3×3 or 6×6 matrix.", "positive_semidefinite": False}
    if not np.all(np.isfinite(array)):
        return {**base, "status": "invalid", "reason": "Covariance contains a non-finite value.", "positive_semidefinite": False}
    if not np.allclose(array, array.T, rtol=1e-6, atol=1e-8):
        return {**base, "status": "invalid", "reason": "Covariance is not symmetric.", "positive_semidefinite": False}
    eigenvalues = np.linalg.eigvalsh(array)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if np.any(eigenvalues < -1e-8 * scale):
        return {
            **base,
            "status": "invalid",
            "reason": "Covariance is not positive semidefinite. It is not treated as a safe or zero risk.",
            "positive_semidefinite": False,
            "min_eigenvalue": float(eigenvalues.min()),
        }
    if not covariance_epoch or not state_epoch or covariance_epoch != state_epoch:
        return {
            **base,
            "status": "invalid",
            "reason": "Covariance epoch does not match the state epoch. This version does not propagate covariance.",
            "positive_semidefinite": True,
            "min_eigenvalue": float(eigenvalues.min()),
        }
    if not covariance_frame or covariance_frame != state_frame:
        return {
            **base,
            "status": "invalid",
            "reason": "Covariance frame is missing or differs from the state frame.",
            "positive_semidefinite": True,
            "min_eigenvalue": float(eigenvalues.min()),
        }
    return {
        **base,
        "status": "present_not_propagated",
        "reason": (
            "The matrix is finite, symmetric, and positive semidefinite, and "
            "its epoch and frame match the state. No STM or process noise is "
            "applied, so it is not a propagated uncertainty, and Pc is still "
            "not computed."
        ),
        "positive_semidefinite": True,
        "min_eigenvalue": float(eigenvalues.min()),
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "frame": covariance_frame,
        "epoch": covariance_epoch,
    }


def geometric_screen(
    primary: Sequence[dict[str, Any]],
    secondary: Sequence[dict[str, Any]],
    *,
    frame: str,
    threshold_km: float,
) -> dict[str, Any]:
    """Minimum sampled separation. Not a probability."""
    by_time = {row["time"]: row for row in secondary if row.get("r_teme_km") or row.get("r_km")}
    best = None
    for row in primary:
        other = by_time.get(row["time"])
        if other is None:
            continue
        key = "r_teme_km" if "r_teme_km" in row else "r_km"
        if key not in other:
            continue
        r1 = np.asarray(row[key], dtype=np.float64)
        r2 = np.asarray(other[key], dtype=np.float64)
        miss = float(np.linalg.norm(r1 - r2))
        if best is None or miss < best["miss_km"]:
            v1 = np.asarray(row.get("v_teme_km_s") or row.get("v_km_s"), dtype=np.float64)
            v2 = np.asarray(other.get("v_teme_km_s") or other.get("v_km_s"), dtype=np.float64)
            relative = r2 - r1
            radial_hat = r1 / max(float(np.linalg.norm(r1)), 1e-12)
            cross_hat = np.cross(r1, v1)
            cross_norm = float(np.linalg.norm(cross_hat))
            if cross_norm < 1e-9:
                components = None
            else:
                cross_hat = cross_hat / cross_norm
                along_hat = np.cross(cross_hat, radial_hat)
                components = {
                    "radial_km": float(np.dot(relative, radial_hat)),
                    "along_track_km": float(np.dot(relative, along_hat)),
                    "cross_track_km": float(np.dot(relative, cross_hat)),
                    "frame": f"{frame} RTN of the primary at the sample",
                    "status": "derived",
                }
            best = {
                "time": row["time"],
                "miss_km": miss,
                "relative_speed_km_s": float(np.linalg.norm(v1 - v2)),
                "components": components,
            }
    listed = best is not None and best["miss_km"] <= threshold_km
    return {
        "frame": frame,
        "sample_count": 0 if best is None else 1,
        "geometric_miss_km": None if best is None else best["miss_km"],
        "time_of_closest_sample": None if best is None else best["time"],
        "relative_speed_km_s": None if best is None else best["relative_speed_km_s"],
        "rtn_components": None if best is None else best["components"],
        "within_display_threshold": listed if best is not None else False,
        "threshold_km": threshold_km,
        "threshold_note": (
            "Distances above this value are left off the short list. The "
            "threshold is a product display setting, not a safety criterion, "
            "and a larger miss is not a collision probability of zero."
        ),
        "screening_kind": "sampled_geometric_separation",
        "pc": None,
        "pc_status": "not_computable",
        "pc_reason": PC_NOTE,
        "nc": None,
        "nc_status": "not_implemented",
        "nc_reason": NC_NOTE,
        "safety_determination": "not_made",
        "risk_tier": None,
        "advisory": "Human-reviewed display only. No maneuver is computed or commanded.",
        "visualization_note": (
            "No uncertainty ellipsoid or Gaussian splat is drawn. A picture of "
            "a cloud would not be a calibrated collision probability."
        ),
        "escalation": {
            "implemented": False,
            "plan": (
                "A later version may add a validated 2D Pc only when covariance, "
                "hard-body radius, and encounter geometry are all present and the "
                "implementation has been checked. 3D Nc (NASA GSC-18593-1; Hall 2021) "
                "estimates an expected collision count and can relax the short-encounter "
                "assumption; it is not an automatic next step when 2D assumptions fail, "
                "and it does not accept an invalid covariance. Monte Carlo remains a "
                "further, separate method. None of these are computed now."
            ),
            "references": [
                HALL_DOI,
                "https://software.nasa.gov/software/GSC-18593-1",
                "https://ntrs.nasa.gov/api/citations/20205007226/downloads/CARA_3D-NC%20and%20BFMC%20Presentation_ATrainMOWG_Fall2020_Final.pdf",
                "https://ntrs.nasa.gov/api/citations/20240003468/downloads/CA_Hanbook_Appendix.pdf",
            ],
        },
    }
