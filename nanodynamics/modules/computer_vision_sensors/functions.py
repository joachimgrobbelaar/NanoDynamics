"""
nanodynamics/modules/computer_vision_sensors/functions.py

Pillar 6: Computer Vision, Spatial Imagery & Kinematics.

Provides data-loading helpers and physics / image-analysis functions for:
    - Spacecraft pose estimation error metrics
    - 3-D cloud-field occlusion-mask generation via ray-cast geometry
    - Urban-footprint segmentation from satellite imagery
    - Temporal change-vector analysis on aerial / satellite image pairs
    - ML-surrogate molecular-orbital integration (stub / linear extrapolation)

All numerical work uses NumPy; units follow SI/metric conventions
(m, km, s, rad, kg, W/m²) unless noted in individual docstrings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from nanodynamics.core.constants import *          # noqa: F401, F403
from nanodynamics.data_ingestion.loaders import get_dataset

logger = logging.getLogger(__name__)

__all__ = [
    # ── Spacecraft Pose ───────────────────────────────────────────────────
    "load_spacecraft_pose_dataset",
    "evaluate_pose_error",
    # ── 3-D Cloud Occlusion ───────────────────────────────────────────────
    "load_3d_clouds_subset",
    "project_cloud_occlusion_mask",
    # ── Urban Footprints ──────────────────────────────────────────────────
    "load_city_satellite_imagery",
    "extract_urban_footprints",
    # ── Temporal Change ───────────────────────────────────────────────────
    "load_temporal_cityline",
    "compute_temporal_change_vector",
    # ── ML-Accelerated Integration ────────────────────────────────────────
    "load_timewarp_dynamics",
    "accelerate_molecular_orbital_integrator",
]


# ============================================================================
# 1. Spacecraft Pose Estimation
# ============================================================================

def load_spacecraft_pose_dataset(**kwargs: Any) -> Any:
    """Load the multi-target spacecraft pose-estimation dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
        The HuggingFace dataset object keyed as
        ``spacecraft_pose_estimation`` in the NanoDynamics registry.
    """
    logger.info("Loading spacecraft pose-estimation dataset …")
    return get_dataset("spacecraft_pose_estimation", **kwargs)


def evaluate_pose_error(
    pred_translation: NDArray[np.floating],
    pred_quaternion: NDArray[np.floating],
    gt_translation: NDArray[np.floating],
    gt_quaternion: NDArray[np.floating],
) -> Dict[str, float]:
    """Compute translation and rotational pose-estimation errors.

    Parameters
    ----------
    pred_translation : ndarray, shape (3,)
        Predicted position vector [m].
    pred_quaternion : ndarray, shape (4,)
        Predicted attitude quaternion (scalar-last **or** scalar-first —
        the metric is invariant to convention since we use the absolute
        dot-product).
    gt_translation : ndarray, shape (3,)
        Ground-truth position vector [m].
    gt_quaternion : ndarray, shape (4,)
        Ground-truth attitude quaternion.

    Returns
    -------
    dict
        ``translation_error_m`` – Euclidean distance [m].
        ``angular_error_rad``   – Geodesic angular distance [rad],
        computed as θ = 2 · arccos(|q₁ · q₂|).
        ``angular_error_deg``   – Same, converted to degrees.
    """
    pred_t = np.asarray(pred_translation, dtype=np.float64)
    gt_t = np.asarray(gt_translation, dtype=np.float64)
    pred_q = np.asarray(pred_quaternion, dtype=np.float64)
    gt_q = np.asarray(gt_quaternion, dtype=np.float64)

    if pred_t.shape != (3,) or gt_t.shape != (3,):
        raise ValueError(
            f"Translation vectors must have shape (3,); "
            f"got pred={pred_t.shape}, gt={gt_t.shape}"
        )
    if pred_q.shape != (4,) or gt_q.shape != (4,):
        raise ValueError(
            f"Quaternions must have shape (4,); "
            f"got pred={pred_q.shape}, gt={gt_q.shape}"
        )

    # ── Translation error (Euclidean norm) ────────────────────────────────
    translation_error: float = float(np.linalg.norm(pred_t - gt_t))

    # ── Angular error (quaternion geodesic distance) ──────────────────────
    # Normalise to unit quaternions for numerical safety.
    pred_q = pred_q / (np.linalg.norm(pred_q) + 1e-15)
    gt_q = gt_q / (np.linalg.norm(gt_q) + 1e-15)

    dot = np.clip(np.abs(np.dot(pred_q, gt_q)), 0.0, 1.0)
    angular_error_rad: float = float(2.0 * np.arccos(dot))
    angular_error_deg: float = float(np.degrees(angular_error_rad))

    logger.debug(
        "Pose error  ‖Δt‖=%.4f m  Δθ=%.4f rad (%.2f°)",
        translation_error,
        angular_error_rad,
        angular_error_deg,
    )

    return {
        "translation_error_m": translation_error,
        "angular_error_rad": angular_error_rad,
        "angular_error_deg": angular_error_deg,
    }


# ============================================================================
# 2. 3-D Cloud Occlusion Mask
# ============================================================================

def load_3d_clouds_subset(**kwargs: Any) -> Any:
    """Load the NASA downstream 3-D cloud subset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
        The HuggingFace dataset keyed as ``downstream_3dclouds``.
    """
    logger.info("Loading downstream 3-D clouds subset …")
    return get_dataset("downstream_3dclouds", **kwargs)


def project_cloud_occlusion_mask(
    cloud_points_3d: NDArray[np.floating],
    camera_position: NDArray[np.floating],
    camera_direction: NDArray[np.floating],
    fov_deg: float = 60.0,
) -> NDArray[np.bool_]:
    """Compute a binary occlusion / visibility mask via FOV-cone ray-cast.

    For each 3-D cloud point the function checks whether the point lies
    inside the camera's conical field-of-view.  A point is *visible*
    (mask = True) when:

    1. The vector from the camera to the point has a positive projection
       along the camera look-direction (i.e. the point is *in front* of
       the camera).
    2. The half-angle between that vector and the look-direction is
       less than or equal to ``fov_deg / 2``.

    Parameters
    ----------
    cloud_points_3d : ndarray, shape (N, 3)
        Point-cloud coordinates [m] or [km] — unit-agnostic.
    camera_position : ndarray, shape (3,)
        Camera origin in the same frame as *cloud_points_3d*.
    camera_direction : ndarray, shape (3,)
        Unit (or un-normalised) look-direction of the camera.
    fov_deg : float, default 60.0
        Total field-of-view cone angle [deg].

    Returns
    -------
    ndarray of bool, shape (N,)
        ``True`` where the point is inside the FOV cone.
    """
    pts = np.asarray(cloud_points_3d, dtype=np.float64)
    cam_pos = np.asarray(camera_position, dtype=np.float64).ravel()
    cam_dir = np.asarray(camera_direction, dtype=np.float64).ravel()

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"cloud_points_3d must have shape (N, 3); got {pts.shape}"
        )

    # Normalise look-direction
    cam_dir_norm = cam_dir / (np.linalg.norm(cam_dir) + 1e-15)

    # Vectors from camera to each point
    delta = pts - cam_pos[np.newaxis, :]                      # (N, 3)
    dist = np.linalg.norm(delta, axis=1, keepdims=True)       # (N, 1)
    delta_unit = delta / (dist + 1e-15)                       # (N, 3)

    # Cosine of the angle between each ray and the camera direction
    cos_angle = np.dot(delta_unit, cam_dir_norm)               # (N,)

    # Half-angle threshold
    cos_half_fov = np.cos(np.radians(fov_deg / 2.0))

    # Visibility: in front of camera AND within the FOV cone
    in_front = np.dot(delta, cam_dir_norm) > 0.0               # (N,)
    within_fov = cos_angle >= cos_half_fov                     # (N,)
    mask: NDArray[np.bool_] = np.logical_and(in_front, within_fov)

    n_visible = int(np.sum(mask))
    logger.debug(
        "Occlusion mask: %d / %d points visible (FOV=%.1f°)",
        n_visible,
        len(mask),
        fov_deg,
    )
    return mask


# ============================================================================
# 3. Urban Footprint Extraction
# ============================================================================

def load_city_satellite_imagery(**kwargs: Any) -> Any:
    """Load city satellite imagery dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
        The HuggingFace dataset keyed as ``city_satellite``.
    """
    logger.info("Loading city satellite imagery …")
    return get_dataset("city_satellite", **kwargs)


def extract_urban_footprints(
    image_array: NDArray[np.integer | np.floating],
    threshold: int = 128,
) -> Tuple[NDArray[np.bool_], float]:
    """Segment urban areas from a satellite image via intensity thresholding.

    A simple but effective baseline: pixels whose intensity exceeds
    *threshold* are classified as urban.

    Parameters
    ----------
    image_array : ndarray, shape (H, W) or (H, W, C)
        Grayscale or multi-channel satellite image (uint8 expected, but
        any numeric dtype is accepted).  For colour images the mean
        across channels is used as the luminance proxy.
    threshold : int, default 128
        Pixel intensity threshold (0–255 scale).

    Returns
    -------
    binary_mask : ndarray of bool, shape (H, W)
        ``True`` for pixels classified as urban.
    coverage_fraction : float
        Fraction of the image classified as urban (0.0–1.0).
    """
    img = np.asarray(image_array, dtype=np.float64)

    # Collapse to single-channel if necessary
    if img.ndim == 3:
        img = np.mean(img, axis=-1)
    elif img.ndim != 2:
        raise ValueError(
            f"image_array must be 2-D or 3-D; got ndim={img.ndim}"
        )

    binary_mask: NDArray[np.bool_] = img > float(threshold)
    total_pixels = binary_mask.size
    urban_pixels = int(np.sum(binary_mask))
    coverage_fraction = urban_pixels / total_pixels if total_pixels > 0 else 0.0

    logger.debug(
        "Urban footprint: %d / %d px (%.2f %% coverage)",
        urban_pixels,
        total_pixels,
        coverage_fraction * 100.0,
    )
    return binary_mask, coverage_fraction


# ============================================================================
# 4. Temporal Change-Vector Analysis
# ============================================================================

def load_temporal_cityline(**kwargs: Any) -> Any:
    """Load temporal aerial cityline construction-sample dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
        The HuggingFace dataset keyed as ``temporal_cityline``.
    """
    logger.info("Loading temporal aerial cityline dataset …")
    return get_dataset("temporal_cityline", **kwargs)


def compute_temporal_change_vector(
    t1_array: NDArray[np.integer | np.floating],
    t2_array: NDArray[np.integer | np.floating],
) -> Tuple[NDArray[np.floating], float]:
    """Compute pixel-wise change between two co-registered images.

    Parameters
    ----------
    t1_array : ndarray, shape (H, W) or (H, W, C)
        Image captured at time *t₁*.
    t2_array : ndarray, shape (H, W) or (H, W, C)
        Image captured at time *t₂*, same shape as *t1_array*.

    Returns
    -------
    change_magnitude : ndarray of float64, shape (H, W)
        Per-pixel absolute difference magnitude.  For multi-channel
        images this is the Euclidean norm across channels.
    mean_change : float
        Spatial mean of *change_magnitude*.
    """
    t1 = np.asarray(t1_array, dtype=np.float64)
    t2 = np.asarray(t2_array, dtype=np.float64)

    if t1.shape != t2.shape:
        raise ValueError(
            f"Shape mismatch: t1={t1.shape} vs t2={t2.shape}"
        )

    diff = np.abs(t2 - t1)

    if diff.ndim == 3:
        # Multi-channel → L2 norm across channels
        change_magnitude: NDArray[np.floating] = np.sqrt(
            np.sum(diff ** 2, axis=-1)
        )
    elif diff.ndim == 2:
        change_magnitude = diff
    else:
        raise ValueError(
            f"Arrays must be 2-D or 3-D; got ndim={diff.ndim}"
        )

    mean_change = float(np.mean(change_magnitude))

    logger.debug(
        "Temporal change: mean=%.4f  max=%.4f  shape=%s",
        mean_change,
        float(np.max(change_magnitude)),
        change_magnitude.shape,
    )
    return change_magnitude, mean_change


# ============================================================================
# 5. ML-Accelerated Molecular-Orbital Integration (Stub)
# ============================================================================

def load_timewarp_dynamics(**kwargs: Any) -> Any:
    """Load Microsoft TimeWarp dynamics dataset.

    Returns
    -------
    datasets.Dataset | datasets.IterableDataset
        The HuggingFace dataset keyed as ``timewarp``.
    """
    logger.info("Loading TimeWarp molecular dynamics dataset …")
    return get_dataset("timewarp", **kwargs)


def accelerate_molecular_orbital_integrator(
    state_vector: NDArray[np.floating],
    dt: float,
    n_skip: int = 10,
) -> NDArray[np.floating]:
    """Surrogate ML-accelerated molecular-orbital integrator (stub).

    In a production pipeline this would call a trained neural-network
    surrogate (e.g. a TimeWarp / equivariant GNN model) to predict the
    state after *n_skip* integration steps.  The current implementation
    uses **linear extrapolation** as a placeholder:

        state_new = state + dt × n_skip × gradient_estimate

    where *gradient_estimate* is a finite-difference approximation of the
    state's first derivative (here simply ``state`` itself, treating it as
    a velocity-like quantity — replace with a real force/gradient
    evaluation).

    Parameters
    ----------
    state_vector : ndarray, shape (D,)
        Current molecular system state (positions, momenta, etc.).
    dt : float
        Integration time-step [s] (or natural units of the system).
    n_skip : int, default 10
        Number of fine-grained integration steps the surrogate replaces.

    Returns
    -------
    ndarray, shape (D,)
        Predicted state after ``n_skip`` steps of size ``dt``.

    .. todo::
        Replace linear extrapolation with a trained surrogate model
        (e.g. equivariant GNN, TimeWarp, or neural ODE).
    """
    state = np.asarray(state_vector, dtype=np.float64)

    if state.ndim != 1:
        raise ValueError(
            f"state_vector must be 1-D; got shape {state.shape}"
        )

    # TODO: Replace with trained surrogate model inference.
    # ──────────────────────────────────────────────────────────────────────
    # Placeholder: treat the state itself as a crude gradient estimate
    # (equivalent to assuming dx/dt ≈ x, i.e. exponential-growth proxy).
    # A real implementation would evaluate forces / call a model here.
    gradient_estimate = state.copy()
    # ──────────────────────────────────────────────────────────────────────

    state_new: NDArray[np.floating] = state + dt * n_skip * gradient_estimate

    logger.debug(
        "ML integrator stub: dt=%.4e  n_skip=%d  |Δstate|=%.6e",
        dt,
        n_skip,
        float(np.linalg.norm(state_new - state)),
    )
    return state_new
