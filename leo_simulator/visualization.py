"""Plotting and visualization routines for LEO nanosatellite trajectories and perturbations."""

import os

# Ensure headless execution
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from leo_simulator.constants import R_EARTH
from leo_simulator.propagator import PropagationResult


def plot_orbit_3d(
    result: PropagationResult,
    save_path: str | None = None,
    title: str = "3D LEO Orbit Trajectory",
    r_earth_km: float = R_EARTH / 1000.0,
) -> plt.Figure:
    """Plot 3D Earth and orbit trajectory.

    Args:
        result: PropagationResult object.
        save_path: Optional file path to save plot image (PNG/PDF/SVG).
        title: Plot title.
        r_earth_km: Radius of Earth sphere in km.

    Returns:
        matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Wireframe Earth
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x_sphere = r_earth_km * np.outer(np.cos(u), np.sin(v))
    y_sphere = r_earth_km * np.outer(np.sin(u), np.sin(v))
    z_sphere = r_earth_km * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="lightblue", alpha=0.35, linewidth=0.6)

    # Orbit trajectory in km
    r_km = result.r / 1000.0
    ax.plot(r_km[:, 0], r_km[:, 1], r_km[:, 2], color="red", linewidth=1.5, label="Satellite Trajectory")

    # Start and End markers
    ax.scatter(r_km[0, 0], r_km[0, 1], r_km[0, 2], color="green", s=40, label="Start (t=0)")
    ax.scatter(r_km[-1, 0], r_km[-1, 1], r_km[-1, 2], color="darkred", s=40, label="End")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X (km, ECI)")
    ax.set_ylabel("Y (km, ECI)")
    ax.set_zlabel("Z (km, ECI)")
    ax.legend(loc="upper right")

    # Set equal aspect ratio
    max_range = np.array([
        r_km[:, 0].max() - r_km[:, 0].min(),
        r_km[:, 1].max() - r_km[:, 1].min(),
        r_km[:, 2].max() - r_km[:, 2].min(),
    ]).max() / 2.0

    mid_x = (r_km[:, 0].max() + r_km[:, 0].min()) * 0.5
    mid_y = (r_km[:, 1].max() + r_km[:, 1].min()) * 0.5
    mid_z = (r_km[:, 2].max() + r_km[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_altitude_decay(
    results: PropagationResult | dict[str, PropagationResult],
    save_path: str | None = None,
    title: str = "Orbital Altitude Evolution (Atmospheric Drag)",
) -> plt.Figure:
    """Plot altitude vs time.

    Args:
        results: Single PropagationResult or dict mapping labels to PropagationResult.
        save_path: Optional file path to save plot.
        title: Plot title.

    Returns:
        matplotlib.figure.Figure
    """
    if isinstance(results, PropagationResult):
        results_dict = {"Simulation": results}
    else:
        results_dict = results

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, res in results_dict.items():
        t_hours = res.t / 3600.0
        alt_km = res.altitudes / 1000.0
        ax.plot(t_hours, alt_km, label=label, linewidth=1.8)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Elapsed Time (hours)")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_raan_regression(
    result: PropagationResult,
    analytical_rate: float | None = None,
    save_path: str | None = None,
    title: str = "Ascending Node Precession (J2 Oblateness)",
) -> plt.Figure:
    """Plot RAAN (Omega) vs time, comparing numerical result with analytical secular rate.

    Args:
        result: PropagationResult object.
        analytical_rate: Optional analytical rate in rad / s.
        save_path: Optional file path to save plot.
        title: Plot title.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    t_hours = result.t / 3600.0
    # Unwrap RAAN for smooth rate visualization
    raan_deg = np.degrees(np.unwrap(result.raans))

    ax.plot(t_hours, raan_deg, color="navy", linewidth=2.0, label="Numerical Integration (DOP853)")

    if analytical_rate is not None:
        # Initial RAAN + analytical_rate * t
        t_sec = result.t
        raan_analytical = np.degrees(result.raans[0] + analytical_rate * t_sec)
        ax.plot(t_hours, raan_analytical, "r--", linewidth=1.6, label="Analytical J2 First-Order Rate")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Elapsed Time (hours)")
    ax.set_ylabel("Right Ascension of Ascending Node (deg)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_orbital_elements_history(
    result: PropagationResult,
    save_path: str | None = None,
    title: str = "Orbital Elements Time History",
) -> plt.Figure:
    """Plot multi-panel Keplerian elements history (a, e, i, RAAN).

    Args:
        result: PropagationResult object.
        save_path: Optional file path to save plot.
        title: Plot title.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    t_hours = result.t / 3600.0

    # Semi-major axis
    ax1.plot(t_hours, result.semi_major_axes / 1000.0, color="crimson", linewidth=1.5)
    ax1.set_ylabel("Semi-major Axis a (km)")
    ax1.set_xlabel("Time (hours)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Eccentricity
    ax2.plot(t_hours, result.eccentricities, color="teal", linewidth=1.5)
    ax2.set_ylabel("Eccentricity e")
    ax2.set_xlabel("Time (hours)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Inclination
    ax3.plot(t_hours, np.degrees(result.inclinations), color="purple", linewidth=1.5)
    ax3.set_ylabel("Inclination i (deg)")
    ax3.set_xlabel("Time (hours)")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # RAAN
    ax4.plot(t_hours, np.degrees(np.unwrap(result.raans)), color="navy", linewidth=1.5)
    ax4.set_ylabel("RAAN $\\Omega$ (deg)")
    ax4.set_xlabel("Time (hours)")
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
