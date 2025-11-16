from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _plot_time_series(
    df: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    labels: List[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    plt.figure(figsize=(10, 4))
    have_any = False
    for col, label in zip(value_cols, labels):
        if col in df:
            y = df[col].to_numpy()
            # Unwrap if angle series (avoids sawtooth at ±180°)
            if any(k in col for k in ("roll", "pitch", "yaw")):
                y = np.rad2deg(np.unwrap(np.deg2rad(y)))
            plt.plot(df[time_col], y, label=label, linewidth=1.2)
            have_any = True
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if have_any:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def _plot_3d_trajectory(df: pd.DataFrame, output_path: Path) -> Path:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Drop NaN rows before plotting to avoid broken trajectories
    pos = df[["pos_x_cm", "pos_y_cm", "pos_z_cm"]].dropna()
    if not pos.empty:
        ax.plot(pos["pos_x_cm"], pos["pos_y_cm"], pos["pos_z_cm"], linewidth=1.0)
    ax.set_title("3D Trajectory (cm)")
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_zlabel("Z [cm]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def generate_session_plots(results_df: pd.DataFrame, session_id: str, output_dir: Path) -> List[Path]:
    output_paths: List[Path] = []

    pos_plot = output_dir / f"{session_id}_positions.png"
    output_paths.append(
        _plot_time_series(
            results_df,
            "time_s",
            ["pos_x_cm", "pos_y_cm", "pos_z_cm"],
            ["X", "Y", "Z"],
            "Position vs Time",
            "Position [cm]",
            pos_plot,
        )
    )

    vel_plot = output_dir / f"{session_id}_velocities.png"
    output_paths.append(
        _plot_time_series(
            results_df,
            "time_s",
            ["vel_x_cm_s", "vel_y_cm_s", "vel_z_cm_s"],
            ["X", "Y", "Z"],
            "Velocity vs Time",
            "Velocity [cm/s]",
            vel_plot,
        )
    )

    euler_plot = output_dir / f"{session_id}_orientation.png"
    output_paths.append(
        _plot_time_series(
            results_df,
            "time_s",
            ["roll_deg", "pitch_deg", "yaw_deg"],
            ["Roll", "Pitch", "Yaw"],
            "Orientation vs Time",
            "Angle [deg]",
            euler_plot,
        )
    )

    traj_plot = output_dir / f"{session_id}_trajectory3d.png"
    output_paths.append(_plot_3d_trajectory(results_df, traj_plot))

    return output_paths

