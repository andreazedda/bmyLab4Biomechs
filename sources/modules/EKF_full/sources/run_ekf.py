from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    # Allow running as a standalone script: python run_ekf.py
    PACKAGE_ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(PACKAGE_ROOT))
    from sources.modules.EKF_full.sources.config import (  # type: ignore
        DEFAULT_PIPELINE_CONFIG,
        ensure_output_dir,
    )
    from sources.modules.EKF_full.sources.data_loading import (  # type: ignore
        build_time_aligned_frame,
        estimate_world_magnetic_field,
        list_sessions,
        load_session,
    )
    from sources.modules.EKF_full.sources.ekf_model import (  # type: ignore
        IDX_ANG,
        InertialPositionEKF,
    )
    from sources.modules.EKF_full.sources.math_utils import euler_to_rotation_matrix, wrap_angles  # type: ignore
    from sources.modules.EKF_full.sources.plotting import generate_session_plots  # type: ignore
else:
    from .config import DEFAULT_PIPELINE_CONFIG, ensure_output_dir
    from .data_loading import (
        build_time_aligned_frame,
        estimate_world_magnetic_field,
        list_sessions,
        load_session,
    )
    from .ekf_model import InertialPositionEKF, IDX_ANG
    from .math_utils import euler_to_rotation_matrix, wrap_angles
    from .plotting import generate_session_plots


@dataclass
class SessionArtifacts:
    csv: Path
    plots: List[Path]


def _calibrated_gyro(row: pd.Series) -> np.ndarray:
    gx = row.get("gyro_gx_raw", np.nan)
    gy = row.get("gyro_gy_raw", np.nan)
    gz = row.get("gyro_gz_raw", np.nan)
    bx = row.get("gyro_gx_bias", 0.0)
    by = row.get("gyro_gy_bias", 0.0)
    bz = row.get("gyro_gz_bias", 0.0)
    vec = np.array([gx - bx, gy - by, gz - bz], dtype=float)
    return np.nan_to_num(vec)


def _acc_vector(row: pd.Series) -> np.ndarray:
    vec = np.array([row["ax"], row["ay"], row["az"]], dtype=float)
    return np.nan_to_num(vec)


def _mag_vector(row: pd.Series) -> np.ndarray:
    cols = ("magn_mx", "magn_my", "magn_mz")
    if not set(cols).issubset(row.index):
        return np.array([np.nan, np.nan, np.nan])
    vec = np.array([row[c] for c in cols], dtype=float)
    return vec if not np.any(np.isnan(vec)) else np.array([np.nan, np.nan, np.nan])


def _euler_measurement(row: pd.Series) -> np.ndarray:
    cols = ("euler_roll_deg", "euler_pitch_deg", "euler_yaw_deg")
    if not set(cols).issubset(row.index):
        return np.array([np.nan, np.nan, np.nan])
    return np.deg2rad([row[c] for c in cols])


def _should_apply_zupt(row: pd.Series, config=DEFAULT_PIPELINE_CONFIG) -> bool:
    detection = config.detection
    acc_norm = np.linalg.norm(_acc_vector(row))
    gyro_norm = np.linalg.norm(_calibrated_gyro(row))
    mag_norm = np.linalg.norm(_mag_vector(row))

    acc_stable = abs(acc_norm - detection.gravity) < detection.zero_vel_acc_window
    gyro_stable = gyro_norm < detection.zero_vel_gyro_window
    mag_stable = mag_norm < detection.zero_vel_mag_window if not np.isnan(mag_norm) else True
    return acc_stable and gyro_stable and mag_stable


def _mag_measurement_fn(mag_world: np.ndarray):
    def fn(state: np.ndarray) -> np.ndarray:
        rotation = euler_to_rotation_matrix(*state[IDX_ANG])
        return rotation.T @ mag_world

    return fn


def _euler_measurement_fn(state: np.ndarray) -> np.ndarray:
    return state[IDX_ANG]


def _angle_residual(z: np.ndarray, h: np.ndarray) -> np.ndarray:
    return wrap_angles(z - h)


def run_session(session_id: str, config=DEFAULT_PIPELINE_CONFIG, make_plots: bool = False) -> SessionArtifacts:
    session_data = load_session(session_id)
    aligned = build_time_aligned_frame(session_data, tolerance_ms=config.detection.max_interp_gap_ms)
    output_dir = ensure_output_dir()

    ekf = InertialPositionEKF(config=config)

    initial_euler = aligned[["euler_roll_deg", "euler_pitch_deg", "euler_yaw_deg"]].iloc[0]
    initial_angles = np.deg2rad(initial_euler.fillna(0.0).values)
    initial_gyro_bias = aligned[["gyro_gx_bias", "gyro_gy_bias", "gyro_gz_bias"]].iloc[0].fillna(0.0).values
    ekf.initialize(initial_angles_rad=initial_angles, gyro_bias=initial_gyro_bias)

    mag_world = estimate_world_magnetic_field(aligned)
    mag_fn = _mag_measurement_fn(mag_world) if np.linalg.norm(mag_world) > 0 else None

    noise = config.noise
    R_euler = np.eye(3) * np.deg2rad(noise.euler_meas_deg) ** 2
    R_mag = np.eye(3) * (noise.mag_meas_uT ** 2)
    R_zupt = np.eye(3) * noise.zero_vel

    results: List[Dict[str, float]] = []
    prev_time = aligned["time_s"].iloc[0]
    
    # Drift correction: track reference position
    zupt_count = 0
    stationary_positions = []
    sample_count = 0
    last_reset_time = prev_time
    
    # Reference position (initialize at origin)
    reference_position = np.zeros(3)
    
    print(f"[EKF] Processing {len(aligned)} samples at ~{config.sensor_sampling_hz}Hz")
    print(f"[EKF] Euler measurement noise: {noise.euler_meas_deg}° (HIGH TRUST)")
    print(f"[EKF] Position bounding: ±{config.detection.max_position_m}m")

    for _, row in aligned.iterrows():
        current_time = row["time_s"]
        dt = current_time - prev_time
        acc = _acc_vector(row)
        gyro = _calibrated_gyro(row)
        
        # Get Euler angles from phone's sensor fusion (MORE RELIABLE than gyro integration!)
        euler_meas = _euler_measurement(row)
        has_euler = not np.any(np.isnan(euler_meas))
        
        # Use phone's angles to correct drift, but don't override completely
        if has_euler and sample_count % 5 == 0:  # Every 5 samples
            # Blend measured angles with current estimate
            current_angles = ekf.state[IDX_ANG]
            angle_error = wrap_angles(euler_meas - current_angles)
            # Apply 50% correction towards measured angles
            ekf.state[IDX_ANG] = wrap_angles(current_angles + angle_error * 0.5)
        
        ekf.predict(acc, gyro, dt)

        # Additional Euler update for covariance correction (optional refinement)
        if has_euler:
            ekf.update(euler_meas, _euler_measurement_fn, R_euler * 2.0, residual_fn=_angle_residual)

        # Magnetometer update (optional, less reliable indoors)
        if mag_fn is not None and np.linalg.norm(mag_world) > 20.0:  # Only if mag field is strong enough
            mag_meas = _mag_vector(row)
            if not np.any(np.isnan(mag_meas)) and np.linalg.norm(mag_meas) > 20.0:
                try:
                    ekf.update(mag_meas, mag_fn, R_mag)
                except np.linalg.LinAlgError:
                    pass  # Skip this magnetometer update if singular

        # ZUPT with position anchoring
        if _should_apply_zupt(row, config=config):
            ekf.zero_velocity_update(R_zupt)
            zupt_count += 1
            state = ekf.get_state()
            stationary_positions.append(state[:3].copy())
            
            # Update reference position from stationary periods
            if len(stationary_positions) >= 2:
                reference_position = np.mean(stationary_positions[-2:], axis=0)
        
        # Periodic position drift correction
        sample_count += 1
        time_since_reset = current_time - last_reset_time
        
        if time_since_reset >= config.detection.position_reset_interval_s:
            state = ekf.get_state()
            current_pos = state[:3]
            
            # If position has drifted too far from reference, apply correction
            pos_error = current_pos - reference_position
            error_magnitude = np.linalg.norm(pos_error)
            
            if error_magnitude > config.detection.max_position_m:
                # Hard clamp to maximum distance
                correction = pos_error * (1.0 - config.detection.max_position_m / error_magnitude)
                ekf.state[:3] -= correction * 0.7  # 70% correction
                print(f"[EKF] Position bounded: drift={error_magnitude:.2f}m, corrected by {np.linalg.norm(correction):.2f}m")
            elif error_magnitude > config.detection.max_position_m * 0.5:
                # Soft correction if more than half the limit
                correction = pos_error * 0.2  # 20% correction
                ekf.state[:3] -= correction
            
            last_reset_time = current_time

        state = ekf.get_state()
        results.append(
            {
                "timestamp_ms": row["timestamp"],
                "time_s": current_time,
                "pos_x_cm": state[0] * 100.0,
                "pos_y_cm": state[1] * 100.0,
                "pos_z_cm": state[2] * 100.0,
                "vel_x_cm_s": state[3] * 100.0,
                "vel_y_cm_s": state[4] * 100.0,
                "vel_z_cm_s": state[5] * 100.0,
                "roll_deg": np.rad2deg(state[6]),
                "pitch_deg": np.rad2deg(state[7]),
                "yaw_deg": np.rad2deg(state[8]),
            }
        )

        prev_time = current_time

    results_df = pd.DataFrame(results)
    output_path = output_dir / f"{session_id}_ekf_results.csv"
    results_df.to_csv(output_path, index=False)

    plots: List[Path] = []
    if make_plots:
        plots = generate_session_plots(results_df, session_id, output_dir)

    return SessionArtifacts(csv=output_path, plots=plots)


def parse_args():
    parser = argparse.ArgumentParser(description="EKF position estimator for Android recordings.")
    parser.add_argument(
        "--session",
        type=str,
        help="Session identifier, e.g. 2025-10-28-10-19-35. Default: most recent.",
    )
    parser.add_argument("--list-sessions", action="store_true", help="List available sessions and exit.")
    parser.add_argument("--all", action="store_true", help="Process every available session.")
    parser.add_argument("--plot", action="store_true", help="Generate position/velocity/orientation plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    sessions = list_sessions()
    if not sessions:
        raise SystemExit("No sessions found inside the txt input directory.")

    if args.list_sessions:
        print("Available sessions:")
        for sess in sessions:
            print(f"  - {sess}")
        return

    target_sessions: List[str]
    if args.all:
        target_sessions = sessions
    else:
        target_sessions = [args.session or sessions[-1]]

    for sess in target_sessions:
        outputs = run_session(sess, make_plots=args.plot)
        print(f"[EKF] Session {sess} processed -> {outputs.csv}")
        if args.plot:
            for plot_path in outputs.plots:
                print(f"   Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
