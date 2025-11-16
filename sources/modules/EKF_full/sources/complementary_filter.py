"""
Complementary Filter for Position Estimation
=============================================
Uses ALL sensors with proper frequency separation:
- HIGH frequency: Accelerometer (short-term dynamics)
- LOW frequency: Biomechanical constraints + periodic motion model
- Orientation: Trust Euler angles from phone
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import signal

if __package__ is None or __package__ == "":
    PACKAGE_ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(PACKAGE_ROOT))
    from sources.modules.EKF_full.sources.config import ensure_output_dir
    from sources.modules.EKF_full.sources.data_loading import (
        build_time_aligned_frame,
        list_sessions,
        load_session,
    )
    from sources.modules.EKF_full.sources.math_utils import euler_to_rotation_matrix
else:
    from .config import ensure_output_dir
    from .data_loading import build_time_aligned_frame, list_sessions, load_session
    from .math_utils import euler_to_rotation_matrix


class ComplementaryPositionFilter:
    """
    Complementary filter that separates frequency domains:
    
    HIGH-pass filtered acceleration → short-term position changes
    LOW-pass filtered position → biomechanical constraints (squats are periodic!)
    """
    
    def __init__(self, dt: float, cutoff_hz: float = 0.5):
        """
        Args:
            dt: Sample period (s)
            cutoff_hz: Crossover frequency (Hz) - separate high/low frequency
        """
        self.dt = dt
        self.cutoff_hz = cutoff_hz
        
        # Design Butterworth filters
        nyquist = 0.5 / dt
        normalized_cutoff = cutoff_hz / nyquist
        
        # High-pass for acceleration (removes DC drift)
        self.b_high, self.a_high = signal.butter(4, normalized_cutoff, btype='high')
        
        # Low-pass for position (smooths biomechanical model)
        self.b_low, self.a_low = signal.butter(4, normalized_cutoff, btype='low')
        
        # State
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        
    def update(
        self,
        accel_world: np.ndarray,
        biomech_position: np.ndarray,
    ) -> np.ndarray:
        """
        Fuse high-frequency acceleration with low-frequency biomechanical model.
        
        Args:
            accel_world: World-frame acceleration [3] (m/s²)
            biomech_position: Position from biomechanical model [3] (m)
            
        Returns:
            Fused position estimate [3] (m)
        """
        # High-frequency: integrate acceleration
        self.vel += accel_world * self.dt
        pos_high_freq = self.pos + self.vel * self.dt
        
        # Complementary fusion
        alpha = self.dt / (self.dt + 1.0 / (2.0 * np.pi * self.cutoff_hz))
        
        # Trust high-freq for short-term, biomech for long-term
        self.pos = alpha * pos_high_freq + (1 - alpha) * biomech_position
        
        return self.pos.copy()


def detect_squat_phases(
    pitch_deg: np.ndarray,
    vel_magnitudes: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Detect squat phases from pitch angle (forward lean) and velocity.
    
    Returns:
        phase: Array of phase labels
            0 = standing
            1 = descending
            2 = bottom
            3 = ascending
    """
    # Smooth pitch to remove noise
    pitch_smooth = signal.savgol_filter(pitch_deg, window_length=21, polyorder=3)
    
    # Derivative of pitch
    dpitch_dt = np.gradient(pitch_smooth, dt)
    
    # Phase detection
    phases = np.zeros(len(pitch_deg), dtype=int)
    
    stationary_threshold = 0.05  # m/s
    pitch_change_threshold = 5.0  # deg/s
    
    for i in range(len(phases)):
        vel = vel_magnitudes[i]
        dpitch = abs(dpitch_dt[i])
        
        if vel < stationary_threshold and dpitch < pitch_change_threshold:
            phases[i] = 0  # Standing
        elif dpitch_dt[i] > pitch_change_threshold:
            phases[i] = 1  # Descending (leaning forward)
        elif dpitch_dt[i] < -pitch_change_threshold:
            phases[i] = 3  # Ascending (straightening up)
        else:
            phases[i] = 2  # Bottom (transition)
    
    return phases


def biomechanical_position_model(
    phases: np.ndarray,
    pitch_deg: np.ndarray,
    roll_deg: np.ndarray,
) -> np.ndarray:
    """
    Estimate position from biomechanical model of squat motion.
    
    During squats:
    - Z (vertical): Correlates with knee angle ≈ function of pitch
    - X (lateral): Small, correlates with roll
    - Y (forward): Correlates with pitch (trunk lean)
    
    Args:
        phases: Motion phases [N]
        pitch_deg: Forward/backward lean [N]
        roll_deg: Left/right tilt [N]
        
    Returns:
        positions: Model-based position [N x 3] in meters
    """
    N = len(phases)
    positions = np.zeros((N, 3))
    
    # Constants from biomechanics literature
    # Squat depth typically 30-60cm, trunk lean 0-40°
    MAX_SQUAT_DEPTH = 0.5  # meters
    TRUNK_LENGTH = 0.6  # meters (chest to hip)
    
    for i in range(N):
        phase = phases[i]
        pitch = np.deg2rad(pitch_deg[i])
        roll = np.deg2rad(roll_deg[i])
        
        if phase == 0:  # Standing
            positions[i] = [0, 0, 0]
        elif phase in [1, 2, 3]:  # Moving
            # Z: Vertical displacement (squat depth)
            # Approximate from pitch angle
            positions[i, 2] = -MAX_SQUAT_DEPTH * (abs(pitch) / np.deg2rad(40))
            
            # Y: Forward displacement from trunk lean
            positions[i, 1] = TRUNK_LENGTH * np.sin(pitch)
            
            # X: Lateral displacement from roll
            positions[i, 0] = TRUNK_LENGTH * np.sin(roll) * 0.1  # Smaller
    
    return positions


def run_complementary_filter(session_id: str) -> pd.DataFrame:
    """
    Run complementary filter using ALL sensors.
    """
    print(f"[COMP] Loading session {session_id}")
    
    session = load_session(session_id)
    df = build_time_aligned_frame(session, tolerance_ms=15)
    
    if df.empty:
        raise ValueError("No data after time alignment")
    
    # Extract all sensor data
    acc = df[['ax', 'ay', 'az']].values
    euler = df[['euler_roll_deg', 'euler_pitch_deg', 'euler_yaw_deg']].values
    
    # Fill NaN in Euler angles with forward/backward fill
    euler_df = pd.DataFrame(euler, columns=['roll', 'pitch', 'yaw'])
    euler_df = euler_df.fillna(method='ffill').fillna(method='bfill')
    euler = euler_df.values
    
    # Check if still have NaN
    if np.any(np.isnan(euler)):
        print(f"[WARN] Still have {np.sum(np.isnan(euler))} NaN values in Euler angles after fill")
        euler = np.nan_to_num(euler, nan=0.0)
    
    # Sample rate
    timestamps_s = (df['timestamp'].values - df['timestamp'].values[0]) / 1000
    dt = np.median(np.diff(timestamps_s))
    print(f"[COMP] Sample rate: {1/dt:.1f} Hz")
    
    # Initialize filter
    comp_filter = ComplementaryPositionFilter(dt=dt, cutoff_hz=0.5)
    
    # Process
    N = len(df)
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    phases = np.zeros(N, dtype=int)
    
    # First pass: detect phases
    vel_magnitudes = np.zeros(N)
    phases = detect_squat_phases(euler[:, 1], vel_magnitudes, dt)
    
    # Second pass: apply complementary filter
    for i in range(N):
        # Rotate acceleration to world frame
        R = euler_to_rotation_matrix(
            np.deg2rad(euler[i, 0]),
            np.deg2rad(euler[i, 1]),
            np.deg2rad(euler[i, 2])
        )
        
        # Check for NaN in rotation matrix
        if np.any(np.isnan(R)):
            print(f"[WARN] NaN in rotation matrix at sample {i}, euler={euler[i]}")
            R = np.eye(3)
        
        accel_world = R @ acc[i]
        
        # Check for NaN in acceleration
        if np.any(np.isnan(accel_world)):
            print(f"[WARN] NaN in accel_world at sample {i}, acc={acc[i]}, R valid={not np.any(np.isnan(R))}")
            accel_world = np.zeros(3)
        
        accel_world[2] -= 9.80665  # Remove gravity
        
        # Biomechanical position model
        biomech_pos = biomechanical_position_model(
            phases[i:i+1],
            euler[i:i+1, 1],
            euler[i:i+1, 0]
        )[0]
        
        # Fuse
        positions[i] = comp_filter.update(accel_world, biomech_pos)
        velocities[i] = comp_filter.vel
        
        vel_magnitudes[i] = np.linalg.norm(comp_filter.vel)
    
    # Re-detect phases with actual velocities
    phases = detect_squat_phases(euler[:, 1], vel_magnitudes, dt)
    
    # Build results
    results = pd.DataFrame({
        'timestamp_ms': df['timestamp'].values,
        'time_s': timestamps_s,
        'pos_x_cm': positions[:, 0] * 100,
        'pos_y_cm': positions[:, 1] * 100,
        'pos_z_cm': positions[:, 2] * 100,
        'vel_x_cm_s': velocities[:, 0] * 100,
        'vel_y_cm_s': velocities[:, 1] * 100,
        'vel_z_cm_s': velocities[:, 2] * 100,
        'roll_deg': euler[:, 0],
        'pitch_deg': euler[:, 1],
        'yaw_deg': euler[:, 2],
        'phase': phases,
    })
    
    # Save
    output_dir = ensure_output_dir()
    output_path = output_dir / f"{session_id}_complementary_results.csv"
    results.to_csv(output_path, index=False)
    print(f"[COMP] Saved: {output_path}")
    
    # Stats
    print(f"\n[COMP] Position ranges:")
    print(f"  X: {positions[:, 0].min()*100:.1f} to {positions[:, 0].max()*100:.1f} cm")
    print(f"  Y: {positions[:, 1].min()*100:.1f} to {positions[:, 1].max()*100:.1f} cm")
    print(f"  Z: {positions[:, 2].min()*100:.1f} to {positions[:, 2].max()*100:.1f} cm")
    print(f"\n[COMP] Velocity ranges:")
    print(f"  Magnitude: {vel_magnitudes.min():.3f} to {vel_magnitudes.max():.3f} m/s")
    print(f"\n[COMP] Phases detected:")
    for phase_id, phase_name in [(0, 'Standing'), (1, 'Descending'), (2, 'Bottom'), (3, 'Ascending')]:
        count = np.sum(phases == phase_id)
        pct = 100 * count / len(phases)
        print(f"  {phase_name}: {count} samples ({pct:.1f}%)")
    
    return results


if __name__ == "__main__":
    sessions = list_sessions()
    if not sessions:
        print("No sessions found")
        sys.exit(1)
    
    # Test on one session
    session_id = sessions[-1]
    print(f"Testing complementary filter on: {session_id}\n")
    
    results = run_complementary_filter(session_id)
