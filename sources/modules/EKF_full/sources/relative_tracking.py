"""
Relative Position Tracking for Repetitive Exercises
====================================================
Instead of absolute position (which drifts), measure displacement per repetition.
Resets position at each standing phase to prevent drift accumulation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def detect_stationary_phases(
    velocities: np.ndarray,
    threshold_m_s: float = 0.05,
    min_duration_samples: int = 20,
) -> List[Tuple[int, int]]:
    """
    Detect stationary periods (standing phases) from velocity data.
    
    Args:
        velocities: Velocity magnitude array [N samples]
        threshold_m_s: Max velocity to consider stationary (m/s)
        min_duration_samples: Minimum samples to qualify as stationary
        
    Returns:
        List of (start_idx, end_idx) tuples for each stationary phase
    """
    vel_magnitude = np.sqrt(np.sum(velocities**2, axis=1))
    is_stationary = vel_magnitude < threshold_m_s
    
    phases = []
    start_idx = None
    
    for i, stat in enumerate(is_stationary):
        if stat and start_idx is None:
            start_idx = i
        elif not stat and start_idx is not None:
            duration = i - start_idx
            if duration >= min_duration_samples:
                phases.append((start_idx, i - 1))
            start_idx = None
    
    # Handle last phase
    if start_idx is not None and len(is_stationary) - start_idx >= min_duration_samples:
        phases.append((start_idx, len(is_stationary) - 1))
    
    return phases


def extract_repetitions(
    positions: np.ndarray,
    velocities: np.ndarray,
    timestamps: np.ndarray,
    stationary_threshold: float = 0.05,
    min_standing_duration: int = 20,
) -> pd.DataFrame:
    """
    Extract per-repetition metrics from position/velocity data.
    
    Args:
        positions: Position array [N x 3] in meters (X, Y, Z)
        velocities: Velocity array [N x 3] in m/s
        timestamps: Time array [N] in seconds
        stationary_threshold: Velocity threshold for standing detection
        min_standing_duration: Min samples for valid standing phase
        
    Returns:
        DataFrame with columns:
        - rep_number: Repetition index
        - start_time: Start timestamp (s)
        - end_time: End timestamp (s)
        - duration: Total duration (s)
        - descent_m: Vertical descent (m)
        - ascent_m: Vertical ascent (m)
        - lateral_x_m: Lateral displacement X (m)
        - lateral_y_m: Lateral displacement Y (m)
        - max_velocity_m_s: Peak velocity during rep
        - avg_velocity_m_s: Average velocity during rep
    """
    # Detect standing phases
    standing_phases = detect_stationary_phases(
        velocities, stationary_threshold, min_standing_duration
    )
    
    if len(standing_phases) < 2:
        return pd.DataFrame()  # Not enough phases to extract reps
    
    reps = []
    
    for i in range(len(standing_phases) - 1):
        # Repetition is from end of one standing phase to start of next
        rep_start = standing_phases[i][1]  # End of standing
        rep_end = standing_phases[i + 1][0]  # Start of next standing
        
        if rep_end <= rep_start:
            continue
        
        # Extract positions during this rep
        rep_positions = positions[rep_start:rep_end + 1]
        rep_velocities = velocities[rep_start:rep_end + 1]
        rep_times = timestamps[rep_start:rep_end + 1]
        
        # Calculate vertical displacement (Z-axis)
        z_positions = rep_positions[:, 2]
        min_z = np.min(z_positions)
        max_z = np.max(z_positions)
        
        # Find descent (going down) and ascent (going up)
        # Descent: from max to min
        # Ascent: from min to max
        descent = max_z - min_z
        ascent = max_z - min_z  # Symmetric for squats
        
        # Lateral displacement (X, Y)
        lateral_x = np.max(rep_positions[:, 0]) - np.min(rep_positions[:, 0])
        lateral_y = np.max(rep_positions[:, 1]) - np.min(rep_positions[:, 1])
        
        # Velocity metrics
        vel_magnitudes = np.sqrt(np.sum(rep_velocities**2, axis=1))
        max_vel = np.max(vel_magnitudes)
        avg_vel = np.mean(vel_magnitudes)
        
        reps.append({
            'rep_number': i + 1,
            'start_time': rep_times[0],
            'end_time': rep_times[-1],
            'duration': rep_times[-1] - rep_times[0],
            'descent_m': descent,
            'ascent_m': ascent,
            'lateral_x_m': lateral_x,
            'lateral_y_m': lateral_y,
            'max_velocity_m_s': max_vel,
            'avg_velocity_m_s': avg_vel,
        })
    
    return pd.DataFrame(reps)


def apply_relative_tracking(
    positions: np.ndarray,
    velocities: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Apply position reset at each standing phase to prevent drift accumulation.
    
    Returns:
        Corrected positions [N x 3] with resets applied
    """
    standing_phases = detect_stationary_phases(velocities)
    
    corrected_positions = positions.copy()
    
    for phase_start, phase_end in standing_phases:
        # Reset position to zero at each standing phase
        offset = positions[phase_start]
        corrected_positions[phase_start:] -= offset
    
    return corrected_positions


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Load EKF results
    results_csv = Path("data/outputs/2025-10-28-10-30-39_ekf_results.csv")
    if not results_csv.exists():
        print(f"File not found: {results_csv}")
        sys.exit(1)
    
    df = pd.read_csv(results_csv)
    
    positions = df[['pos_x_cm', 'pos_y_cm', 'pos_z_cm']].values / 100  # cm to m
    velocities = df[['vel_x_cm_s', 'vel_y_cm_s', 'vel_z_cm_s']].values / 100  # cm/s to m/s
    timestamps = df['time_s'].values
    
    # Extract repetitions
    print(f"\nAnalyzing {len(positions)} samples...")
    print(f"Velocity range: {np.min(velocities):.3f} to {np.max(velocities):.3f} m/s")
    
    # Try different thresholds
    for threshold in [0.5, 0.3, 0.2, 0.1, 0.05]:
        reps_df = extract_repetitions(
            positions, velocities, timestamps,
            stationary_threshold=threshold,
            min_standing_duration=10
        )
        if len(reps_df) > 0:
            print(f"\nUsing threshold={threshold} m/s -> Found {len(reps_df)} repetitions")
            break
    else:
        print("\nNo repetitions detected with any threshold!")
        print("\nVelocity magnitude statistics:")
        vel_mag = np.sqrt(np.sum(velocities**2, axis=1))
        print(f"  Min: {np.min(vel_mag):.4f} m/s")
        print(f"  Max: {np.max(vel_mag):.4f} m/s")
        print(f"  Mean: {np.mean(vel_mag):.4f} m/s")
        print(f"  Median: {np.median(vel_mag):.4f} m/s")
        print(f"  P95: {np.percentile(vel_mag, 95):.4f} m/s")
        sys.exit(1)
    
    print("\n=== Repetition Analysis ===")
    print(reps_df.to_string(index=False))
    
    # Save results
    output_path = results_csv.parent / f"{results_csv.stem}_repetitions.csv"
    reps_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
