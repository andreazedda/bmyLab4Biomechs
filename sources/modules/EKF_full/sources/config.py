from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = MODULE_ROOT / "data"
INPUT_TXT_DIR = DATA_DIR / "inputs" / "txts"
OUTPUT_DIR = DATA_DIR / "outputs"


@dataclass(frozen=True)
class NoiseParams:
    """Process and measurement noise parameters tuned for chest-mounted phone during squats"""
    pos: float = 5e-3  # Higher - expect drift without absolute position reference
    vel: float = 5e-2  # Higher - integrate acceleration with caution
    angles: float = 1e-3  # Process noise for angles - let measurements dominate
    accel_bias: float = 5e-4  # Moderate - accelerometer bias drifts slowly
    gyro_bias: float = 1e-6  # Moderate - gyro bias drifts slowly
    euler_meas_deg: float = 0.5  # TRUST phone's sensor fusion! Very accurate
    mag_meas_uT: float = 30.0  # Magnetometer less reliable indoors
    zero_vel: float = 1e-3  # Aggressive ZUPT when stationary


@dataclass(frozen=True)
class DetectionParams:
    """Detection thresholds tuned for chest-mounted phone during squats"""
    gravity: float = 9.80665
    zero_vel_acc_window: float = 0.3  # Detect standing/rest phases
    zero_vel_gyro_window: float = 0.08  # Low rotation during rest
    zero_vel_mag_window: float = 0.5  # Less critical
    max_interp_gap_ms: int = 15
    
    # Position bounding for biomechanics
    max_position_m: float = 2.0  # Chest can't move more than 2m from start
    position_reset_interval_s: float = 2.0  # Reset drift every 2 seconds


@dataclass(frozen=True)
class PipelineConfig:
    sensor_sampling_hz: float = 100.0
    noise: NoiseParams = NoiseParams()
    detection: DetectionParams = DetectionParams()


DEFAULT_PIPELINE_CONFIG = PipelineConfig()


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
