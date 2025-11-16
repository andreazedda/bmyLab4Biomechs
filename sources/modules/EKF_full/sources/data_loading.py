from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import INPUT_TXT_DIR, DEFAULT_PIPELINE_CONFIG
from .math_utils import euler_to_rotation_matrix


SENSOR_METADATA = {
    "acc": {
        "prefix": "FILE_SENSOR_ACC",
        "columns": ["timestamp", "ax", "ay", "az"],
    },
    "gyro": {
        "prefix": "FILE_GYRO_UNCALIBRATED",
        "columns": ["timestamp", "gx_raw", "gy_raw", "gz_raw", "gx_bias", "gy_bias", "gz_bias"],
    },
    "magn": {
        "prefix": "FILE_MAGN",
        "columns": ["timestamp", "mx", "my", "mz"],
    },
    "euler": {
        "prefix": "FILE_EULER",
        "columns": ["timestamp", "roll_deg", "pitch_deg", "yaw_deg"],
    },
    "rel_rot": {
        "prefix": "FILE_REL_ROT",
        "columns": ["timestamp", "qx", "qy", "qz", "qw"],  # Quaternion
    },
}


@dataclass
class SessionData:
    session_id: str
    sensors: Dict[str, pd.DataFrame]

    def get(self, key: str) -> pd.DataFrame:
        return self.sensors.get(key, pd.DataFrame())


def list_sessions(input_dir: Path | None = None) -> List[str]:
    input_dir = input_dir or INPUT_TXT_DIR
    acc_files = sorted(input_dir.glob("FILE_SENSOR_ACC*.txt"))
    sessions = []
    for file_path in acc_files:
        session = file_path.stem.replace("FILE_SENSOR_ACC", "")
        if session:
            sessions.append(session)
    return sorted(set(sessions))


def _load_sensor_file(sensor_key: str, session_id: str, input_dir: Path) -> pd.DataFrame:
    metadata = SENSOR_METADATA[sensor_key]
    path = input_dir / f"{metadata['prefix']}{session_id}.txt"
    if not path.exists():
        return pd.DataFrame(columns=metadata["columns"])
    df = pd.read_csv(path, header=None, names=metadata["columns"])
    return df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)


def load_session(session_id: str, input_dir: Path | None = None) -> SessionData:
    input_dir = input_dir or INPUT_TXT_DIR
    sensors = {key: _load_sensor_file(key, session_id, input_dir) for key in SENSOR_METADATA.keys()}
    return SessionData(session_id=session_id, sensors=sensors)


def _merge_sensor(base: pd.DataFrame, sensor_df: pd.DataFrame, prefix: str, tolerance_ms: int) -> pd.DataFrame:
    if sensor_df.empty:
        return base
    renamed = sensor_df.rename(
        columns={
            col: f"{prefix}_{col}"
            for col in sensor_df.columns
            if col != "timestamp"
        }
    )
    merged = pd.merge_asof(
        base,
        renamed,
        on="timestamp",
        direction="nearest",
        tolerance=tolerance_ms,
    )
    return merged


def build_time_aligned_frame(
    session_data: SessionData,
    tolerance_ms: int | None = None,
) -> pd.DataFrame:
    tolerance_ms = tolerance_ms or DEFAULT_PIPELINE_CONFIG.detection.max_interp_gap_ms

    acc_df = session_data.get("acc")
    if acc_df.empty:
        raise ValueError("Accelerometer data is required to build the time base.")

    base = acc_df.copy()
    base["time_s"] = (base["timestamp"] - base["timestamp"].iloc[0]) / 1000.0

    for sensor_key in ("gyro", "magn", "euler"):
        sensor_df = session_data.get(sensor_key)
        base = _merge_sensor(base, sensor_df, sensor_key, tolerance_ms)
        expected_cols = [
            f"{sensor_key}_{col}"
            for col in SENSOR_METADATA[sensor_key]["columns"]
            if col != "timestamp"
        ]
        for col in expected_cols:
            if col not in base.columns:
                base[col] = np.nan

    # Reorder columns for clarity
    for col in base.columns:
        if col != "timestamp":
            base[col] = pd.to_numeric(base[col], errors="coerce")

    ordered_cols = ["timestamp", "time_s", "ax", "ay", "az"]
    remaining_cols = [c for c in base.columns if c not in ordered_cols]
    base = base[ordered_cols + remaining_cols]

    # Interpolate to fill small gaps (exclude Euler angles from linear interpolation)
    angle_cols = {"euler_roll_deg", "euler_pitch_deg", "euler_yaw_deg"}
    numeric_cols = [c for c in base.columns if c not in ("timestamp") and c not in angle_cols]
    base[numeric_cols] = base[numeric_cols].interpolate(limit_direction="both")
    return base


def estimate_world_magnetic_field(aligned_frame: pd.DataFrame) -> np.ndarray:
    if not {"magn_mx", "magn_my", "magn_mz", "euler_roll_deg"}.issubset(aligned_frame.columns):
        return np.array([0.0, 0.0, 0.0])

    valid = aligned_frame[
        [
            "magn_mx",
            "magn_my",
            "magn_mz",
            "euler_roll_deg",
            "euler_pitch_deg",
            "euler_yaw_deg",
        ]
    ].dropna()
    if valid.empty:
        return np.array([0.0, 0.0, 0.0])

    world_vectors = []
    for _, row in valid.iterrows():
        roll = np.deg2rad(row["euler_roll_deg"])
        pitch = np.deg2rad(row["euler_pitch_deg"])
        yaw = np.deg2rad(row["euler_yaw_deg"])
        rotation = euler_to_rotation_matrix(roll, pitch, yaw)
        body_vec = np.array([row["magn_mx"], row["magn_my"], row["magn_mz"]])
        world_vec = rotation @ body_vec
        world_vectors.append(world_vec)

    return np.mean(world_vectors, axis=0)
