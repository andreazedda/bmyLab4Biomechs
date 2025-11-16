from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def deg2rad(values: Iterable[float]) -> np.ndarray:
    return np.deg2rad(values)


def rad2deg(values: Iterable[float]) -> np.ndarray:
    return np.rad2deg(values)


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angles(vec: np.ndarray) -> np.ndarray:
    return np.vectorize(wrap_angle)(vec)


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # ZYX rotation (yaw -> pitch -> roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rotation_matrix_to_euler(matrix: np.ndarray) -> np.ndarray:
    pitch = -math.asin(np.clip(matrix[2, 0], -1.0, 1.0))
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return np.array([roll, pitch, yaw])


def numerical_jacobian(func, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    fx = np.asarray(func(x))
    jac = np.zeros((fx.size, x.size))
    for i in range(x.size):
        perturb = np.zeros_like(x)
        perturb[i] = eps
        fp = np.asarray(func(x + perturb))
        fm = np.asarray(func(x - perturb))
        jac[:, i] = (fp - fm) / (2.0 * eps)
    return jac


def gravity_vector(magnitude: float = 9.80665) -> np.ndarray:
    return np.array([0.0, 0.0, magnitude])

