from __future__ import annotations

import numpy as np

from .config import DEFAULT_PIPELINE_CONFIG
from .math_utils import (
    euler_to_rotation_matrix,
    gravity_vector,
    numerical_jacobian,
    wrap_angle,
    wrap_angles,
)


STATE_SIZE = 15
IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_ANG = slice(6, 9)
IDX_ACC_BIAS = slice(9, 12)
IDX_GYRO_BIAS = slice(12, 15)


class InertialPositionEKF:
    def __init__(self, config=DEFAULT_PIPELINE_CONFIG):
        self.config = config
        self.state = np.zeros(STATE_SIZE)
        self.covariance = np.eye(STATE_SIZE) * 1e-3

    def initialize(self, initial_angles_rad: np.ndarray, gyro_bias: np.ndarray | None = None):
        self.state[IDX_ANG] = initial_angles_rad
        if gyro_bias is not None:
            self.state[IDX_GYRO_BIAS] = gyro_bias

    def predict(self, acc_meas: np.ndarray, gyro_meas: np.ndarray, dt: float):
        dt = max(dt, 1e-3)

        def transition(x):
            return self._state_transition(x, acc_meas, gyro_meas, dt)

        prev_state = self.state.copy()
        self.state = transition(prev_state)
        F = numerical_jacobian(transition, prev_state)
        Q = self._process_noise(dt)
        self.covariance = F @ self.covariance @ F.T + Q
        self.state[IDX_ANG] = wrap_angles(self.state[IDX_ANG])
        
        # Prevent covariance explosion
        self._limit_covariance()

    def update(self, measurement: np.ndarray, measurement_fn, R: np.ndarray, residual_fn=None):
        if np.any(np.isnan(measurement)):
            return

        h = measurement_fn(self.state)
        residual = measurement - h if residual_fn is None else residual_fn(measurement, h)
        H = numerical_jacobian(measurement_fn, self.state)

        S = H @ self.covariance @ H.T + R
        # Check for singularity before solving
        try:
            K = self.covariance @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            # Singular matrix - add small regularization
            S += np.eye(S.shape[0]) * 1e-9
            K = self.covariance @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        self.state = self.state + K @ residual
        I = np.eye(self.covariance.shape[0])
        # Joseph form covariance update
        self.covariance = (I - K @ H) @ self.covariance @ (I - K @ H).T + K @ R @ K.T
        # Symmetrize covariance to prevent numerical drift
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.state[IDX_ANG] = wrap_angles(self.state[IDX_ANG])
        
        # Prevent covariance explosion after update
        self._limit_covariance()

    def _limit_covariance(self):
        """Prevent covariance explosion by clamping diagonal elements"""
        # Clamp position uncertainties
        for i in range(3):
            if self.covariance[i, i] > 100.0 ** 2:
                self.covariance[i, i] = 100.0 ** 2
        
        # Clamp velocity uncertainties
        for i in range(3, 6):
            if self.covariance[i, i] > 50.0 ** 2:
                self.covariance[i, i] = 50.0 ** 2
        
        # Clamp angle uncertainties
        for i in range(6, 9):
            if self.covariance[i, i] > (np.deg2rad(180)) ** 2:
                self.covariance[i, i] = (np.deg2rad(180)) ** 2
        
        # Clamp accelerometer bias uncertainties
        for i in range(9, 12):
            if self.covariance[i, i] > 10.0 ** 2:
                self.covariance[i, i] = 10.0 ** 2
        
        # Clamp gyro bias uncertainties
        for i in range(12, 15):
            if self.covariance[i, i] > (np.deg2rad(90)) ** 2:
                self.covariance[i, i] = (np.deg2rad(90)) ** 2

    def zero_velocity_update(self, R: np.ndarray):
        def vel_fn(x):
            return x[IDX_VEL]

        self.update(np.zeros(3), vel_fn, R)

    def _state_transition(self, state: np.ndarray, acc_meas: np.ndarray, gyro_meas: np.ndarray, dt: float) -> np.ndarray:
        next_state = state.copy()

        # Map body rates -> Euler angle rates (ZYX convention)
        roll, pitch, yaw = state[IDX_ANG]
        gyro_bias = state[IDX_GYRO_BIAS]
        gyro_input = gyro_meas - gyro_bias
        p, q, r = gyro_input
        
        sr, cr = np.sin(roll), np.cos(roll)
        tp = np.tan(pitch)
        cp = np.cos(pitch)
        if abs(cp) < 1e-6:
            cp = 1e-6  # Avoid singularity at pitch = ±90°
        
        # Euler rate transformation matrix
        J = np.array([
            [1.0,    sr*tp,   cr*tp],
            [0.0,    cr,      -sr   ],
            [0.0,    sr/cp,   cr/cp ],
        ])
        euler_dot = J @ gyro_input
        next_state[IDX_ANG] = state[IDX_ANG] + euler_dot * dt

        rotation = euler_to_rotation_matrix(*next_state[IDX_ANG])
        acc_bias = state[IDX_ACC_BIAS]
        corrected_acc = acc_meas - acc_bias
        acc_world = rotation @ corrected_acc - gravity_vector(self.config.detection.gravity)

        next_state[IDX_VEL] = state[IDX_VEL] + acc_world * dt
        next_state[IDX_POS] = (
            state[IDX_POS] + state[IDX_VEL] * dt + 0.5 * acc_world * dt * dt
        )

        # Biases modeled as random walk (handled in process noise)
        next_state[IDX_ACC_BIAS] = state[IDX_ACC_BIAS]
        next_state[IDX_GYRO_BIAS] = state[IDX_GYRO_BIAS]

        return next_state

    def _process_noise(self, dt: float) -> np.ndarray:
        noise = self.config.noise
        q = np.zeros((STATE_SIZE, STATE_SIZE))
        q[IDX_POS, IDX_POS] = np.eye(3) * noise.pos * dt
        q[IDX_VEL, IDX_VEL] = np.eye(3) * noise.vel * dt
        q[IDX_ANG, IDX_ANG] = np.eye(3) * noise.angles * dt
        q[IDX_ACC_BIAS, IDX_ACC_BIAS] = np.eye(3) * noise.accel_bias * dt
        q[IDX_GYRO_BIAS, IDX_GYRO_BIAS] = np.eye(3) * noise.gyro_bias * dt
        return q

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def get_position_cm(self) -> np.ndarray:
        return self.state[IDX_POS] * 100.0
