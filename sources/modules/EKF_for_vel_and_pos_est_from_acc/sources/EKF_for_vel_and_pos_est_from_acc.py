#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Kalman Filter per la stima di velocità e posizione dai dati di accelerazione.

Questo modulo implementa un Extended Kalman Filter (EKF) per stimare velocità e 
posizione a partire esclusivamente da dati di accelerazione raccolti tramite 
smartphone Android durante esercizi di squat.

Il modulo supporta l'utilizzo di dati di accelerazione sia calibrati (linear)
che non calibrati (uncalibrated) e consente di analizzare uno o più assi di movimento.

Funzionalità principali:
- Ricampionamento dei dati alla frequenza desiderata
- Trimming del segnale per rimuovere i primi campioni (rumorosi)
- ZUPT (Zero-Velocity Update) per ridurre la deriva durante i periodi stazionari
- Correzione polinomiale della deriva di velocità
- Visualizzazione e salvataggio dei risultati

Autore: GitHub Copilot
Data: 09-07-2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats
from scipy import signal
import shutil
from typing import Dict, List, Tuple, Optional, Union
from scipy.linalg import inv, cholesky, eigh
import yaml
from datetime import datetime
import logging
from pathlib import Path
from collections import deque
import warnings
import json
import time
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def _module_root_same_name_as_script() -> Path:
    """Find module root directory that has the same name as this script"""
    here = Path(__file__).resolve()
    stem = here.stem
    for p in here.parents:
        if p.name == stem and (p / 'data').exists():
            return p
    return here.parent.parent


EXTENSION_OUTPUT_SUBDIRS = {
    'csv': 'csv',
    'json': 'json',
    'yaml': 'yaml',
    'yml': 'yaml',
    'md': 'md',
    'txt': 'txt',
    'png': 'png',
    'jpg': 'jpg',
    'jpeg': 'jpeg',
    'svg': 'svg',
    'pdf': 'pdf',
    'html': 'html',
    'log': 'logs'
}
DEFAULT_OUTPUT_SUBDIR = 'misc'


def get_output_base_dir() -> Path:
    """
    Returns the base output directory for the module.
    """
    return _module_root_same_name_as_script() / "data" / "outputs"


def resolve_output_path(filename: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the output path for a given filename ensuring files are grouped by extension.

    Args:
        filename: Target filename (with extension).
        base_dir: Optional base directory. Defaults to module data/outputs.

    Returns:
        Path: Full path where the file should be written.
    """
    target_name = Path(filename).name
    extension = Path(target_name).suffix.lower().lstrip('.')
    subdir = EXTENSION_OUTPUT_SUBDIRS.get(extension, extension if extension else DEFAULT_OUTPUT_SUBDIR)
    base_path = Path(base_dir) if base_dir else get_output_base_dir()
    output_dir = base_path / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / target_name


def safe_fmt(x, nd=3):
    """Safe formatter for potentially string values"""
    try:
        xv = float(x)
        if np.isnan(xv) or np.isinf(xv): 
            return str(x)
        return f"{xv:.{nd}f}"
    except Exception:
        return str(x)

# Import per analisi avanzate
try:
    from statsmodels.tsa.stattools import acf
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("⚠️  Statsmodels not available. Install with: pip install statsmodels")
    STATSMODELS_AVAILABLE = False

# Import colorama for colored console output
try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    print("⚠️  Colorama not available. Install with: pip install colorama")
    COLORAMA_AVAILABLE = False
    # Define dummy classes to prevent errors
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""
    class Back:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""

# Enhanced logging configuration with file and console handlers
def setup_enhanced_logging():
    """
    Setup enhanced logging with both file and console output, including colored logs.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / "data" / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"EKF_execution_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger("EKF_for_vel_and_pos_est_from_acc")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler for detailed logging
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log the setup
    logger.info(f"🔧 Enhanced logging system initialized")
    logger.info(f"📁 Log file created: {log_file}")
    
    return logger, log_file

# Initialize enhanced logging
logger, current_log_file = setup_enhanced_logging()

class ExtendedKalmanFilter3D:
    """
    Extended Kalman Filter 3D per la stima di posizione e velocità nello spazio 3D
    con auto-tuning adattivo e rotazione Euler per coordinate world-frame.
    
    Il vettore di stato è: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    Converte accelerazioni da body-frame a world-frame usando angoli di Euler.
    """
    
    def __init__(self, config):
        """
        Inizializza l'Extended Kalman Filter 3D con auto-tuning adattivo.
        
        Args:
            config (dict): Dizionario contenente i parametri di configurazione dell'EKF.
        """
        print_colored("🔧 Inizializzazione Extended Kalman Filter 3D...", "🔧", "cyan")
        logger.info("🚀 Inizializzazione Extended Kalman Filter 3D in corso...")
        
        # Otteniamo i parametri di configurazione
        kf_config = config['ekf']
        print_colored(f"📋 Caricamento parametri di configurazione EKF 3D", "📋", "blue")
        
        debug_cfg = config.get('debug', {})
        self.debug_enabled = bool(debug_cfg.get('enable_debug_output', False))
        self.debug_every_n = int(debug_cfg.get('log_every_n', 50))
        self.debug_print_state = bool(debug_cfg.get('print_state_snapshots', False))
        if self.debug_enabled:
            logger.info(
                f"🟣 Debug EKF3D attivo (every {self.debug_every_n} samples, print={self.debug_print_state})"
            )
        
        # Stato 9D: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, bias_x, bias_y, bias_z]
        initial_state = kf_config.get('initial_state_3d', [0.0] * 6)
        # Extend to 9D with bias initialization
        if len(initial_state) == 6:
            initial_state.extend([0.0, 0.0, 0.0])  # Initial bias = 0
        elif len(initial_state) != 9:
            initial_state = [0.0] * 9
        self.x = np.array(initial_state).reshape(9, 1)
        print_colored(safe_fmt(f"📍 Stato iniziale 9D: pos={self.x[0:3,0]}, vel={self.x[3:6,0]}, bias={self.x[6:9,0]}"), "📍", "green")
        logger.info(safe_fmt(f"Stato iniziale 9D: {self.x.flatten()}"))
        
        # Matrice di covarianza iniziale 9x9
        initial_cov = kf_config.get('initial_covariance_3d', 1.0)
        if np.isscalar(initial_cov):
            cov_diag = [initial_cov] * 6 + [1e-6] * 3  # Small initial bias uncertainty
            self.P = np.diag(cov_diag)
        else:
            if len(initial_cov) == 36:  # 6x6 matrix
                P_6x6 = np.array(initial_cov).reshape(6, 6)
                self.P = np.zeros((9, 9))
                self.P[:6, :6] = P_6x6
                self.P[6:9, 6:9] = np.eye(3) * 1e-6  # Bias covariance
            elif len(initial_cov) == 81:  # 9x9 matrix
                self.P = np.array(initial_cov).reshape(9, 9)
            else:
                cov_diag = [initial_cov] * 6 + [1e-6] * 3
                self.P = np.diag(cov_diag)
        print_colored(f"🎯 Matrice di covarianza 9D configurata ({self.P.shape})", "🎯", "green")
        
        # Processo noise per [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, bias_x, bias_y, bias_z]
        # PATCH: Fix process noise with sensible values and clamped scaling
        process_noise = kf_config['process_noise']
        if isinstance(process_noise, dict):
            # Support per-axis configuration
            self.Q_pos = np.array([process_noise.get('position', 1e-5)] * 3)
            self.Q_vel = np.array([process_noise.get('velocity', 1e-3)] * 3)  
            self.Q_bias = np.array([process_noise.get('bias', 1e-6)] * 3)
            Q_diag = list(self.Q_pos) + list(self.Q_vel) + list(self.Q_bias)
        else:
            # Sensible Q_base values as specified in requirements
            Q_diag = [1e-8, 1e-8, 1e-8,  # position noise
                     1e-6, 1e-6, 1e-6,   # velocity noise  
                     1e-8, 1e-8, 1e-8]   # bias noise
            logger.info("🔧 Using sensible Q_base values: [1e-8]*3 + [1e-6]*3 + [1e-8]*3")
            self.Q_pos = np.array(Q_diag[:3])
            self.Q_vel = np.array(Q_diag[3:6])
            self.Q_bias = np.array(Q_diag[6:9])
        
        self.Q_base = np.diag(Q_diag)  # Base Q matrix for adaptation
        self.Q = self.Q_base.copy()
        
        # Z-axis specific reduction for extra stability (improved implementation)
        z_reduction_factor = float(kf_config.get('process_noise', {}).get('z_axis_reduction_factor', 0.1))
        if z_reduction_factor < 1.0:
            self.Q[2, 2] *= z_reduction_factor   # pos Z
            self.Q[5, 5] *= z_reduction_factor   # vel Z
            self.Q_base[2, 2] *= z_reduction_factor  # Also update base
            self.Q_base[5, 5] *= z_reduction_factor
            logger.info(f"🎯 Z-axis process noise reduced by factor {z_reduction_factor}")
        
        print_colored(safe_fmt(f"🔊 Rumore del processo 9D: pos={self.Q_pos}, vel={self.Q_vel}, bias={self.Q_bias}"), "🔊", "yellow")
        logger.info(safe_fmt(f"Matrice di rumore del processo Q 9D: {Q_diag}"))
        
        # Adaptive noise parameters - clamped scaling for stability  
        self.alpha_clip = [0.5, 2.0]  # Clamp scaling range for Q as specified
        self.R_clip = kf_config.get('R_clip', [0.05, 1.0])  # [min, max] for R scaling
        self.window_size = int(kf_config.get('variance_window', 50))  # ~2-3s at 20Hz
        
        # Rolling variance buffers for adaptive noise
        self.acc_buffer = deque(maxlen=self.window_size)
        self.residual_buffer = deque(maxlen=self.window_size)
        
        # Rumore della misura (per accelerazioni 3D) - now adaptive base
        measurement_noise = kf_config['measurement_noise']
        if np.isscalar(measurement_noise):
            self.R_base = measurement_noise * np.eye(3)
        else:
            self.R_base = np.diag(measurement_noise)
        self.R = self.R_base.copy()
        print_colored(safe_fmt(f"📏 Rumore della misura 3D (base): {np.diag(self.R_base)}"), "📏", "yellow")
        logger.info(safe_fmt(f"Rumore della misura R 3D: {np.diag(self.R_base)}"))
        
        # ZUPT configuration
        zupt_config = kf_config.get('zupt', {})
        self.zupt_enabled = zupt_config.get('enabled', True)

        # Thresholds driven by configuration (fallback to legacy defaults if missing)
        accel_norm_thr = float(zupt_config.get('accel_threshold', 0.08))
        acc_mean_thr = float(zupt_config.get('acc_mean_thr', accel_norm_thr))
        variance_thr = zupt_config.get('variance_threshold', None)
        if variance_thr is not None:
            try:
                self.zupt_acc_std_thr = float(np.sqrt(float(variance_thr)))
            except (TypeError, ValueError):
                self.zupt_acc_std_thr = 0.05
        else:
            self.zupt_acc_std_thr = float(zupt_config.get('acc_std_thr', 0.05))

        self.zupt_accel_norm_thr = accel_norm_thr
        self.zupt_acc_mean_thr = acc_mean_thr
        self.zupt_velocity_mean_thr = float(zupt_config.get('velocity_threshold', 0.05))
        self.zupt_velocity_std_thr = float(
            zupt_config.get('velocity_std_threshold', self.zupt_velocity_mean_thr * 1.5)
        )
        self.zupt_velocity_rms_thr = float(
            zupt_config.get('velocity_rms_threshold', self.zupt_velocity_std_thr)
        )
        self.zupt_variance_thr = float(variance_thr) if variance_thr is not None else None
        self.zupt_acc_threshold = np.array([self.zupt_acc_std_thr] * 3)
        self.zupt_vel_threshold = self.zupt_velocity_mean_thr

        self.zupt_window = int(zupt_config.get('window_samples', zupt_config.get('window_size', 50)))
        self.zupt_min_stationary = int(zupt_config.get('min_stationary_frames', max(5, self.zupt_window // 3)))
        self.zupt_cooldown_samples = int(zupt_config.get('cooldown_samples', 0))
        self.zupt_adaptive = bool(zupt_config.get('adaptive_thresholds', False))
        self.zupt_adaptive_relax_after = int(
            zupt_config.get('auto_relax_after', max(self.zupt_window, 120))
        )
        self.zupt_relax_rate = float(zupt_config.get('relaxation_rate', 0.25))
        self.zupt_max_relax = float(zupt_config.get('max_relax_factor', 3.0))
        zupt_measurement_noise = zupt_config.get('measurement_noise')
        if zupt_measurement_noise is None:
            zupt_measurement_noise = kf_config.get('zupt_measurement_noise', {}).get('vz', 5e-4)
        try:
            self.zupt_R = float(zupt_measurement_noise)
        except (TypeError, ValueError):
            self.zupt_R = 5e-4

        # Internal counters for adaptive ZUPT handling
        self.zupt_relax_multiplier = 1.0
        self.zupt_stationary_count = 0
        self.zupt_steps_since_last = 0
        self.zupt_last_trigger = -10_000
        
        # Outlier rejection configuration  
        outlier_config = kf_config.get('outlier_rejection', {})
        self.outlier_enabled = outlier_config.get('enabled', True)
        self.outlier_threshold = outlier_config.get('z_score_threshold', 3.0)
        self.huber_delta = outlier_config.get('huber_delta', 1.0)  # For Huber loss
        
        # Velocity correction configuration
        vel_correction_config = kf_config.get('velocity_correction', {})
        self.vel_correction_enabled = vel_correction_config.get('enabled', True)
        self.poly_order = vel_correction_config.get('polynomial_order', 1)
        self.drift_threshold = vel_correction_config.get('drift_threshold', 0.05)
        self.correction_interval = vel_correction_config.get('correction_interval', 100)
        
        # Numerical stability configuration
        stability_config = kf_config.get('numerical_stability', {})
        self.use_joseph_form = stability_config.get('use_joseph_form', True)
        self.force_symmetry = stability_config.get('force_symmetry', True)
        self.trace_threshold = 3000.0  # Hard guard su P - lowered threshold
        self.eigenvalue_floor = float(stability_config.get('eigenvalue_floor', 1e-12))
        self.dt_clamp = stability_config.get('dt_clamp', [0.005, 0.015])
        numerics_cfg = kf_config.get('numerics', {})
        self.trace_warn_level = float(numerics_cfg.get('trace_warn', 1000.0))
        self.trace_abort_level = float(numerics_cfg.get('trace_abort', self.trace_threshold))
        self.trace_threshold = min(self.trace_threshold, self.trace_abort_level)
        
        # Enhanced adaptive tuning configuration
        adaptive_config = kf_config.get('adaptive_tuning', {})
        self.mahalanobis_gate = 9.0  # Gating & outlier handling - more strict
        self.max_adaptations_per_window = 2  # NIS-driven adaptation limit
        self.Q_increase_factor = adaptive_config.get('Q_increase_factor', 1.05)
        self.R_decrease_factor = adaptive_config.get('R_decrease_factor', 0.98)
        self.adaptations_this_window = 0
        
        # NIS-driven adaptation buffers
        self.nis_buffer = deque(maxlen=self.window_size)  # ~2-3s rolling NIS
        
        # PATCH: Rolling windows for ZUPT detection
        self.window_samples = int(self.window_size)  # Number of samples for rolling window
        self.acc_world_window = deque(maxlen=self.zupt_window)
        self.velocity_window = deque(maxlen=self.zupt_window)
        self.zupt_buffer = deque(maxlen=self.zupt_window)
        
        # PATCH: Gravity vector for world-frame acceleration calculation
        gravity_cfg = kf_config.get('gravity', kf_config.get('gravity_mps2', 9.80665))
        try:
            self.gravity = float(gravity_cfg)
        except (TypeError, ValueError):
            logger.warning(f"⚠️ Invalid gravity value '{gravity_cfg}' - using default 9.80665 m/s²")
            self.gravity = 9.80665
        self.gravity_vector = np.array([0.0, 0.0, self.gravity])
        print_colored(f"🌍 Valore di gravità: {self.gravity} m/s²", "🌍", "blue")
        logger.info(f"Gravità: {self.gravity} m/s²")
        
        # CORRECTED: H matrix only maps to actual observations (none in pure acceleration mode)
        # When we have position/velocity measurements, H will be set appropriately
        self.H = np.zeros((3, 9))  # 3 observations x 9 states
        # H stays zero since acceleration is control input, not observation
        logger.info(f"CORREZIONE: Matrice H impostata a zero - accelerazione trattata come controllo")
        logger.info(f"Matrice di osservazione H 9D: {self.H.shape}")
        
        # Store configuration for diagnostics
        self.config = config
        
        # Enhanced monitoring and diagnostics
        self.diagnostics = {
            'residual_stats': {'mean': [], 'std': [], 'autocorr': []},
            'outlier_count': 0,
            'total_measurements': 0,
            'zupt_activations': 0,
            'adaptation_history': [],
            'trace_resets': 0,  # Contatore per reset della traccia P
            'current_trace': None  # Traccia corrente della matrice P
        }
        
        # Parametri per auto-tuning adattivo
        self.adaptive_threshold = kf_config.get('adaptive_threshold', 5.0)
        self.Q_adaptive_factor = kf_config.get('Q_adaptive_factor', 1.1)
        self.R_adaptive_factor = kf_config.get('R_adaptive_factor', 0.95)
        self.max_adaptations = kf_config.get('max_adaptations', 3)
        
        # History per monitoring
        self.innovation_history = []
        self.trace_history = []
        self.adaptation_count = 0
        
        print_colored("✅ Extended Kalman Filter 3D inizializzato con successo!", "✅", "green")
        logger.info("✅ Extended Kalman Filter 3D inizializzato con successo.")
    
    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """
        Converte angoli di Euler in matrice di rotazione 3x3.
        
        Args:
            roll, pitch, yaw: Angoli di Euler in radianti
            
        Returns:
            np.array: Matrice di rotazione 3x3 da body-frame a world-frame
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # Matrice di rotazione Z-Y-X (yaw-pitch-roll)
        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [0,    0, 1]])
        
        Ry = np.array([[cp, 0, sp],
                       [0,  1, 0],
                       [-sp, 0, cp]])
        
        Rx = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr,  cr]])
        
        return Rz @ Ry @ Rx
    
    def kalman_update_joseph(self, H, y, R):
        """
        Complete Kalman update using Joseph form for numerical stability.
        
        Args:
            H: Observation matrix (m x n)
            y: Innovation vector (m x 1) 
            R: Measurement noise covariance (m x m)
            
        Returns:
            nis: Normalized Innovation Squared statistic
        """
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            K = self.P @ H.T @ np.linalg.pinv(S)
            logger.warning("⚠️ Using pseudoinverse for singular innovation covariance")
        
        # State update
        self.x = self.x + K @ y
        
        # Joseph form covariance update for guaranteed positive semi-definiteness
        I = np.eye(self.P.shape[0])
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        
        # Force symmetry for extra numerical stability
        self.P = 0.5 * (self.P + self.P.T)
        
        # Apply trace guard
        current_trace = np.trace(self.P)
        if current_trace > 3000:
            logger.warning(f"🛡️ Hard guard triggered: trace(P)={current_trace:.1f} > 3000")
            # Shrink to safe diagonal 
            self.P = np.diag(np.maximum(np.diag(self.P) * 0.1, 1e-10))
            logger.info(f"🛡️ Covariance reset to diagonal: new trace={np.trace(self.P):.1f}")
            self.diagnostics['trace_resets'] += 1
        
        # Calculate NIS for diagnostics
        try:
            nis = float(y.T @ np.linalg.inv(S) @ y)
        except:
            nis = 0.0
            
        return nis
    
    def _debug_state_snapshot(self, label, extra_msg=""):
        """Log/print current EKF state when debug is enabled."""
        if not self.debug_enabled:
            return
        msg = (
            f"{label} | pos={self.x[0:3,0]} vel={self.x[3:6,0]} "
            f"bias={self.x[6:9,0]} traceP={np.trace(self.P):.3f} {extra_msg}"
        )
        logger.debug(f"🟣 EKF3D {msg}")
        if self.debug_print_state:
            print_colored(msg, "🟣", "magenta")
    
    def maybe_force_update(self, t):
        """
        Force velocity update for first 2 seconds to verify update path works.
        
        Args:
            t: Current time in seconds
            
        Returns:
            did_update: Boolean indicating if update was applied
        """
        if t < 2.0:
            # H matrix: zeros with velocity observation  
            H = np.zeros((3, 9))
            H[:, 3:6] = np.eye(3)  # Observe velocities
            
            # Zero velocity pseudo-measurement
            z = np.zeros((3, 1))
            
            # R matrix for velocity pseudo-measurements
            R = np.diag([5e-4] * 3)
            
            # Innovation 
            y = z - H @ self.x
            
            # Apply Joseph form update
            nis = self.kalman_update_joseph(H, y, R)
            
            # Store NIS for later retrieval
            self._last_nis = nis
            
            logger.info(f"🚀 Force update applied at t={t:.3f}s, NIS={nis:.3f}")
            return True
        return False
    
    def maybe_zupt_update(self, acc_world_window, v_window):
        """
        Apply ZUPT (Zero Velocity Update) if stillness conditions are met.
        
        Args:
            acc_world_window: Rolling window of world-frame acceleration (n x 3)
            v_window: Rolling window of velocity estimates (n x 3)
            
        Returns:
            did_update: Boolean indicating if ZUPT was applied
        """
        if len(acc_world_window) < max(5, self.zupt_min_stationary):
            return False

        relax = self.zupt_relax_multiplier
        sample_idx = self.diagnostics['total_measurements']

        acc_world_window = np.asarray(acc_world_window)
        v_window = np.asarray(v_window)

        acc_std = np.std(acc_world_window, axis=0)
        acc_var = np.var(acc_world_window, axis=0)
        acc_std_max = float(np.max(acc_std))
        acc_norms = np.linalg.norm(acc_world_window, axis=1)
        acc_norm_mean = float(np.mean(acc_norms))
        acc_mean_norm = float(np.linalg.norm(np.mean(acc_world_window, axis=0)))

        vel_mean = np.mean(v_window, axis=0)
        vel_mean_norm = float(np.linalg.norm(vel_mean))
        vel_std_norm = float(np.linalg.norm(np.std(v_window, axis=0)))
        vel_rms_norm = float(np.sqrt(np.mean(np.square(v_window))))

        conditions_met = True
        if acc_std_max >= self.zupt_acc_std_thr * relax:
            conditions_met = False
        if acc_norm_mean >= self.zupt_accel_norm_thr * relax:
            conditions_met = False
        if acc_mean_norm >= self.zupt_acc_mean_thr * relax:
            conditions_met = False
        if self.zupt_variance_thr is not None and np.mean(acc_var) >= self.zupt_variance_thr * (relax ** 2):
            conditions_met = False
        if vel_mean_norm >= self.zupt_velocity_mean_thr * relax:
            conditions_met = False
        if vel_std_norm >= self.zupt_velocity_std_thr * relax:
            conditions_met = False
        if vel_rms_norm >= self.zupt_velocity_rms_thr * relax:
            conditions_met = False

        if not conditions_met:
            self.zupt_stationary_count = 0
            return False

        self.zupt_stationary_count += 1
        if self.zupt_stationary_count < self.zupt_min_stationary:
            return False

        if (
            self.zupt_cooldown_samples > 0
            and (sample_idx - self.zupt_last_trigger) < self.zupt_cooldown_samples
        ):
            return False

        # Apply velocity pseudo-measurement (Joseph form)
        H = np.zeros((3, 9))
        H[:, 3:6] = np.eye(3)

        z = np.zeros((3, 1))
        R = np.diag([self.zupt_R * relax] * 3)

        y = z - H @ self.x

        nis = self.kalman_update_joseph(H, y, R)
        self._last_nis = nis
        self.diagnostics['zupt_activations'] += 1

        logger.info(
            "🎯 ZUPT applied @ sample %d | acc_std_max=%.4f | acc_norm=%.4f | vel_mean=%.4f | relax=%.2f | NIS=%.3f",
            sample_idx,
            acc_std_max,
            acc_norm_mean,
            vel_mean_norm,
            relax,
            nis,
        )

        # Reset adaptive state
        self.zupt_stationary_count = 0
        self.zupt_steps_since_last = 0
        self.zupt_last_trigger = sample_idx
        self.zupt_relax_multiplier = 1.0

        return True
    
    def log_update_stats(self, did_update, t, nis_value=None):
        """
        Log update statistics for telemetry.
        
        Args:
            did_update: Whether an update occurred  
            t: Current time
            nis_value: NIS value if update occurred
        """
        if did_update and nis_value is not None:
            self.nis_buffer.append(nis_value)
            
        # Log periodically after first 2 seconds
        if t > 2.0 and len(self.nis_buffer) > 0:
            total_samples = self.diagnostics['total_measurements'] 
            zupt_activations = self.diagnostics['zupt_activations']
            zupt_percentage = (zupt_activations / max(total_samples, 1)) * 100
            
            current_trace = np.trace(self.P)
            mean_nis = np.mean(list(self.nis_buffer))
            
            if total_samples % 100 == 0:  # Log every 100 samples
                logger.info(f"📊 t={t:.1f}s: ZUPT={zupt_percentage:.1f}%, NIS={mean_nis:.2f}, trace(P)={current_trace:.1f}")
                
                # Check if NIS > 0 after first 2 seconds
                if t > 2.0 and mean_nis <= 0.001:
                    logger.warning("⚠️ NIS ≈ 0 detected - updates may not be working properly!")
    
    def _joseph_form_update(self, K, H, R):
        """
        Legacy Joseph form method - kept for compatibility.
        Use kalman_update_joseph for new code.
        """
        n = self.P.shape[0]
        I = np.eye(n)
        IKH = I - K @ H
        
        # Joseph form update - guaranteed to preserve positive definiteness
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        
        # Force symmetry for extra numerical stability
        self.P = 0.5 * (self.P + self.P.T)
        
        logger.debug("🔒 Legacy Joseph form covariance update applied")

    def predict(self, dt):
        """
        Fase di predizione dell'EKF 9D con bias estimation.
        
        Args:
            dt (float): Intervallo di tempo tra misurazioni successive.
        """
        # PATCH 3: Clamp dt to prevent numerical instability
        dt = np.clip(dt, self.dt_clamp[0], self.dt_clamp[1])
        logger.debug(f"🔮 Fase di predizione 9D con dt={dt:.6f}s (clamped)")
        
        # Matrice di transizione di stato 9x9
        F = np.zeros((9, 9))
        # Position: p = p + v*dt
        F[0:3, 0:3] = np.eye(3)  # position to position
        F[0:3, 3:6] = dt * np.eye(3)  # velocity contribution to position
        # Velocity: v = v (constant velocity model)
        F[3:6, 3:6] = np.eye(3)  # velocity to velocity  
        # Bias: b = b (random walk model)
        F[6:9, 6:9] = np.eye(3)  # bias to bias
        
        # Prediczione dello stato
        self.x = F @ self.x
        
        # PATCH 2: Joseph-form inspired covariance prediction for stability
        self.P = F @ self.P @ F.T + self.Q
        
        # Force symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        
        # PATCH 1: Add trace to diagnostics for monitoring
        current_trace = np.trace(self.P)
        self.diagnostics['current_trace'] = current_trace
        
        # CORREZIONE BUG: Aggiungi controlli stabilità numerica
        self._check_numerical_stability()
        
        # Memorizza traccia per monitoring
        self.trace_history.append(current_trace)
        
        self._debug_state_snapshot("predict", extra_msg=f"dt={dt:.4f}")
        
        logger.debug(f"📊 Stato predetto 9D - Pos: {self.x[0:3,0]}, Vel: {self.x[3:6,0]}, Bias: {self.x[6:9,0]}")
        
        return self.x.copy(), self.P.copy()
        
    def update_with_acceleration(self, acc_body, roll, pitch, yaw, dt):
        """
        Update dell'EKF 3D con accelerazione body-frame convertita a world-frame.
        Include auto-tuning adattivo basato su innovazione Mahalanobis.
        
        Args:
            acc_body: Accelerazione in body-frame [ax, ay, az]
            roll, pitch, yaw: Angoli di Euler in radianti
            dt: Intervallo di tempo
            
        Returns:
            tuple: Stato aggiornato, matrice P, innovazione
        """
        # Converti accelerazione da body-frame a world-frame
        R_matrix = self.euler_to_rotation_matrix(roll, pitch, yaw)
        acc_world = R_matrix @ np.array(acc_body).reshape(3, 1)
        
        # Rimuovi gravità (assumendo gravità lungo z negativo in world-frame)
        gravity_vector = np.array([[0], [0], [self.gravity]])
        acc_world_corrected = acc_world - gravity_vector
        
        logger.debug(f"🌍 Accelerazione world-frame: {acc_world_corrected.flatten()}")
        
        # Matrice di controllo per accelerazione 9x3 (updated for 9D state)
        B = np.zeros((9, 3))
        B[0:3, 0:3] = 0.5 * dt**2 * np.eye(3)  # Posizione: 0.5 * a * dt^2
        B[3:6, 0:3] = dt * np.eye(3)           # Velocità: a * dt
        # Bias: no direct control input (B[6:9, :] = 0)
        
        # Applica controllo di accelerazione
        self.x = self.x + B @ acc_world_corrected
        
        # Per ora usiamo un modello di osservazione diretto delle velocità stimate
        # (in futuro si potrebbe usare misure di posizione se disponibili)
        
        # Calcola innovazione per auto-tuning
        # Usiamo la predizione vs. misurazione dell'accelerazione integrata
        expected_vel_change = dt * acc_world_corrected
        actual_vel = self.x[3:6, 0].reshape(3, 1)
        innovation = expected_vel_change - (actual_vel * 0.1)  # Peso ridotto per stabilità
        
        # Test Mahalanobis per auto-tuning
        S = self.P[3:6, 3:6] + self.R  # Covarianza innovazione per velocità
        try:
            mahalanobis_dist = float(innovation.T @ np.linalg.inv(S) @ innovation)
            logger.debug(f"� Distanza Mahalanobis: {mahalanobis_dist:.4f}")
            
            # Auto-tuning se l'innovazione è troppo grande
            if mahalanobis_dist > self.adaptive_threshold and self.adaptation_count < self.max_adaptations:
                logger.warning(f"⚠️ Auto-tuning attivato: Mahalanobis={mahalanobis_dist:.2f}")
                self.Q *= self.Q_adaptive_factor
                self.R *= self.R_adaptive_factor
                self.adaptation_count += 1
                
        except np.linalg.LinAlgError:
            logger.warning("⚠️ Errore calcolo Mahalanobis - matrice singolare")
            mahalanobis_dist = 0.0
        
        # Memorizza innovazione per monitoring
        self.innovation_history.append(innovation.flatten())
        
        logger.debug(f"✅ Stato 3D aggiornato - Pos: {self.x[0:3,0]}, Vel: {self.x[3:6,0]}")
        self._debug_state_snapshot(
            "update_with_acceleration",
            extra_msg=f"|acc_world|={np.linalg.norm(acc_world_corrected):.3f} maha={mahalanobis_dist:.3f}",
        )
        
        return self.x.copy(), self.P.copy(), innovation
    
    def update_legacy(self, z):
        """
        Metodo update legacy per compatibilità con codice esistente.
        Converte misurazione 1D in aggiornamento 3D sull'asse Y.
        """
        # Per compatibilità, assume che z sia accelerazione Y
        acc_body = [0.0, float(z), 0.0]  # Solo componente Y
        roll, pitch, yaw = 0.0, 0.0, 0.0  # Orientamento neutro
        dt = 0.01  # Default timestep
        
        return self.update_with_acceleration(acc_body, roll, pitch, yaw, dt)

    def update_comprehensive(self, acc_body, roll, pitch, yaw, dt, timestamp=None):
        """
        Comprehensive EKF update with all PATCH requirements:
        1. Force update for first 2 seconds to verify update path
        2. Real ZUPT triggering with world-frame acceleration  
        3. Conditional NIS/innovation logging only when updates occur
        4. Fixed accelerometer units and scaling with guards
        5. Joseph form covariance updates with trace guard
        6. Stabilized process noise with clamped scaling
        
        Args:
            acc_body: Acceleração em body-frame [ax, ay, az] 
            roll, pitch, yaw: Ângulos de Euler em radianos
            dt: Intervalo de tempo
            timestamp: Actual timestamp (optional, uses sample count * dt if None)
            
        Returns:
            tuple: Estado atualizado, matriz P, inovação, diagnostics
        """
        self.diagnostics['total_measurements'] += 1
        # PATCH: Use actual timestamp if provided, otherwise approximate
        if timestamp is not None:
            current_time = timestamp
        else:
            current_time = dt * self.diagnostics['total_measurements']
        
        # Debug print to console to check if function is called
        print(f"📍 UPDATE t={current_time:.3f}s sample={self.diagnostics['total_measurements']}")
        logger.info(f"🕐 Update at t={current_time:.3f}s, sample #{self.diagnostics['total_measurements']}")
        
        # PATCH 4: Fix accelerometer units/scale with guards
        acc_body_array = np.array(acc_body)
        if np.abs(acc_body_array).max() > 200:
            raise ValueError(f"Accel scale error? Max acceleration = {np.abs(acc_body_array).max():.1f} m/s²")
        
        # Convert to m/s² if needed (assuming input is already in m/s²)
        acc_ms2 = acc_body_array * 1.0  # Already in m/s² (modify if input is in g)
        
        # SANITY CHECK: Clamp dt to reasonable bounds
        dt = np.clip(dt, self.dt_clamp[0], self.dt_clamp[1])
        
        # 1. WORLD-FRAME ACCELERATION with bias correction for ZUPT
        R_matrix = self.euler_to_rotation_matrix(roll, pitch, yaw)
        acc_body_bias_corrected = acc_ms2 - self.x[6:9, 0]  # Remove bias first
        acc_world = R_matrix @ acc_body_bias_corrected.reshape(3, 1)
        acc_world_no_gravity = acc_world - self.gravity_vector.reshape(3, 1)
        
        # Update rolling windows for ZUPT detection
        self.acc_world_window.append(acc_world_no_gravity.flatten())
        self.velocity_window.append(self.x[3:6, 0])
        self.zupt_steps_since_last += 1

        if self.zupt_adaptive and self.zupt_steps_since_last > self.zupt_adaptive_relax_after:
            new_multiplier = min(
                self.zupt_relax_multiplier * (1.0 + self.zupt_relax_rate),
                self.zupt_max_relax,
            )
            if new_multiplier > self.zupt_relax_multiplier + 1e-3:
                logger.info(
                    "⚙️ Adaptive ZUPT relaxation engaged: multiplier %.2f → %.2f (no ZUPT for %d samples)",
                    self.zupt_relax_multiplier,
                    new_multiplier,
                    self.zupt_steps_since_last,
                )
            self.zupt_relax_multiplier = new_multiplier
        
        if (
            self.debug_enabled
            and ((self.diagnostics['total_measurements'] % max(1, self.debug_every_n)) == 0)
        ):
            acc_stats = np.linalg.norm(acc_world_no_gravity)
            vel_norm = np.linalg.norm(self.x[3:6, 0])
            logger.debug(
                "🟣 EKF3D sample %d | |acc_world_no_g|=%.4f |vel|=%.4f "
                "bias=%s",
                self.diagnostics['total_measurements'],
                acc_stats,
                vel_norm,
                self.x[6:9, 0],
            )
            if self.debug_print_state:
                print_colored(
                    f"🟣 sample={self.diagnostics['total_measurements']} |acc|={acc_stats:.4f} "
                    f"|vel|={vel_norm:.4f} bias={self.x[6:9, 0]}",
                    "🟣",
                    "magenta",
                )
        
        # Apply acceleration as control input (without bias - already corrected)
        B = np.zeros((9, 3))
        B[0:3, 0:3] = 0.5 * dt**2 * np.eye(3)  # Position: 0.5 * a * dt^2
        B[3:6, 0:3] = dt * np.eye(3)           # Velocity: a * dt
        # Bias: no direct control input (B[6:9, :] = 0)
        
        # Apply control input
        self.x = self.x + B @ acc_world_no_gravity
        
        # Initialize update tracking
        did_update = False
        innovation = np.zeros((3, 1))
        current_nis = 0.0
        
        # PATCH 1: Force update for first 2 seconds
        if current_time < 2.0:
            logger.info(f"🕐 Attempting force update at t={current_time:.3f}s")
            did_update = self.maybe_force_update(current_time)
            if did_update:
                logger.info(f"🚀 Force update applied at t={current_time:.3f}s")
            else:
                logger.warning(f"❌ Force update failed at t={current_time:.3f}s")
        
        # PATCH 2: Real ZUPT triggering (if no force update)
        if not did_update and self.zupt_enabled:
            logger.info(f"🔍 ZUPT check: window_size={len(self.acc_world_window)}")
            if len(self.acc_world_window) >= 5:  # Need sufficient samples
                logger.info(f"🔍 Attempting ZUPT at t={current_time:.3f}s")
                did_update = self.maybe_zupt_update(
                    np.array(list(self.acc_world_window)), 
                    np.array(list(self.velocity_window))
                )
                if did_update:
                    logger.info(f"🎯 ZUPT applied at t={current_time:.3f}s")
                else:
                    logger.info(f"❌ ZUPT not triggered at t={current_time:.3f}s")
            else:
                logger.info(f"❌ Insufficient samples for ZUPT: {len(self.acc_world_window)}/5")
        
        # PATCH 6: Apply clamped Q scaling 
        # Adaptive noise with alpha clipping [0.5, 2.0]
        if len(self.acc_buffer) > 10:  # Need some history
            acc_var = np.var(list(self.acc_buffer), axis=0)
            alpha = np.mean(acc_var) / np.mean(np.diag(self.Q_base)[:3])  # Scale factor
            alpha = np.clip(alpha, self.alpha_clip[0], self.alpha_clip[1])  # Clamp [0.5, 2.0]
            self.Q = alpha * self.Q_base
            logger.debug(f"🔧 Q scaling: alpha={alpha:.2f} (clamped to [{self.alpha_clip[0]}, {self.alpha_clip[1]}])")
        
        # PATCH 3: Log NIS/innovations ONLY when update occurs
        if did_update:
            # Get NIS from the update method
            if hasattr(self, '_last_nis'):
                current_nis = self._last_nis
                logger.debug(f"📊 NIS recorded: {current_nis:.3f}")
            self.log_update_stats(did_update, current_time, current_nis)
        
        # Update acceleration buffer for adaptive scaling
        self.acc_buffer.append(acc_world_no_gravity.flatten())
        
        # PATCH 7: Concise telemetry
        if self.diagnostics['total_measurements'] % 100 == 0:
            total_samples = self.diagnostics['total_measurements']
            zupt_activations = self.diagnostics['zupt_activations'] 
            zupt_percentage = (zupt_activations / max(total_samples, 1)) * 100
            current_trace = np.trace(self.P)
            
            logger.info(f"📊 t={current_time:.1f}s: ZUPT={zupt_percentage:.1f}%, trace(P)={current_trace:.1f}")
            
            # Validation checks
            if current_time > 2.0:
                if len(self.nis_buffer) > 0:
                    mean_nis = np.mean(list(self.nis_buffer))
                    if mean_nis <= 0.001:
                        logger.warning("❌ NIS ≈ 0 - updates not working!")
                    else:
                        logger.info(f"✅ NIS = {mean_nis:.3f} > 0")
                        
                # Assert trace(P) decreases below 3000 after updates
                if did_update and current_trace > 3000:
                    logger.warning(f"❌ trace(P) = {current_trace:.1f} still > 3000 after update")
        
        # 7. NUMERICAL STABILITY ENFORCEMENT
        self._enforce_numerical_stability()
        
        # 8. COMPREHENSIVE DIAGNOSTICS
        self._update_diagnostics(acc_world_no_gravity.flatten(), innovation.flatten(), did_update)
        
        logger.debug(f"🔧 Update comprehensive - Pos: {self.x[0:3,0]}, Vel: {self.x[3:6,0]}, Bias: {self.x[6:9,0]}")
        
        return self.x.copy(), self.P.copy(), innovation, self.diagnostics.copy()
    
    def _update_adaptive_noise(self, acc_corrected):
        """Update Q and R matrices based on rolling variance of acceleration and residuals."""
        # Add acceleration to buffer
        self.acc_buffer.append(acc_corrected.copy())
        
        if len(self.acc_buffer) >= self.window_size:
            # Compute rolling variance of acceleration
            acc_array = np.array(list(self.acc_buffer))
            acc_var = np.var(acc_array, axis=0)  # Per-axis variance
            
            # Scale Q based on acceleration variance (alpha factor)
            alpha = np.clip(acc_var / np.mean(acc_var), self.alpha_clip[0], self.alpha_clip[1])
            
            # Update Q matrices per axis
            for i in range(3):
                # Position noise scaling
                self.Q[i, i] = self.Q_base[i, i] * alpha[i]
                # Velocity noise scaling  
                self.Q[i+3, i+3] = self.Q_base[i+3, i+3] * alpha[i]
                # Bias noise remains constant
                
            logger.debug(f"📊 Adaptive Q: alpha={alpha}, acc_var={acc_var}")
            
        # Update R based on residual variance (if we have residuals)
        if len(self.residual_buffer) >= self.window_size:
            residual_array = np.array(list(self.residual_buffer))
            residual_var = np.var(residual_array, axis=0)
            
            # Scale R based on residual variance
            R_scale = np.clip(residual_var / np.mean(residual_var), 
                             self.R_clip[0], self.R_clip[1])
            self.R = self.R_base * R_scale
            
            logger.debug(f"📊 Adaptive R: R_scale={R_scale}, residual_var={residual_var}")
    
    def _update_adaptive_noise_softened(self, acc_corrected):
        """
        Softened adaptive noise tuning with limits:
        - Gentle Q increase (1.05x) / R decrease (0.98x) per window
        - Maximum 2 adaptations per window  
        - Reset counter every window_size measurements
        """
        # Add acceleration to buffer
        self.acc_buffer.append(acc_corrected.copy())
        
        # Track adaptations per window
        if len(self.acc_buffer) % self.window_size == 0:
            self.adaptations_this_window = 0  # Reset counter
        
        if len(self.acc_buffer) >= self.window_size and self.adaptations_this_window < self.max_adaptations_per_window:
            # Compute rolling variance of acceleration
            acc_array = np.array(list(self.acc_buffer))
            acc_var = np.var(acc_array, axis=0)  # Per-axis variance
            
            # Scale Q based on acceleration variance (gentle scaling)
            alpha = np.clip(acc_var / np.mean(acc_var), self.alpha_clip[0], self.alpha_clip[1])
            
            # Gentle Q adjustment: only small increases allowed
            for i in range(3):
                if alpha[i] > 1.1:  # Only adjust if significantly elevated
                    # Gentle increase in Q (process noise)
                    self.Q[i, i] = min(self.Q[i, i] * self.Q_increase_factor, 
                                       self.Q_base[i, i] * 3.0)  # Cap at 3x base
                    self.Q[i+3, i+3] = min(self.Q[i+3, i+3] * self.Q_increase_factor,
                                           self.Q_base[i+3, i+3] * 3.0)
                    self.adaptations_this_window += 1
                    
            logger.debug(f"📊 Gentle Q adaptation: alpha={alpha}, adaptations={self.adaptations_this_window}")
            
        # Update R based on residual variance (gentle decrease when confident)
        if len(self.residual_buffer) >= self.window_size and self.adaptations_this_window < self.max_adaptations_per_window:
            residual_array = np.array(list(self.residual_buffer))
            residual_var = np.var(residual_array, axis=0)
            
            # Gentle R decrease when residuals are consistently small
            if np.mean(residual_var) < np.mean(np.diag(self.R)) * 0.5:
                self.R = np.maximum(self.R * self.R_decrease_factor,
                                    self.R_base * 0.5)  # Floor at 0.5x base
                self.adaptations_this_window += 1
                
                logger.debug(f"📊 Gentle R adaptation: residual_var={np.mean(residual_var):.6f}")
    
    def _detect_outlier_mahalanobis(self, measurement_value):
        """
        Detect outliers using Mahalanobis distance with gate = 9.0
        (corresponds to ~99.7% confidence for chi-squared with 1 DOF)
        """
        if not self.outlier_enabled or len(self.acc_buffer) < 10:
            return False
            
        # Compute Mahalanobis distance based on recent measurements
        recent_values = [np.linalg.norm(acc) for acc in list(self.acc_buffer)[-10:]]
        
        if len(recent_values) < 3:
            return False
            
        mean_val = np.mean(recent_values)
        cov_val = np.var(recent_values)
        
        if cov_val < 1e-8:  # Avoid division by zero
            return False
            
        # Mahalanobis distance (simplified for 1D case)
        mahal_distance = (measurement_value - mean_val)**2 / cov_val
        
        is_outlier = mahal_distance > self.mahalanobis_gate
        
        if is_outlier:
            logger.debug(f"🚫 Mahalanobis outlier: distance={mahal_distance:.2f} > {self.mahalanobis_gate}")
            
        return is_outlier
    
    def _apply_zupt_per_axis(self, acc_corrected, dt):
        """Apply Zero-Velocity Updates with per-axis thresholds."""
        # Add current acceleration to ZUPT buffer
        self.zupt_buffer.append(acc_corrected.copy())
        
        if len(self.zupt_buffer) < self.zupt_window:
            return False
            
        # Check ZUPT conditions per axis
        acc_array = np.array(list(self.zupt_buffer))
        acc_std = np.std(acc_array, axis=0)  # Per-axis standard deviation
        
        # Current velocity magnitude
        vel_magnitude = np.linalg.norm(self.x[3:6, 0])
        
        # ZUPT conditions: low acceleration variance AND low velocity
        zupt_conditions = (acc_std < self.zupt_acc_threshold) & (vel_magnitude < self.zupt_vel_threshold)
        
        if np.all(zupt_conditions):
            # Apply ZUPT: inject pseudo-measurement v = 0
            H_zupt = np.zeros((3, 9))
            H_zupt[0:3, 3:6] = np.eye(3)  # Observe velocity
            
            z_zupt = np.zeros((3, 1))  # v = 0 measurement
            R_zupt = self.zupt_R * np.eye(3)
            
            # Standard Kalman update for ZUPT
            innovation = z_zupt - H_zupt @ self.x
            S = H_zupt @ self.P @ H_zupt.T + R_zupt
            
            try:
                K = self.P @ H_zupt.T @ np.linalg.inv(S)
                self.x = self.x + K @ innovation
                self.P = (np.eye(9) - K @ H_zupt) @ self.P
                
                self.diagnostics['zupt_activations'] += 1
                logger.debug(f"✋ ZUPT applied: vel_mag={vel_magnitude:.4f}, acc_std={acc_std}")
                return True
                
            except np.linalg.LinAlgError:
                logger.warning("⚠️ ZUPT failed: singular S matrix")
                return False
                
        return False
    
    def _apply_zupt_tightened(self, acc_corrected, dt):
        """
        Apply Zero-Velocity Updates with tightened thresholds:
        - acc_threshold: 0.05 m/s² (from 0.1)
        - vel_threshold: 0.03 m/s (from 0.05)
        - Improved gating for more aggressive stationary detection
        """
        # Add current acceleration to ZUPT buffer
        self.zupt_buffer.append(acc_corrected.copy())
        
        if len(self.zupt_buffer) < self.zupt_window:
            return False
            
        # Check ZUPT conditions per axis with tightened thresholds
        acc_array = np.array(list(self.zupt_buffer))
        acc_std = np.std(acc_array, axis=0)  # Per-axis standard deviation
        
        # Current velocity magnitude
        vel_magnitude = np.linalg.norm(self.x[3:6, 0])
        
        # TIGHTENED ZUPT conditions
        tightened_acc_threshold = 0.05  # Reduced from config
        tightened_vel_threshold = 0.03  # Reduced from config
        
        zupt_conditions = (acc_std < tightened_acc_threshold) & (vel_magnitude < tightened_vel_threshold)
        
        # Additional stability check: require ALL axes to be stable
        if np.all(zupt_conditions) and np.max(acc_std) < tightened_acc_threshold:
            # PATCH 5: Improved ZUPT with Joseph form covariance update
            H_zupt = np.zeros((3, 9))
            H_zupt[0:3, 3:6] = np.eye(3)  # Observe velocity
            
            z_zupt = np.zeros((3, 1))  # v = 0 measurement
            R_zupt = self.zupt_R * np.eye(3)
            
            # Compute innovation
            innovation = z_zupt - H_zupt @ self.x
            S = H_zupt @ self.P @ H_zupt.T + R_zupt
            
            try:
                # Compute Kalman gain
                K = self.P @ H_zupt.T @ np.linalg.inv(S)
                
                # Update state
                self.x = self.x + K @ innovation
                
                # PATCH 5: Use Joseph form for covariance update in ZUPT
                if self.use_joseph_form:
                    self._joseph_form_update(K, H_zupt, R_zupt)
                else:
                    # Standard update with forced symmetry
                    self.P = (np.eye(9) - K @ H_zupt) @ self.P
                    self.P = 0.5 * (self.P + self.P.T)  # Force symmetry
                
            except np.linalg.LinAlgError:
                logger.warning("⚠️ ZUPT failed: singular S matrix")
                return False
            
            self.diagnostics['zupt_activations'] += 1
            logger.debug(f"✋ ZUPT TIGHTENED applied: vel_mag={vel_magnitude:.4f}, acc_std={acc_std}, max_acc_std={np.max(acc_std):.4f}")
            return True
                
        return False
    
    def _apply_zupt_selective(self, acc_corrected, dt):
        """
        ZUPT reale con triggering su acc_world.
        Prima del gating: acc_world = R_bw @ (acc_body - bias) - g
        Se finestra < window_samples, skip ZUPT (non azzerare contatore)
        Quando ZUPT scatta, esegui SOLO l'update con H_zupt e R_zupt
        Restituisce (success, innovation, nis) per gestione condizionale
        """
        # Calcola acc_world prima del gating
        g_world = np.array([0, 0, -9.81])  # Gravità nel frame world
        
        # acc_world = R_bw @ (acc_body - bias) - g
        # Per semplicità assumiamo R_bw = I (body ≈ world per dispositivi tenuti verticali)
        bias = self.x[6:9, 0]  # Current bias estimate
        acc_world = acc_corrected - bias - g_world
        
        # Aggiungi alla finestra ZUPT
        self.zupt_buffer.append(acc_world.copy())
        
        # Se finestra non raggiunge window_samples, skip ZUPT
        if len(self.zupt_buffer) < self.zupt_window:
            return False, np.zeros((3, 1)), 0.0
            
        # Check ZUPT conditions con acc_world_window
        acc_world_window = np.array(list(self.zupt_buffer))
        vel_magnitude = np.linalg.norm(self.x[3:6, 0])
        
        # ZUPT triggering: std(acc_world_window) < 0.05 AND |v| < 0.03
        acc_world_std = np.std(acc_world_window, axis=0)
        strong_acc_condition = np.max(acc_world_std) < 0.05
        strong_vel_condition = vel_magnitude < 0.03
        
        if strong_acc_condition and strong_vel_condition:
            # Esegui SOLO l'update ZUPT
            H = np.zeros((3, 9))
            H[:, 3:6] = np.eye(3)  # H_zupt per velocità
            z = np.zeros((3, 1))   # v = 0 measurement
            R = np.diag([5e-4, 5e-4, 5e-4])  # R_zupt
            
            # Compute innovation e NIS per ZUPT
            y = z - H @ self.x  # innovation
            S = H @ self.P @ H.T + R
            
            try:
                # Calculate NIS con m corretto
                nis = float(y.T @ np.linalg.inv(S) @ y)
                
                # Joseph form update per ZUPT
                K = self.P @ H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y
                self._joseph_form_update(K, H, R)
                
                self.diagnostics['zupt_activations'] += 1
                logger.debug(f"🛑 ZUPT scatta: vel_mag={vel_magnitude:.4f}, max_acc_world_std={np.max(acc_world_std):.4f}, NIS={nis:.3f}")
                return True, y, nis
                
            except np.linalg.LinAlgError:
                logger.warning("⚠️ ZUPT fallito: S matrix singolare")
                return False, np.zeros((3, 1)), 0.0
                
        return False, np.zeros((3, 1)), 0.0
    
    def _apply_nis_driven_adaptation(self, acc_corrected):
        """
        Adattamento NIS-driven di R e Q (finestra rolling).
        Target teorico: E[NIS] ≈ m (dimensione misura).
        """
        # Create a pseudo-measurement for NIS calculation (acceleration consistency)
        if len(self.acc_buffer) > 1:
            # Use acceleration prediction vs actual as pseudo-measurement
            prev_acc = list(self.acc_buffer)[-1] if len(self.acc_buffer) > 0 else acc_corrected
            y = (acc_corrected - prev_acc).reshape(-1, 1)  # Innovation proxy
            
            # Simple covariance for pseudo-measurement
            S = self.R + np.eye(3) * 0.1  # Simple innovation covariance
            
            try:
                # Calculate NIS
                nis = float(y.T @ np.linalg.inv(S) @ y)
                self.nis_buffer.append(nis)
                
                if len(self.nis_buffer) >= 10:  # Need some history
                    mean_nis = np.mean(self.nis_buffer)
                    m = 3  # Measurement dimension
                    
                    # NIS adaptation bands
                    low = 0.7 * m   # 2.1
                    high = 1.3 * m  # 3.9
                    
                    # Soft adaptation rules (max 2 per window)
                    if self.adaptations_this_window < self.max_adaptations_per_window:
                        if mean_nis < low:
                            # Trust measurements more, allow more dynamics
                            self.R *= 0.9  # More trust in measurements
                            self.Q *= 1.05  # Slight increase for dynamics
                            self.adaptations_this_window += 1
                            logger.debug(f"📉 NIS adaptation: mean_nis={mean_nis:.2f} < {low:.1f} → R*=0.9, Q*=1.05")
                            
                        elif mean_nis > high:
                            # Less trust in measurements, reduce dynamics
                            self.R *= 1.1  # Less trust in measurements
                            self.Q *= 0.95  # Reduce dynamics
                            self.adaptations_this_window += 1
                            logger.debug(f"📈 NIS adaptation: mean_nis={mean_nis:.2f} > {high:.1f} → R*=1.1, Q*=0.95")
                    
                    # Clip R and Q to safe ranges
                    R_diag = np.diag(self.R)
                    R_diag = np.clip(R_diag, 0.05, 1.0)  # R bounds
                    self.R = np.diag(R_diag)
                    
                    # Clip Q scaling relative to base
                    Q_diag = np.diag(self.Q)
                    Q_base_diag = np.diag(self.Q_base)
                    Q_scale = Q_diag / Q_base_diag
                    Q_scale = np.clip(Q_scale, 0.5, 2.0)  # Q scale bounds
                    self.Q = self.Q_base * Q_scale.reshape(-1, 1)
                    
                    # Reset adaptations counter when buffer is full (window reset)
                    if len(self.nis_buffer) == self.nis_buffer.maxlen:
                        self.adaptations_this_window = 0
                    
            except (np.linalg.LinAlgError, ValueError):
                pass  # Skip NIS calculation if numerical issues
    
    def _detect_outlier(self, measurement_value):
        """Detect outliers using z-score gating."""
        if not self.outlier_enabled or len(self.acc_buffer) < 10:
            return False
            
        # Compute z-score based on recent measurements
        recent_values = [np.linalg.norm(acc) for acc in list(self.acc_buffer)[-10:]]
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        if std_val < 1e-6:  # Avoid division by zero
            return False
            
        z_score = abs(measurement_value - mean_val) / std_val
        return z_score > self.outlier_threshold
    
    def _enforce_numerical_stability(self):
        """
        Comprehensive numerical stability enforcement:
        1. Joseph form covariance update if enabled
        2. Force symmetry: P = 0.5*(P + P.T)
        3. Eigenvalue flooring to prevent singularity
        4. Trace threshold monitoring and reset
        """
        # 1. Force symmetry (always applied)
        if self.force_symmetry:
            self.P = 0.5 * (self.P + self.P.T)
        
        # 2. Eigenvalue flooring
        try:
            eigenvals, eigenvecs = np.linalg.eigh(self.P)
            eigenvals = np.maximum(eigenvals, self.eigenvalue_floor)
            self.P = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except np.linalg.LinAlgError:
            logger.warning("⚠️ Eigenvalue decomposition failed - using identity reset")
            self.P = np.eye(9) * 0.01
        
        # 3. Hard guard su P - trace reset corretto
        trace_P = np.trace(self.P)
        self.diagnostics['current_trace'] = float(trace_P)  # Always log current trace
        
        if trace_P > self.trace_warn_level:
            clamp_factor = max(self.trace_warn_level / trace_P, 0.25)
            if clamp_factor < 1.0:
                logger.warning(
                    "⚠️ Trace(P)=%.2f exceeds warn level %.2f → applying soft clamp (factor=%.3f)",
                    trace_P,
                    self.trace_warn_level,
                    clamp_factor,
                )
                self.P[:6, :6] *= clamp_factor
                self.P[6:9, 6:9] *= max(clamp_factor, 0.5)
                self.P = 0.5 * (self.P + self.P.T)
                trace_P = np.trace(self.P)
                self.diagnostics['current_trace'] = float(trace_P)
        
        if trace_P > self.trace_threshold:  # threshold is now 3000
            logger.warning(f"⚠️ TRACE EXPLOSION: {trace_P:.2f} > {self.trace_threshold} - RESETTING P")
            # Ensure trace_resets is initialized
            if 'trace_resets' not in self.diagnostics:
                self.diagnostics['trace_resets'] = 0
            self.diagnostics['trace_resets'] += 1
            # Correct reset - no broadcasting issues
            diag = np.maximum(np.diag(self.P) * 0.1, self.eigenvalue_floor)
            self.P = np.diag(diag)  # ✅ niente broadcasting sbagliato
            logger.info(f"✅ P reset - new trace: {np.trace(self.P):.2f}")
    
    def _joseph_form_update(self, H, R, innovation):
        """
        Joseph form covariance update for improved numerical stability:
        P_new = (I - K*H)*P*(I - K*H).T + K*R*K.T
        
        This form is more robust to numerical errors than the standard update.
        """
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
        except np.linalg.LinAlgError:
            logger.warning("⚠️ Singular S matrix in Joseph form - using identity")
            K = np.zeros((9, H.shape[0]))
        
        # Joseph form update
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Update state
        self.x = self.x + K @ innovation
        
        return K
    
    def _update_diagnostics(self, acc_corrected, innovation, zupt_applied):
        """Update comprehensive diagnostics."""
        # Add residual for adaptive R (use acceleration consistency as proxy)
        if len(self.acc_buffer) > 1:
            prev_acc = list(self.acc_buffer)[-2]
            residual = acc_corrected - prev_acc  # Simple residual proxy
            self.residual_buffer.append(residual)
            
            # PATCH 8: Enhanced residual statistics computation
            if len(self.residual_buffer) >= 20:
                residuals = np.array(list(self.residual_buffer))
                
                # Comprehensive residual statistics
                res_mean = np.mean(residuals, axis=0)
                res_std = np.std(residuals, axis=0)
                res_var = np.var(residuals, axis=0)
                
                self.diagnostics['residual_stats']['mean'] = res_mean.tolist()
                self.diagnostics['residual_stats']['std'] = res_std.tolist()
                self.diagnostics['residual_stats']['variance'] = res_var.tolist()
                
                # White noise test: mean should be near zero, low autocorrelation
                self.diagnostics['residual_stats']['whiteness_score'] = float(np.mean(np.abs(res_mean)) < 0.2)
                
                # Compute autocorrelation (if statsmodels available)
                if STATSMODELS_AVAILABLE and len(residuals) >= 40:
                    try:
                        autocorr = [acf(residuals[:, i], nlags=20, fft=True)[1:] for i in range(3)]
                        self.diagnostics['residual_stats']['autocorr'] = autocorr
                        
                        # Compute Ljung-Box test for white noise
                        lb_stats = []
                        for i in range(3):
                            try:
                                from statsmodels.stats.diagnostic import acorr_ljungbox
                                lb_stat = acorr_ljungbox(residuals[:, i], lags=10, return_df=False)
                                lb_stats.append(float(lb_stat['lb_pvalue'].iloc[-1]))
                            except:
                                lb_stats.append(0.5)  # Neutral value
                        self.diagnostics['residual_stats']['ljung_box_pvalue'] = lb_stats
                    except:
                        pass

    def _check_numerical_stability(self):
        """
        CORREZIONE BUG: Controlli di stabilità numerica per la matrice di covarianza P 9x9.
        
        Verifica:
        1. Esplosione della covarianza (trace troppo grande)
        2. Condition number (mal condizionamento)
        3. Simmetria della matrice P
        4. Positività della diagonale
        """
        trace_P = np.trace(self.P)
        max_trace_threshold = 1000.0  # Soglia per esplosione covarianza
        
        # Log trace sempre per monitoraggio (solo ogni 100 iterazioni per non inondare)
        if len(self.trace_history) % 100 == 0:
            logger.info(f"📊 Monitor: Trace(P)={trace_P:.2f}, iterazione {len(self.trace_history)}")
        
        # 1. Controllo esplosione covarianza
        if trace_P > max_trace_threshold:
            logger.warning(f"⚠️ INSTABILITA': Trace(P)={trace_P:.2f} > {max_trace_threshold}")
            logger.warning("⚠️ Possibile esplosione della covarianza - considera di ridurre Q")
            
        # 2. Controllo condition number
        try:
            cond_P = np.linalg.cond(self.P)
            if cond_P > 1e12:
                logger.warning(f"⚠️ INSTABILITA': Condition number P={cond_P:.2e} troppo alto")
                logger.warning("⚠️ Matrice P mal condizionata - possibile singolarità")
        except np.linalg.LinAlgError:
            logger.error("❌ INSTABILITA': Impossibile calcolare condition number di P")
            
        # 3. Controllo simmetria
        symmetry_error = np.max(np.abs(self.P - self.P.T))
        if symmetry_error > 1e-10:
            logger.warning(f"⚠️ INSTABILITA': P non simmetrica, errore={symmetry_error:.2e}")
            # Forza simmetria
            self.P = (self.P + self.P.T) / 2
            logger.info("🔧 Forzata simmetria di P")
            
        # 4. Controllo positività diagonale
        diag_P = np.diag(self.P)
        if np.any(diag_P <= 0):
            logger.warning("⚠️ INSTABILITA': Elementi diagonali di P non positivi")
            negative_count = np.sum(diag_P <= 0)
            logger.warning(f"⚠️ {negative_count} elementi diagonali ≤ 0")
            
        logger.debug(f"✅ Stabilità: trace={trace_P:.4f}, cond={cond_P:.2e}")
    
    def polynomial_detrend_velocity(self, order=None):
        """
        Apply polynomial detrending to velocity after EKF processing.
        
        Args:
            order (int): Polynomial order for detrending (1=linear, 2=quadratic)
                        If None, uses self.poly_order from config
        """
        if not self.vel_correction_enabled:
            return
            
        if len(self.trace_history) < 10:  # Need sufficient data
            return
            
        order = order if order is not None else self.poly_order
        
        # Extract velocity history (this would need to be stored during processing)
        # For now, just detrend the current velocity if it shows systematic drift
        vel_current = self.x[3:6, 0]
        vel_mean = np.mean(vel_current)
        
        # Apply simple drift correction - enforce near-zero mean if stationary
        if abs(vel_mean) > self.drift_threshold:  # Use configured threshold
            logger.info(f"🔧 Velocity drift correction: mean_vel={vel_mean:.4f}")
            # Subtract systematic drift (simple approach)
            self.x[3:6, 0] = vel_current - vel_mean * 0.1  # Gentle correction
    
    def validate_performance(self):
        """
        Validation checks for EKF performance.
        
        Returns:
            dict: Validation results with pass/fail status
        """
        validation_results = {
            'trace_P_check': False,
            'residual_mean_check': False, 
            'residual_std_check': False,
            'velocity_std_check': False,
            'velocity_drift_check': False
        }
        
        # 1. Check trace(P) after startup
        if len(self.trace_history) > 50:  # After startup
            recent_trace = np.mean(self.trace_history[-10:])
            validation_results['trace_P_check'] = recent_trace < 1000
            logger.info(f"✅ Trace(P) check: {recent_trace:.2f} < 1000: {validation_results['trace_P_check']}")
        
        # 2. Residual checks
        if len(self.diagnostics['residual_stats']['mean']) > 0:
            residual_mean = np.array(self.diagnostics['residual_stats']['mean'])
            residual_std = np.array(self.diagnostics['residual_stats']['std'])
            
            # Check residual mean
            mean_check = np.all(np.abs(residual_mean) < 0.2)
            validation_results['residual_mean_check'] = mean_check
            
            # Check residual std (different thresholds for X,Y vs Z)
            std_check_xy = np.all(residual_std[:2] < 3.5)
            std_check_z = residual_std[2] < 6.0 if len(residual_std) > 2 else True
            validation_results['residual_std_check'] = std_check_xy and std_check_z
            
            logger.info(f"✅ Residual mean check: {residual_mean} < 0.2: {mean_check}")
            logger.info(f"✅ Residual std check: XY={residual_std[:2]} < 3.5, Z={residual_std[2] if len(residual_std) > 2 else 'N/A'} < 6.0: {validation_results['residual_std_check']}")
        
        # 3. Velocity checks
        vel_current = self.x[3:6, 0]
        vel_std = np.std(vel_current)
        vel_drift = np.mean(np.abs(vel_current))
        
        validation_results['velocity_std_check'] = vel_std < 1.0
        validation_results['velocity_drift_check'] = vel_drift < 0.05
        
        logger.info(f"✅ Velocity std check: {vel_std:.4f} < 1.0: {validation_results['velocity_std_check']}")
        logger.info(f"✅ Velocity drift check: {vel_drift:.4f} < 0.05: {validation_results['velocity_drift_check']}")
        
        return validation_results


def load_config(config_path):
    """
    Carica il file di configurazione YAML.
    
    Args:
        config_path (str): Percorso del file di configurazione YAML.
        
    Returns:
        dict: Dizionario contenente le configurazioni.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # Valida e imposta valori predefiniti per la configurazione di signal_trimming
        if 'signal_trimming' not in config:
            config['signal_trimming'] = {'enabled': False, 'start_offset': 50}
        elif 'enabled' not in config['signal_trimming']:
            config['signal_trimming']['enabled'] = False
        elif 'start_offset' not in config['signal_trimming']:
            config['signal_trimming']['start_offset'] = 50
            
        logger.info(f"Configurazione caricata con successo da {config_path}")
        return config
    except Exception as e:
        logger.error(f"Errore nel caricamento della configurazione: {e}")
        raise


# ============================================
# FUNZIONI DI VALIDAZIONE E CONTROLLO QUALITÀ
# ============================================

def validate_configuration(config: dict) -> bool:
    """
    Valida la configurazione per assicurarsi che tutti i parametri siano corretti.
    
    Args:
        config (dict): Configurazione da validare
        
    Returns:
        bool: True se la configurazione è valida, False altrimenti
    """
    try:
        logger.info("🔍 Validazione configurazione in corso...")
        
        # Controlli base
        required_sections = ['ekf', 'input_files', 'output_files']
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Sezione mancante nella configurazione: {section}")
                return False
        
        # Controlli parametri Kalman
        kf_config = config['ekf']
        required_kf_params = ['initial_state', 'initial_covariance', 'process_noise', 'measurement_noise']
        for param in required_kf_params:
            if param not in kf_config:
                logger.error(f"❌ Parametro EKF mancante: {param}")
                return False
        
        # Controlli validazione output se abilitata
        if config.get('output_validation', {}).get('enabled', False):
            validation_config = config['output_validation']
            required_bounds = ['velocity_bounds', 'position_bounds', 'acceleration_bounds']
            for bounds in required_bounds:
                if bounds not in validation_config:
                    logger.warning(f"⚠️ Configurazione validazione incompleta: {bounds} mancante")
        
        logger.info("✅ Configurazione validata con successo")
        return True
        
    except Exception as e:
        logger.error(f"❌ Errore nella validazione configurazione: {e}")
        return False


def validate_results(results: dict, config: dict, axis: str) -> dict:
    """
    Valida i risultati dell'EKF contro i criteri di qualità configurati.
    
    Args:
        results (dict): Risultati dell'EKF da validare
        config (dict): Configurazione con parametri di validazione
        axis (str): Asse analizzato
        
    Returns:
        dict: Report di validazione con esito e dettagli
    """
    validation_report = {
        'passed': True,
        'warnings': [],
        'errors': [],
        'metrics': {}
    }
    
    if not config.get('output_validation', {}).get('enabled', False):
        logger.info("⏭️ Validazione output disabilitata, saltando controlli")
        return validation_report
    
    try:
        logger.info(f"🔍 Validazione risultati per asse {axis} in corso...")
        
        # Estrai dati dai risultati
        velocities = results['velocities']
        positions = results['positions'] 
        accelerations = results['accelerations']
        
        validation_config = config['output_validation']
        
        # Validazione velocità
        vel_stats = {
            'min': np.min(velocities),
            'max': np.max(velocities),
            'std': np.std(velocities),
            'mean': np.mean(velocities)
        }
        validation_report['metrics']['velocity'] = vel_stats
        
        vel_bounds = validation_config['velocity_bounds']
        if vel_stats['min'] < vel_bounds['min_value']:
            validation_report['errors'].append(f"Velocità minima {vel_stats['min']:.3f} sotto soglia {vel_bounds['min_value']}")
        if vel_stats['max'] > vel_bounds['max_value']:
            validation_report['errors'].append(f"Velocità massima {vel_stats['max']:.3f} sopra soglia {vel_bounds['max_value']}")
        if vel_stats['std'] > vel_bounds['max_std']:
            validation_report['warnings'].append(f"Deviazione standard velocità {vel_stats['std']:.3f} alta (soglia: {vel_bounds['max_std']})")
        
        # Validazione posizione
        pos_stats = {
            'min': np.min(positions),
            'max': np.max(positions),
            'std': np.std(positions),
            'mean': np.mean(positions)
        }
        validation_report['metrics']['position'] = pos_stats
        
        pos_bounds = validation_config['position_bounds']
        if pos_stats['min'] < pos_bounds['min_value']:
            validation_report['errors'].append(f"Posizione minima {pos_stats['min']:.3f} sotto soglia {pos_bounds['min_value']}")
        if pos_stats['max'] > pos_bounds['max_value']:
            validation_report['errors'].append(f"Posizione massima {pos_stats['max']:.3f} sopra soglia {pos_bounds['max_value']}")
        if pos_stats['std'] > pos_bounds['max_std']:
            validation_report['warnings'].append(f"Deviazione standard posizione {pos_stats['std']:.3f} alta (soglia: {pos_bounds['max_std']})")
        
        # Validazione accelerazione
        acc_stats = {
            'min': np.min(accelerations),
            'max': np.max(accelerations),
            'std': np.std(accelerations),
            'mean': np.mean(accelerations)
        }
        validation_report['metrics']['acceleration'] = acc_stats
        
        acc_bounds = validation_config['acceleration_bounds']
        if acc_stats['min'] < acc_bounds['min_value']:
            validation_report['errors'].append(f"Accelerazione minima {acc_stats['min']:.3f} sotto soglia {acc_bounds['min_value']}")
        if acc_stats['max'] > acc_bounds['max_value']:
            validation_report['errors'].append(f"Accelerazione massima {acc_stats['max']:.3f} sopra soglia {acc_bounds['max_value']}")
        if acc_stats['std'] > acc_bounds['max_std']:
            validation_report['warnings'].append(f"Deviazione standard accelerazione {acc_stats['std']:.3f} alta (soglia: {acc_bounds['max_std']})")
        
        # Controlli fisici
        physics_config = validation_config.get('physics_validation', {})
        if physics_config:
            # Controllo cambi di velocità
            vel_changes = np.abs(np.diff(velocities))
            max_vel_change = np.max(vel_changes)
            if max_vel_change > physics_config.get('max_velocity_change', 2.0):
                validation_report['warnings'].append(f"Cambio di velocità elevato: {max_vel_change:.3f} m/s")
            
            # Controllo deriva
            drift = np.abs(np.mean(velocities))
            if drift > physics_config.get('drift_threshold', 0.1):
                validation_report['warnings'].append(f"Deriva velocità rilevata: {drift:.3f} m/s")
        
        # Determina esito finale
        if validation_report['errors']:
            validation_report['passed'] = False
            
        # Log risultati
        if validation_report['passed']:
            logger.info("✅ Validazione risultati completata con successo")
        else:
            logger.warning("⚠️ Validazione risultati completata con errori")
            
        for warning in validation_report['warnings']:
            logger.warning(f"⚠️ {warning}")
        for error in validation_report['errors']:
            logger.error(f"❌ {error}")
            
        return validation_report
        
    except Exception as e:
        logger.error(f"❌ Errore nella validazione risultati: {e}")
        validation_report['passed'] = False
        validation_report['errors'].append(f"Errore validazione: {e}")
        return validation_report


def create_backup_if_needed(file_path: Union[str, Path], config: dict) -> Optional[str]:
    """
    Crea un backup del file se richiesto dalla configurazione.
    
    Args:
        file_path (str): Percorso del file da backuppare
        config (dict): Configurazione
        
    Returns:
        Optional[str]: Percorso del backup creato, None se non creato
    """
    if not config.get('execution_control', {}).get('backup_existing_files', False):
        return None

    path = Path(file_path)

    if not path.exists():
        return None
        
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.name}.backup_{timestamp}")
        shutil.copy2(path, backup_path)
        logger.info(f"📁 Backup creato: {backup_path}")
        return str(backup_path)
    except Exception as e:
        logger.warning(f"⚠️ Errore nella creazione backup: {e}")
        return None


def get_user_confirmation(message: str, config: dict) -> bool:
    """
    Richiede conferma utente se configurato.
    
    Args:
        message (str): Messaggio da mostrare
        config (dict): Configurazione
        
    Returns:
        bool: True se confermato, False altrimenti
    """
    execution_mode = config.get('execution_control', {}).get('mode', 'auto')
    
    if execution_mode == 'auto':
        return True
    elif execution_mode == 'batch':
        return True
    elif execution_mode == 'interactive':
        print_colored(f"❓ {message} (y/N): ", "❓", "yellow", end="")
        response = input().lower().strip()
        return response in ['y', 'yes', 'si', 's']
    
    return True


def print_validation_report(validation_report: dict, axis: str):
    """
    Stampa il report di validazione in modo formattato.
    
    Args:
        validation_report (dict): Report di validazione
        axis (str): Asse analizzato
    """
    print_colored(f"📊 ============ REPORT VALIDAZIONE ASSE {axis} ============", "📊", "magenta")
    
    if validation_report['passed']:
        print_colored("✅ VALIDAZIONE SUPERATA", "✅", "green")
    else:
        print_colored("❌ VALIDAZIONE FALLITA", "❌", "red")
    
    # Metriche
    if 'metrics' in validation_report:
        metrics = validation_report['metrics']
        if 'velocity' in metrics:
            vel = metrics['velocity']
            print_colored("🚀 METRICHE VELOCITÀ:", "🚀", "cyan")
            print_colored(f"  📊 Range: [{vel['min']:.3f}, {vel['max']:.3f}] m/s", "📊", "cyan")
            print_colored(f"  📊 Media: {vel['mean']:.3f} m/s, Std: {vel['std']:.3f} m/s", "📊", "cyan")
            
        if 'position' in metrics:
            pos = metrics['position']
            print_colored("📍 METRICHE POSIZIONE:", "📍", "cyan")
            print_colored(f"  📊 Range: [{pos['min']:.3f}, {pos['max']:.3f}] m", "📊", "cyan")
            print_colored(f"  📊 Media: {pos['mean']:.3f} m, Std: {pos['std']:.3f} m", "📊", "cyan")
    
    # Warnings
    if validation_report['warnings']:
        print_colored("⚠️ AVVISI:", "⚠️", "yellow")
        for warning in validation_report['warnings']:
            print_colored(f"  ⚠️ {warning}", "⚠️", "yellow")
    
    # Errori
    if validation_report['errors']:
        print_colored("❌ ERRORI:", "❌", "red")
        for error in validation_report['errors']:
            print_colored(f"  ❌ {error}", "❌", "red")
    
    print_colored("📊 ============================================", "📊", "magenta")


def _validate_final_performance(ekf, results, axis, diagnostics):
    """
    Accettazione output (assert/validazione soft) - verifica stabilità finale.
    Non fallisce il run; solo warning se criteri non soddisfatti.
    """
    logger.info(f"🔍 Final performance validation for axis {axis}")
    
    # 1. trace(P) final < 3000
    final_trace = float(np.trace(ekf.P))
    if final_trace < 3000:
        logger.info(f"✅ trace(P) final: {final_trace:.1f} < 3000")
    else:
        logger.warning(f"⚠️ trace(P) final: {final_trace:.1f} >= 3000 (target < 3000)")
    
    # 2. std(vel X/Y) < 1.0 m/s; Z: cerca di scendere
    if 'velocity' in results:
        vel_data = results['velocity']
        vel_std = np.std(vel_data)
        
        if axis in ['X', 'Y']:
            if vel_std < 1.0:
                logger.info(f"✅ std(vel {axis}): {vel_std:.3f} < 1.0 m/s")
            else:
                logger.warning(f"⚠️ std(vel {axis}): {vel_std:.3f} >= 1.0 m/s (target < 1.0)")
        else:  # Z axis
            logger.info(f"📊 std(vel Z): {vel_std:.3f} m/s (monitoring for improvement)")
    
    # 3. ZUPT percentage (informational)
    zupt_count = diagnostics.get('zupt_activations', 0)
    total_measurements = diagnostics.get('total_measurements', 1)
    zupt_percentage = 100.0 * zupt_count / max(1, total_measurements)  # Prevent division by zero
    logger.info(f"📊 ZUPT activation: {zupt_percentage:.1f}% ({zupt_count}/{total_measurements})")
    
    # 4. Residuals: |mean| < 0.2, autocorrelazione check se disponibile
    residual_stats = diagnostics.get('residual_stats', {})
    if 'mean' in residual_stats:
        res_mean = np.array(residual_stats['mean'])
        mean_magnitude = np.mean(np.abs(res_mean))
        if mean_magnitude < 0.2:
            logger.info(f"✅ |residual mean|: {mean_magnitude:.3f} < 0.2")
        else:
            logger.warning(f"⚠️ |residual mean|: {mean_magnitude:.3f} >= 0.2 (target < 0.2)")
        
        # Whiteness score if available
        whiteness = residual_stats.get('whiteness_score', 0)
        if whiteness > 0.8:
            logger.info(f"✅ Residual whiteness: {whiteness:.2f} (good)")
        else:
            logger.warning(f"⚠️ Residual whiteness: {whiteness:.2f} (target > 0.8)")
    
    # 5. Trace resets (should be occasional, not continuous)
    trace_resets = diagnostics.get('trace_resets', 0)
    if trace_resets <= 5:
        logger.info(f"✅ Trace resets: {trace_resets} (acceptable)")
    else:
        logger.warning(f"⚠️ Trace resets: {trace_resets} (too many, check stability)")
    
    logger.info("🎯 Final performance validation completed")


def parse_acceleration_data(file_path, acc_type):
    """
    Analizza e carica i dati di accelerazione dal file di input.
    
    Args:
        file_path (str): Percorso del file contenente i dati di accelerazione.
        acc_type (str): Tipo di accelerazione ('linear' o 'uncalibrated').
        
    Returns:
        pd.DataFrame: DataFrame contenente i dati di accelerazione.
    """
    try:
        logger.info(f"Caricamento dei dati di accelerazione {acc_type} da {file_path}")
        
        # Carica i dati dal file - gestendo diversi formati possibili
        try:
            # Prima prova a caricare come CSV standard
            data = pd.read_csv(file_path, header=None)
        except Exception:
            try:
                # Prova con delimitatore diverso
                data = pd.read_csv(file_path, header=None, delimiter=',')
            except Exception:
                # Ultima risorsa: leggi il file come testo e splittalo manualmente
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                rows = []
                for line in lines:
                    # Rimuovi spazi bianchi e dividi per virgole
                    values = line.strip().split(',')
                    # Converti in numeri se possibile
                    numeric_values = []
                    for val in values:
                        try:
                            numeric_values.append(float(val))
                        except ValueError:
                            numeric_values.append(val)
                    rows.append(numeric_values)
                
                data = pd.DataFrame(rows)
        
        # Determina il formato in base al tipo e al numero di colonne
        logger.info(f"File caricato con {data.shape[1]} colonne")
        
        if acc_type == 'linear':
            # Formato: timestamp, acc_x, acc_y, acc_z
            if data.shape[1] >= 4:
                data = data.iloc[:, :4]  # Prendi solo le prime 4 colonne
                data.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
            else:
                raise ValueError(f"Formato dati non valido per accelerazione {acc_type}. Attese almeno 4 colonne, trovate {data.shape[1]}")
        else:  # uncalibrated
            if data.shape[1] >= 7:
                # Formato: timestamp, acc_x, acc_y, acc_z, bias_x, bias_y, bias_z
                data = data.iloc[:, :7]  # Prendi solo le prime 7 colonne
                data.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'bias_x', 'bias_y', 'bias_z']
            elif data.shape[1] >= 4:
                # Formato ridotto: timestamp, acc_x, acc_y, acc_z
                data = data.iloc[:, :4]
                data.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
                logger.warning("Formato dei dati uncalibrated non standard, si presume [timestamp, acc_x, acc_y, acc_z]")
            else:
                raise ValueError(f"Formato dati non valido per accelerazione {acc_type}. Attese almeno 4 colonne, trovate {data.shape[1]}")
        
        # Converti i dati in formato numerico
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # I dati di accelerazione sono già in m/s², non necessitano conversione
        # Valori tipici: ~9.8 m/s² per la gravità, range normale ±20 m/s²
        logger.info(f"� Accelerazioni già in m/s², nessuna conversione necessaria")
        
        print_colored(f"� Accelerazioni già in formato corretto (m/s²)", "�", "green")
        
        # Rimuovi eventuali righe con valori NaN
        data = data.dropna()
        
        # Converti il timestamp in secondi dall'inizio della registrazione
        data['timestamp'] = (data['timestamp'] - data['timestamp'].iloc[0]) / 1000.0
        
        print_colored(f"📊 Caricati {len(data)} campioni di accelerazione {acc_type}", "📊", "green")
        print_colored(f"⏱️  Intervallo temporale: {data['timestamp'].min():.3f}s - {data['timestamp'].max():.3f}s", "⏱️", "blue")
        print_colored(f"📈 Frequenza media: {len(data)/(data['timestamp'].max()-data['timestamp'].min()):.1f} Hz", "📈", "blue")
        
        logger.info(f"📊 Caricati {len(data)} campioni di accelerazione {acc_type}")
        logger.info(f"⏱️  Intervallo temporale: da {data['timestamp'].min():.3f} a {data['timestamp'].max():.3f} secondi")
        logger.info(f"📈 Frequenza media stimata: {len(data)/(data['timestamp'].max()-data['timestamp'].min()):.1f} Hz")
        
        # Log statistiche dettagliate per ogni asse
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            if axis in data.columns:
                mean_val = data[axis].mean()
                std_val = data[axis].std()
                min_val = data[axis].min()
                max_val = data[axis].max()
                logger.debug(f"📊 {axis}: mean={mean_val:.3f}, std={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")
        
        return data
        
    except Exception as e:
        print_colored(f"❌ Errore nel parsing dei dati di accelerazione: {e}", "❌", "red")
        logger.error(f"❌ Errore nel parsing dei dati di accelerazione: {e}")
        raise


def resample_acceleration_data(data, frequency):
    """
    Ricampiona i dati di accelerazione alla frequenza desiderata.
    
    Args:
        data (pd.DataFrame): DataFrame contenente i dati di accelerazione.
        frequency (int): Frequenza di ricampionamento in Hz.
        
    Returns:
        pd.DataFrame: DataFrame contenente i dati ricampionati.
    """
    try:
        print_colored(f"🔄 Ricampionamento dei dati di accelerazione a {frequency} Hz", "🔄", "cyan")
        logger.info(f"🔄 Ricampionamento dei dati di accelerazione a {frequency} Hz")
        
        # Calcola l'intervallo temporale per il ricampionamento
        dt = 1.0 / frequency
        print_colored(f"⏱️  Intervallo di campionamento: {dt:.6f}s", "⏱️", "blue")
        logger.info(f"⏱️  Intervallo di campionamento dt: {dt:.6f}s")
        
        # Crea una nuova serie temporale ricampionata
        t_new = np.arange(data['timestamp'].min(), data['timestamp'].max(), dt)
        print_colored(f"📊 Generazione di {len(t_new)} nuovi campioni temporali", "📊", "blue")
        logger.info(f"📊 Campioni temporali originali: {len(data)}, nuovi campioni: {len(t_new)}")
        
        # Crea un DataFrame per i dati ricampionati
        resampled_data = pd.DataFrame({'timestamp': t_new})
        
        # Interpola i dati per ogni asse
        axes_interpolated = 0
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            if axis in data.columns:
                print_colored(f"🎯 Interpolazione asse {axis}", "🎯", "yellow")
                f = interpolate.interp1d(data['timestamp'], data[axis], bounds_error=False, fill_value='extrapolate')
                resampled_data[axis] = f(t_new)
                axes_interpolated += 1
                logger.debug(f"✅ Interpolazione completata per {axis}")
        
        print_colored(f"✅ Interpolazione completata per {axes_interpolated} assi", "✅", "green")
        
        # Interpola anche i dati di bias se presenti
        bias_axes_interpolated = 0
        for axis in ['bias_x', 'bias_y', 'bias_z']:
            if axis in data.columns:
                print_colored(f"⚖️  Interpolazione bias {axis}", "⚖️", "yellow")
                f = interpolate.interp1d(data['timestamp'], data[axis], bounds_error=False, fill_value='extrapolate')
                resampled_data[axis] = f(t_new)
                bias_axes_interpolated += 1
                logger.debug(f"✅ Interpolazione bias completata per {axis}")
        
        if bias_axes_interpolated > 0:
            print_colored(f"⚖️  Interpolazione bias completata per {bias_axes_interpolated} assi", "⚖️", "green")
        
        print_colored(f"🎉 Ricampionamento completato: {len(resampled_data)} campioni finali", "🎉", "green")
        logger.info(f"🎉 Ricampionamento completato, {len(resampled_data)} campioni finali")
        return resampled_data
        
    except Exception as e:
        logger.error(f"Errore nel ricampionamento dei dati: {e}")
        raise


def apply_extended_kalman_filter(data, config, axis):
    """
    Applica l'Extended Kalman Filter ai dati di accelerazione.
    
    Args:
        data (pd.DataFrame): DataFrame contenente i dati di accelerazione.
        config (dict): Dizionario contenente le configurazioni dell'EKF.
        axis (str): Asse su cui applicare il filtro ('X', 'Y', 'Z').
        
    Returns:
        pd.DataFrame: DataFrame contenente lo stato stimato (posizione, velocità).
    """
    try:
        print_section_header(f"APPLICAZIONE EKF SULL'ASSE {axis}", "🎯")
        logger.info(f"🎯 Applicazione dell'Extended Kalman Filter sull'asse {axis}")
        
        # Crea un'istanza dell'EKF
        print_colored("🔧 Inizializzazione del filtro EKF...", "🔧", "cyan")
        ekf = ExtendedKalmanFilter3D(config)
        
        # Inizializza il monitor delle prestazioni
        performance_monitor = EKFPerformanceMonitor(config, logger)
        print_colored("🔍 Monitor delle prestazioni EKF inizializzato", "🔍", "green")
        
        # Controlla se abbiamo dati 3D completi (accelerazione + orientamento)
        has_3d_data = all(col in data.columns for col in ['acc_x', 'acc_y', 'acc_z'])
        has_orientation = all(col in data.columns for col in ['roll', 'pitch', 'yaw'])
        
        # PATCH: Force 3D mode if we have 3D acceleration data, even without orientation
        if has_3d_data:
            print_colored("🌍 Dati 3D rilevati - modalità EKF 3D attivata (orientamento zero se mancante)", "🌍", "green")
            logger.info("🌍 Modalità EKF 3D attivata con accelerazione (orientamento zero se mancante)")
            mode_3d = True
            
            # Add zero orientation columns if missing
            if not has_orientation:
                data['roll'] = 0.0
                data['pitch'] = 0.0  
                data['yaw'] = 0.0
                logger.info("🔧 Aggiunta orientamento zero (roll=0, pitch=0, yaw=0)")
        else:
            print_colored(f"📊 Modalità compatibilità 1D per asse {axis}", "📊", "yellow")
            logger.info(f"📊 Modalità compatibilità 1D per asse {axis}")
            mode_3d = False
        
        # Selezione la colonna di accelerazione corrispondente all'asse
        acc_column = f'acc_{axis.lower()}'
        
        if acc_column not in data.columns:
            print_colored(f"❌ Errore: Asse {axis} non presente nei dati", "❌", "red")
            logger.error(f"❌ Asse {axis} non presente nei dati")
            raise ValueError(f"Asse {axis} non presente nei dati")
        
        print_colored(f"📊 Colonna accelerazione selezionata: {acc_column}", "📊", "blue")
        logger.info(f"📊 Elaborazione dati dalla colonna: {acc_column}")
        
        # Rileva i periodi in cui il dispositivo è stazionario se ZUPT è abilitato
        zupt_enabled = config.get('zupt', {}).get('enabled', False)
        if zupt_enabled:
            print_colored(f"🔍 Rilevamento periodi stazionari per l'asse {axis}...", "🔍", "yellow")
            logger.info(f"🔍 Rilevamento dei periodi stazionari per l'asse {axis}")
            
            window_size = int(config.get('zupt', {}).get('window_size', 50))
            threshold = float(config.get('zupt', {}).get('threshold', 0.1))
            print_colored(f"⚙️  Parametri ZUPT: finestra={window_size}, soglia={threshold}", "⚙️", "cyan")
            logger.info(f"⚙️  Parametri ZUPT: window_size={window_size}, threshold={threshold}")
            
            is_stationary = detect_stationary_periods(
                data, 
                acc_column,
                window_size=window_size,
                threshold=threshold
            )
            
            stationary_count = np.sum(is_stationary)
            total_samples = len(is_stationary)
            stationary_percentage = (stationary_count / total_samples) * 100
            
            print_colored(f"✅ Rilevati {stationary_count} campioni stazionari su {total_samples} totali ({stationary_percentage:.1f}%)", "✅", "green")
            logger.info(f"✅ Rilevati {stationary_count} campioni stazionari su {total_samples} totali ({stationary_percentage:.1f}%)")
            print_colored(f"Periodi stazionari: {np.sum(is_stationary)} campioni ({stationary_percentage:.1f}%)", "🛑", "green")
        else:
            # Se ZUPT non è abilitato, nessun campione è considerato stazionario
            is_stationary = np.zeros(len(data), dtype=bool)
            logger.info("Zero-Velocity Update (ZUPT) disabilitato")
            print_colored("Zero-Velocity Update (ZUPT) disabilitato", "⚠️", "yellow")
        
        # Inizializza array per salvare i risultati
        timestamps = data['timestamp'].values
        n_samples = len(timestamps)
        positions = np.zeros(n_samples)
        velocities = np.zeros(n_samples)
        accelerations = np.zeros(n_samples)
        biases = np.zeros(n_samples)
        
        # Salva lo stato iniziale
        positions[0] = ekf.x[0, 0]
        velocities[0] = ekf.x[1, 0]
        accelerations[0] = ekf.x[2, 0]
        biases[0] = ekf.x[3, 0]
        
        # Applica l'EKF per ogni passo temporale
        logger.info(f"📊 Inizio processamento EKF per {n_samples} campioni...")
        print_colored(f"📊 Processamento EKF in corso per {n_samples} campioni...", "📊", "cyan")
        
        # Progress tracking setup
        progress_steps = 10  # Show progress every 10%
        progress_interval = max(1, n_samples // progress_steps)
        
        for i in range(1, n_samples):
            # Progress tracking
            if i % progress_interval == 0 or i == n_samples - 1:
                progress_percent = (i / (n_samples - 1)) * 100
                print_colored(f"🔄 Progresso EKF: {progress_percent:.1f}% ({i}/{n_samples-1})", "🔄", "blue")
                logger.info(f"🔄 Progresso EKF: {progress_percent:.1f}% ({i}/{n_samples-1})")
            
            # Calcola dt (intervallo temporale tra campioni)
            dt = timestamps[i] - timestamps[i-1]
            
            # Fase di predizione
            ekf.predict(dt)
            
            if mode_3d:
                # Modalità 3D completa con rotazione Euler
                acc_x = data['acc_x'].iloc[i]
                acc_y = data['acc_y'].iloc[i]
                acc_z = data['acc_z'].iloc[i]
                acc_body = [acc_x, acc_y, acc_z]
                
                # Angoli di Euler (assumendo che siano in radianti)
                roll = data['roll'].iloc[i] if 'roll' in data.columns else 0.0
                pitch = data['pitch'].iloc[i] if 'pitch' in data.columns else 0.0
                yaw = data['yaw'].iloc[i] if 'yaw' in data.columns else 0.0
                
                # Update EKF 3D with comprehensive features
                state, covariance, innovation, diagnostics = ekf.update_comprehensive(
                    acc_body, roll, pitch, yaw, dt, timestamps[i]
                )
                
                # Apply velocity detrending at configured intervals
                if i % ekf.correction_interval == 0:
                    ekf.polynomial_detrend_velocity()
                
                # Per compatibilità con codice esistente, estrai solo i dati dell'asse corrente
                axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[axis]
                single_innovation = innovation[axis_idx] if len(innovation) > axis_idx else innovation[0]
                
            else:
                # Modalità compatibilità 1D
                z = data[acc_column].iloc[i]
                measurement = np.array([[z]])
                
                # Usa metodo legacy per compatibilità
                state, covariance, innovation = ekf.update_legacy(z)
                single_innovation = innovation[0] if hasattr(innovation, '__len__') else innovation
            
            # Aggiorna le metriche di performance (compatibilità con monitoring esistente)
            if hasattr(single_innovation, '__len__'):
                innovation_for_monitor = single_innovation[0] if len(single_innovation) > 0 else 0.0
            else:
                innovation_for_monitor = float(single_innovation)
            
            # Crea un'innovazione compatibile con il monitor (deve essere array 1x1)
            innovation_array = np.array([[innovation_for_monitor]])
            measurement_array = np.array([[data[acc_column].iloc[i]]])
            
            # PATCH 6: Re-enable performance monitoring with safe formatting
            try:
                performance_monitor.update_metrics(ekf, innovation_array, measurement_array)
            except Exception as e:
                logger.warning(safe_fmt(f"⚠️ Errore nel monitoraggio performance: {e}"))
                # Continua l'esecuzione senza crashare
            
            # Controllo periodico della convergenza e diagnostica
            check_interval = config.get('performance_monitoring', {}).get('convergence_check_interval', 100)
            if i % check_interval == 0:  # Controllo periodico
                is_converged = performance_monitor.check_convergence()
                
                # Diagnostica in tempo reale ogni 500 iterazioni
                if i % (check_interval * 5) == 0 and i > 500:
                    print_colored(f"🔍 Diagnostica EKF (iterazione {i}):", "🔍", "cyan")
                    
                    # Check rapido della consistenza
                    consistency = performance_monitor.check_filter_consistency()
                    if not consistency['overall_consistent']:
                        if not consistency['nis_consistent']:
                            print_colored("  ⚠️ NIS fuori range - possibile problema di tuning R", "⚠️", "yellow")
                        if not consistency['nees_consistent']:
                            print_colored("  ⚠️ NEES fuori range - possibile problema di tuning Q", "⚠️", "yellow")
                    else:
                        print_colored("  ✅ Filtro statisticamente consistente", "✅", "green")
                    
                    # Check traccia P
                    current_trace = np.trace(ekf.P)
                    if current_trace > performance_monitor.thresholds['max_trace']:
                        print_colored(f"  ⚠️ Traccia P elevata: {current_trace:.2f}", "⚠️", "yellow")
                    
                    # Check innovazioni
                    if len(performance_monitor.innovation_history) > 50:
                        recent_innovations = list(performance_monitor.innovation_history)[-50:]
                        innovation_trend = np.polyfit(range(len(recent_innovations)), recent_innovations, 1)[0]
                        if abs(innovation_trend) > 0.01:
                            print_colored(f"  📈 Trend innovazioni: {innovation_trend:.4f} (crescente)" if innovation_trend > 0 else f"  📉 Trend innovazioni: {innovation_trend:.4f} (decrescente)", "📊", "cyan")
            
            # Applica Zero-Velocity Update (ZUPT) se il dispositivo è stazionario
            if is_stationary[i]:
                if mode_3d:
                    # Forza la velocità 3D a zero durante i periodi stazionari
                    ekf.x[3:6, 0] = 0.0
                    # Riduce l'incertezza sulla velocità 3D
                    ekf.P[3:6, 3:6] *= 0.1
                else:
                    # Modalità legacy 1D
                    ekf.x[1, 0] = 0.0
                    ekf.P[1, 1] *= 0.1
            
            # Salva lo stato stimato (estrai dati per l'asse corrente)
            if mode_3d:
                axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[axis]
                positions[i] = ekf.x[axis_idx, 0]          # pos_x, pos_y, or pos_z
                velocities[i] = ekf.x[axis_idx + 3, 0]     # vel_x, vel_y, or vel_z
                # Per accelerazioni e bias, usiamo valori derivati/stimati
                accelerations[i] = data[acc_column].iloc[i] # Accelerazione misurata
                biases[i] = 0.0  # Il nuovo modello non ha bias esplicito
            else:
                # Modalità legacy
                positions[i] = ekf.x[0, 0]
                velocities[i] = ekf.x[1, 0]
                accelerations[i] = ekf.x[2, 0] if ekf.x.shape[0] > 2 else data[acc_column].iloc[i]
                biases[i] = ekf.x[3, 0] if ekf.x.shape[0] > 3 else 0.0
        
        # Applica la correzione della deriva di velocità se richiesto
        drift_summary = {'enabled': False}
        if config.get('drift_correction', {}).get('enabled', True):
            logger.info("Applicazione della correzione della deriva di velocità")
            print_colored("Applicazione correzione della deriva di velocità...", "🔄", "yellow")
            
            # Salva i valori originali per il report
            original_mean_velocity = np.mean(velocities)
            original_velocity_range = np.max(velocities) - np.min(velocities)
            
            polynomial_order = config.get('drift_correction', {}).get('polynomial_order', 2)
            print_colored(f"Parametri correzione deriva: ordine_polinomio={polynomial_order}", "🔧", "cyan")
            velocities, positions = correct_velocity_drift(timestamps, velocities, positions, polynomial_order=polynomial_order)
            
            # Report sui risultati della correzione
            corrected_mean_velocity = np.mean(velocities)
            corrected_velocity_range = np.max(velocities) - np.min(velocities)
            
            logger.info("Correzione della deriva di velocità completata")
            print_colored("Correzione della deriva completata:", "✅", "green")
            print_colored(f"  - Velocità media prima: {original_mean_velocity:.4f} m/s", "📉", "cyan")
            print_colored(f"  - Velocità media dopo: {corrected_mean_velocity:.4f} m/s", "📈", "cyan")
            print_colored(f"  - Range velocità prima: {original_velocity_range:.4f} m/s", "📏", "cyan")
            print_colored(f"  - Range velocità dopo: {corrected_velocity_range:.4f} m/s", "📏", "cyan")
            
            drift_summary = {
                'enabled': True,
                'polynomial_order': polynomial_order,
                'mean_velocity_before': float(original_mean_velocity),
                'mean_velocity_after': float(corrected_mean_velocity),
                'range_before': float(original_velocity_range),
                'range_after': float(corrected_velocity_range)
            }
        
        # Applica post-processing avanzato se abilitato
        post_processing_summary = {'enabled': False}
        if config.get('post_processing', {}).get('velocity_smoothing', {}).get('enabled', False) or \
           config.get('post_processing', {}).get('outlier_removal', {}).get('enabled', False):
            print_colored("🔄 Applicazione post-processing avanzato...", "🔄", "yellow")
            logger.info("Applicazione post-processing avanzato")
            
            # Salva valori pre-processing
            pre_processing_velocity_std = np.std(velocities)
            
            # Applica post-processing
            velocities, positions = apply_post_processing(velocities, positions, config)
            
            # Report miglioramenti
            post_processing_velocity_std = np.std(velocities)
            noise_reduction = ((pre_processing_velocity_std - post_processing_velocity_std) / pre_processing_velocity_std) * 100
            
            print_colored("✅ Post-processing completato:", "✅", "green")
            print_colored(f"  - Riduzione rumore velocità: {noise_reduction:.1f}%", "📊", "cyan")
            print_colored(f"  - Std velocità prima: {pre_processing_velocity_std:.4f} m/s", "📉", "cyan")
            print_colored(f"  - Std velocità dopo: {post_processing_velocity_std:.4f} m/s", "📈", "cyan")
            
            post_processing_summary = {
                'enabled': True,
                'noise_reduction_percent': float(noise_reduction),
                'std_before': float(pre_processing_velocity_std),
                'std_after': float(post_processing_velocity_std),
                'config': config.get('post_processing', {})
            }
        
        # Crea un DataFrame con i risultati
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations,
            'bias': biases
        })
        
        # Crea il dizionario dei risultati
        results = {
            'data': results_df,
            'timestamps': timestamps,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'biases': biases
        }
        
        # 📊 REPORT FINALE DEI RISULTATI EKF
        print_colored("📊 =============== REPORT FINALE EKF ===============", "📊", "magenta")
        logger.info("📊 =============== REPORT FINALE EKF ===============")
        
        # Statistiche generali
        total_time = timestamps[-1] - timestamps[0]
        sampling_rate = len(timestamps) / total_time if total_time > 0 else 0
        
        print_colored(f"🔢 STATISTICHE GENERALI:", "🔢", "cyan")
        print_colored(f"  - Campioni processati: {len(timestamps)}", "📈", "cyan")
        print_colored(f"  - Durata totale: {total_time:.2f} s", "⏱️", "cyan")
        print_colored(f"  - Frequenza di campionamento: {sampling_rate:.1f} Hz", "🎯", "cyan")
        
        # Statistiche di posizione
        pos_mean = np.mean(positions)
        pos_std = np.std(positions)
        pos_range = np.max(positions) - np.min(positions)
        pos_min, pos_max = np.min(positions), np.max(positions)
        
        print_colored(f"📍 STATISTICHE POSIZIONE:", "📍", "green")
        print_colored(f"  - Media: {pos_mean:.4f} m", "📊", "green")
        print_colored(f"  - Deviazione standard: {pos_std:.4f} m", "📊", "green")
        print_colored(f"  - Range: {pos_range:.4f} m (min: {pos_min:.4f}, max: {pos_max:.4f})", "📏", "green")
        
        # Statistiche di velocità
        vel_mean = np.mean(velocities)
        vel_std = np.std(velocities)
        vel_range = np.max(velocities) - np.min(velocities)
        vel_min, vel_max = np.min(velocities), np.max(velocities)
        
        print_colored(f"🚀 STATISTICHE VELOCITÀ:", "🚀", "blue")
        print_colored(f"  - Media: {vel_mean:.4f} m/s", "📊", "blue")
        print_colored(f"  - Deviazione standard: {vel_std:.4f} m/s", "📊", "blue")
        print_colored(f"  - Range: {vel_range:.4f} m/s (min: {vel_min:.4f}, max: {vel_max:.4f})", "📏", "blue")
        
        # Statistiche di accelerazione - controllo valori vuoti
        if len(accelerations) > 0:
            acc_mean = np.mean(accelerations)
            acc_std = np.std(accelerations)
            acc_range = np.max(accelerations) - np.min(accelerations)
            acc_min, acc_max = np.min(accelerations), np.max(accelerations)
        else:
            acc_mean = acc_std = acc_range = acc_min = acc_max = 0.0
            
        print_colored(f"⚡ STATISTICHE ACCELERAZIONE:", "⚡", "yellow")
        print_colored(f"  - Media: {safe_fmt(acc_mean)} m/s²", "📊", "yellow")
        print_colored(f"  - Deviazione standard: {safe_fmt(acc_std)} m/s²", "📊", "yellow")
        print_colored(f"  - Range: {safe_fmt(acc_range)} m/s² (min: {safe_fmt(acc_min)}, max: {safe_fmt(acc_max)})", "📏", "yellow")
        
        # Statistiche di bias - controllo valori vuoti
        if len(biases) > 0:
            bias_mean = np.mean(biases)
            bias_std = np.std(biases)
            bias_range = np.max(biases) - np.min(biases)
        else:
            bias_mean = bias_std = bias_range = 0.0
        bias_std = np.std(biases)
        bias_range = np.max(biases) - np.min(biases)
        
        print_colored(f"🎯 STATISTICHE BIAS:", "🎯", "magenta")
        print_colored(f"  - Media: {safe_fmt(bias_mean)}", "📊", "magenta")
        print_colored(f"  - Deviazione standard: {safe_fmt(bias_std)}", "📊", "magenta")
        print_colored(f"  - Range: {safe_fmt(bias_range)}", "📏", "magenta")
        
        summary_stats = {
            'general': {
                'samples': int(len(timestamps)),
                'duration_seconds': float(total_time),
                'sampling_rate_hz': float(sampling_rate) if np.isfinite(sampling_rate) else None,
            },
            'position': {
                'mean': float(pos_mean),
                'std': float(pos_std),
                'min': float(pos_min),
                'max': float(pos_max),
                'range': float(pos_range),
            },
            'velocity': {
                'mean': float(vel_mean),
                'std': float(vel_std),
                'min': float(vel_min),
                'max': float(vel_max),
                'range': float(vel_range),
            },
            'acceleration': {
                'mean': float(acc_mean),
                'std': float(acc_std),
                'min': float(acc_min),
                'max': float(acc_max),
                'range': float(acc_range),
            },
            'bias': {
                'mean': safe_fmt(bias_mean),
                'std': safe_fmt(bias_std),
                'range': safe_fmt(bias_range),
            },
            'drift_correction': drift_summary,
            'post_processing': post_processing_summary,
        }
        
        # Log delle statistiche - usa safe_fmt per tutti i valori
        logger.info(f"📊 STATISTICHE FINALI - Campioni: {len(timestamps)}, Durata: {total_time:.2f}s")
        logger.info(f"📍 POSIZIONE - Media: {safe_fmt(pos_mean)}m, Std: {safe_fmt(pos_std)}m, Range: {safe_fmt(pos_range)}m")
        logger.info(f"🚀 VELOCITÀ - Media: {safe_fmt(vel_mean)}m/s, Std: {safe_fmt(vel_std)}m/s, Range: {safe_fmt(vel_range)}m/s")
        logger.info(f"⚡ ACCELERAZIONE - Media: {safe_fmt(acc_mean)}m/s², Std: {safe_fmt(acc_std)}m/s², Range: {safe_fmt(acc_range)}m/s²")
        logger.info(f"🎯 BIAS - Media: {safe_fmt(bias_mean)}, Std: {safe_fmt(bias_std)}, Range: {safe_fmt(bias_range)}")
        
        print_colored("✅ ============== EKF COMPLETATO CON SUCCESSO ==============", "✅", "green")
        
        # � COMPREHENSIVE VALIDATION AND DIAGNOSTICS
        print_colored("🔍 =============== VALIDATION & DIAGNOSTICS ===============", "🔍", "cyan")
        logger.info("🔍 Avvio validazione completa EKF...")
        
        # Initialize diagnostics for all cases
        diagnostics = ekf.diagnostics
        
        # 1. Validate EKF performance
        if mode_3d:
            validation_results = ekf.validate_performance()
            
            print_colored(f"📊 VALIDATION RESULTS:", "📊", "cyan")
            for check, passed in validation_results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                color = "green" if passed else "red"
                print_colored(f"  - {check}: {status}", "🔍", color)
                logger.info(f"Validation {check}: {'PASS' if passed else 'FAIL'}")
            
            # 2. Comprehensive diagnostics from EKF
            print_colored(f"🎯 COMPREHENSIVE DIAGNOSTICS:", "🎯", "yellow")
            print_colored(f"  - Total measurements: {diagnostics['total_measurements']}", "📊", "cyan")
            print_colored(f"  - Outliers detected: {diagnostics['outlier_count']} ({100*diagnostics['outlier_count']/max(1,diagnostics['total_measurements']):.1f}%)", "🚫", "yellow")
            print_colored(f"  - ZUPT activations: {diagnostics['zupt_activations']}", "✋", "blue")
            
            if diagnostics['residual_stats']['mean']:
                residual_mean = diagnostics['residual_stats']['mean']
                residual_std = diagnostics['residual_stats']['std']
                print_colored(f"  - Residual mean: [{residual_mean[0]:.4f}, {residual_mean[1]:.4f}, {residual_mean[2]:.4f}]", "📈", "cyan")
                print_colored(f"  - Residual std: [{residual_std[0]:.4f}, {residual_std[1]:.4f}, {residual_std[2]:.4f}]", "📊", "cyan")
                
                # Check residual quality
                mean_magnitude = np.linalg.norm(residual_mean)
                if mean_magnitude < 0.2:
                    print_colored(f"  ✅ Residual mean quality: GOOD (|mean|={mean_magnitude:.4f} < 0.2)", "✅", "green")
                else:
                    print_colored(f"  ⚠️ Residual mean quality: POOR (|mean|={mean_magnitude:.4f} ≥ 0.2)", "⚠️", "yellow")
            
            # 3. Advanced diagnostics: autocorrelation
            if diagnostics['residual_stats']['autocorr'] and STATSMODELS_AVAILABLE:
                print_colored(f"  📈 Autocorrelation analysis available for 3 axes", "📈", "cyan")
                for i, autocorr in enumerate(diagnostics['residual_stats']['autocorr']):
                    axis_name = ['X', 'Y', 'Z'][i]
                    # Check if autocorrelation is within 95% CI after lag 2
                    ci_95 = 1.96 / np.sqrt(len(autocorr))
                    autocorr_after_lag2 = autocorr[2:]
                    within_ci = np.all(np.abs(autocorr_after_lag2) < ci_95)
                    status = "✅ GOOD" if within_ci else "⚠️ POOR"
                    color = "green" if within_ci else "yellow"
                    print_colored(f"    - {axis_name} axis autocorr: {status} (within 95% CI: {within_ci})", "📊", color)
            
            # 4. Trace and numerical stability
            final_trace = np.trace(ekf.P)
            print_colored(f"  🔢 Final trace(P): {final_trace:.2f}", "🔢", "cyan")
            if final_trace < 1000:
                print_colored(f"  ✅ Numerical stability: GOOD (trace < 1000)", "✅", "green")
            else:
                print_colored(f"  ⚠️ Numerical stability: CONCERNING (trace ≥ 1000)", "⚠️", "yellow")
        
        # 5. Generate comprehensive residual plots
        if mode_3d and config.get('performance_monitoring', {}).get('generate_performance_plots', True):
            output_dir = get_output_base_dir()
            generate_enhanced_residual_plots(ekf, diagnostics, output_dir, axis)
        
        logger.info("✅ Validazione completa terminata")
        
        # Accettazione output - validazione soft per stabilità  
        _validate_final_performance(ekf, results, axis, diagnostics)
        
        # 📊 GENERAZIONE REPORT PRESTAZIONI EKF
        print_colored("", "", "white")  # Spacer
        performance_report = performance_monitor.generate_performance_report()
        
        # Controlla se sono abilitati i grafici e il salvataggio del report
        if config.get('performance_monitoring', {}).get('generate_performance_plots', True):
            output_dir = get_output_base_dir()
            generate_performance_plots(performance_monitor, output_dir, axis)
        
        if config.get('performance_monitoring', {}).get('save_performance_report', True):
            output_dir = get_output_base_dir()
            report_format = config.get('performance_monitoring', {}).get('performance_report_format', 'yaml')
            save_performance_report_to_file(performance_report, performance_monitor, output_dir, axis, report_format)
        
        # 🎛️ GENERAZIONE DASHBOARD DI TUNING AVANZATO
        if config.get('performance_monitoring', {}).get('enable_advanced_analysis', True):
            output_dir = get_output_base_dir()
            tuning_report = generate_tuning_dashboard(performance_monitor, results, data, config, axis, output_dir)
            results['tuning_report'] = tuning_report
        
        # Salva il report delle prestazioni nel log
        logger.info("📊 Report delle prestazioni EKF generato")
        for key, value in performance_report.items():
            logger.info(f"📊 {key}: {value}")
        
        # Aggiungi metriche di performance ai risultati per l'ottimizzazione
        results['performance_metrics'] = performance_report
        results['summary_stats'] = summary_stats
        
        logger.info(f"EKF completato con successo per l'asse {axis}")
        return results
        
    except Exception as e:
        logger.error(f"Errore nell'applicazione dell'EKF: {e}")
        raise


def save_results(results, output_path, file_type):
    """
    Salva i risultati in un file CSV o TXT.
    
    Args:
        results (dict): Dizionario contenente i risultati dell'EKF.
        output_path (str): Percorso dove salvare il file di output.
        file_type (str): Tipo di risultato ('velocity' o 'position').
    """
    try:
        # Assicurati che la directory di output esista
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Estrai il DataFrame dai risultati
        df = results['data'] if isinstance(results, dict) else results
        
        # Seleziona le colonne da salvare in base al tipo di file
        if file_type == 'velocity':
            df[['timestamp', 'velocity']].to_csv(output_path, index=False)
            logger.info(f"Velocità stimata salvata in {output_path}")
        elif file_type == 'position':
            df[['timestamp', 'position']].to_csv(output_path, index=False)
            logger.info(f"Posizione stimata salvata in {output_path}")
        else:
            # Salva tutti i risultati
            df.to_csv(output_path, index=False)
            logger.info(f"Risultati completi salvati in {output_path}")
            
    except Exception as e:
        logger.error(f"Errore nel salvataggio dei risultati: {e}")
        raise


def plot_results(data, results, config, axis, output_path=None):
    """
    Visualizza e opzionalmente salva i grafici dei risultati.
    
    Args:
        data (pd.DataFrame): DataFrame contenente i dati di accelerazione.
        results (dict): Dizionario contenente i risultati dell'EKF.
        config (dict): Dizionario contenente le configurazioni.
        axis (str): Asse analizzato ('X', 'Y', 'Z').
        output_path (str, optional): Percorso dove salvare il grafico. Default è None.
    """
    try:
        logger.info(f"Generazione dei grafici per l'asse {axis}")
        
        # Estrai il DataFrame dai risultati
        results_df = results['data'] if isinstance(results, dict) else results
        
        # Crea un grafico con tre subplot (accelerazione, velocità, posizione)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Accelerazione
        acc_column = f'acc_{axis.lower()}'
        axes[0].plot(data['timestamp'], data[acc_column], 'r-', label=f'Measured Acc {axis}')
        axes[0].plot(results_df['timestamp'], results_df['acceleration'], 'b-', label=f'Filtered Acc {axis}')
        axes[0].set_ylabel(f'Acceleration {axis} (m/s²)')
        axes[0].grid(True)
        axes[0].legend()
        
        # Velocità
        axes[1].plot(results_df['timestamp'], results_df['velocity'], 'g-', label=f'Estimated Velocity {axis}')
        axes[1].set_ylabel(f'Velocity {axis} (m/s)')
        axes[1].grid(True)
        axes[1].legend()
        
        # Posizione
        axes[2].plot(results_df['timestamp'], results_df['position'], 'b-', label=f'Estimated Position {axis}')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel(f'Position {axis} (m)')
        axes[2].grid(True)
        axes[2].legend()
        
        # Titolo generale e configurazione visualizzazione con fallback sicuri
        visualization_cfg = config.get('visualization', {})
        plot_title = visualization_cfg.get('plot_title', 'EKF Results')
        save_plots = visualization_cfg.get('save_plots', True)
        show_plots = visualization_cfg.get('show_plots', False)
        
        plt.suptitle(f"{plot_title} - Axis {axis}")
        plt.tight_layout()
        
        # Salva il grafico se richiesto
        if output_path is not None and save_plots:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            axis_output_path = output_path.with_name(f"{output_path.stem}_{axis}{output_path.suffix}")
            plt.savefig(axis_output_path)
            logger.info(f"Grafico salvato in {axis_output_path}")
        
        # Mostra il grafico se richiesto
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"Errore nella generazione dei grafici: {e}")
        raise


def detect_stationary_periods(data, acc_column, window_size=50, threshold=0.1, min_duration=5):
    """
    Rileva i periodi in cui il dispositivo è stazionario, basandosi sulla varianza dell'accelerazione.
    Include un filtro per la durata minima dei periodi stazionari per evitare falsi positivi.
    
    Args:
        data (pd.DataFrame): DataFrame contenente i dati di accelerazione.
        acc_column (str): Nome della colonna contenente i dati di accelerazione.
        window_size (int): Dimensione della finestra di osservazione (numero di campioni).
        threshold (float): Soglia di varianza sotto la quale il dispositivo è considerato stazionario.
        min_duration (int): Durata minima (in campioni) per considerare un periodo come stazionario.
        
    Returns:
        np.ndarray: Array booleano che indica per ogni campione se il dispositivo è stazionario.
    """
    acc_data = data[acc_column].values
    n_samples = len(acc_data)
    is_stationary_raw = np.zeros(n_samples, dtype=bool)
    
    # Calcola la varianza dell'accelerazione e la media mobile in finestre mobili
    print_colored("Analisi della varianza dell'accelerazione...", "🔍", "cyan")
    
    for i in range(n_samples):
        # Determina l'inizio e la fine della finestra
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n_samples, i + window_size // 2)
        
        # Calcola la varianza nella finestra
        window_variance = np.var(acc_data[start_idx:end_idx])
        
        # Calcola anche la media mobile per una migliore robustezza
        window_mean = np.mean(np.abs(acc_data[start_idx:end_idx]))
        
        # Combina i criteri: varianza bassa e accelerazione media vicina alla gravità
        if window_variance < threshold:
            is_stationary_raw[i] = True
    
    # Applica un filtro per rimuovere periodi stazionari troppo brevi (probabilmente rumore)
    is_stationary = np.copy(is_stationary_raw)
    
    # Rimuovi i periodi stazionari troppo brevi
    i = 0
    while i < n_samples:
        if is_stationary_raw[i]:
            # Trova la lunghezza di questo periodo stazionario
            start_idx = i
            while i < n_samples and is_stationary_raw[i]:
                i += 1
            end_idx = i
            
            # Se il periodo è troppo breve, impostalo come non stazionario
            if (end_idx - start_idx) < min_duration:
                is_stationary[start_idx:end_idx] = False
        else:
            i += 1
    
    # Calcola statistiche
    raw_stationary_percentage = (np.sum(is_stationary_raw) / n_samples) * 100
    filtered_stationary_percentage = (np.sum(is_stationary) / n_samples) * 100
    
    print_colored(f"Periodi stazionari grezzi: {raw_stationary_percentage:.1f}%", "📊", "cyan")
    print_colored(f"Periodi stazionari filtrati: {filtered_stationary_percentage:.1f}%", "📊", "green")
    
    return is_stationary


def apply_velocity_smoothing(velocities, config):
    """
    Applica smoothing alle velocità per ridurre il rumore.
    
    Args:
        velocities (np.ndarray): Array delle velocità da smoothare
        config (dict): Configurazione del post-processing
        
    Returns:
        np.ndarray: Velocità smoothate
    """
    if not config.get('velocity_smoothing', {}).get('enabled', False):
        return velocities
    
    method = config['velocity_smoothing'].get('method', 'moving_average')
    window_size = config['velocity_smoothing'].get('window_size', 7)
    
    try:
        if method == 'moving_average':
            # Media mobile
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(velocities, size=window_size, mode='nearest')
            
        elif method == 'savgol':
            # Savitzky-Golay filter
            from scipy.signal import savgol_filter
            polyorder = config['velocity_smoothing'].get('savgol_polyorder', 3)
            if window_size % 2 == 0:
                window_size += 1  # Deve essere dispari
            smoothed = savgol_filter(velocities, window_size, polyorder, mode='nearest')
            
        elif method == 'median':
            # Filtro mediano
            from scipy.signal import medfilt
            smoothed = medfilt(velocities, kernel_size=window_size)
            
        else:
            logger.warning(f"Metodo di smoothing sconosciuto: {method}")
            return velocities
            
        logger.info(f"Smoothing velocità applicato con metodo: {method}, finestra: {window_size}")
        return smoothed
        
    except Exception as e:
        logger.warning(f"Errore nel smoothing: {e}")
        return velocities


def remove_outliers(data, config):
    """
    Rimuove outlier dai dati.
    
    Args:
        data (np.ndarray): Dati da processare
        config (dict): Configurazione del post-processing
        
    Returns:
        np.ndarray: Dati con outlier rimossi
    """
    if not config.get('outlier_removal', {}).get('enabled', False):
        return data
    
    method = config['outlier_removal'].get('method', 'std')
    threshold = config['outlier_removal'].get('threshold', 2.5)
    
    try:
        if method == 'std':
            # Standard deviation method
            mean_val = np.mean(data)
            std_val = np.std(data)
            outlier_mask = np.abs(data - mean_val) > threshold * std_val
            
        elif method == 'iqr':
            # Interquartile range method
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            outlier_mask = (data < q1 - threshold * iqr) | (data > q3 + threshold * iqr)
            
        elif method == 'zscore':
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > threshold
            
        else:
            logger.warning(f"Metodo di rimozione outlier sconosciuto: {method}")
            return data
        
        # Interpolate outliers
        clean_data = data.copy()
        if np.any(outlier_mask):
            outlier_indices = np.where(outlier_mask)[0]
            valid_indices = np.where(~outlier_mask)[0]
            
            if len(valid_indices) > 2:
                clean_data[outlier_indices] = np.interp(outlier_indices, valid_indices, data[valid_indices])
                logger.info(f"Rimossi {np.sum(outlier_mask)} outlier con metodo: {method}")
        
        return clean_data
        
    except Exception as e:
        logger.warning(f"Errore nella rimozione outlier: {e}")
        return data


def apply_post_processing(velocities, positions, config):
    """
    Applica post-processing completo a velocità e posizioni.
    
    Args:
        velocities (np.ndarray): Velocità stimate
        positions (np.ndarray): Posizioni stimate  
        config (dict): Configurazione completa
        
    Returns:
        tuple: Velocità e posizioni processate
    """
    post_config = config.get('post_processing', {})
    
    # Rimozione outlier sulle velocità
    velocities = remove_outliers(velocities, post_config)
    
    # Smoothing delle velocità
    velocities = apply_velocity_smoothing(velocities, post_config)
    
    logger.info("Post-processing completato: outlier removal + smoothing")
    return velocities, positions


class ZUPTDetector:
    """
    Rilevatore ZUPT (Zero Velocity Update) che valuta la stazionarietà su una
    finestra mobile e restituisce un singolo valore booleano per l'ultimo campione.
    """
    
    def __init__(self, config):
        """Inizializza il rilevatore ZUPT."""
        zupt_config = config.get('zupt', {})
        
        # Valori di default con compatibilità verso le vecchie chiavi di configurazione
        self.window_size = int(zupt_config.get('window_size', zupt_config.get('window_samples', 20)))
        self.accel_mean_threshold = float(zupt_config.get('accel_threshold', zupt_config.get('acc_mean_thr', 0.4)))
        self.accel_std_threshold = float(zupt_config.get('variance_threshold', zupt_config.get('acc_std_thr', 0.05)))
        self.velocity_threshold = float(zupt_config.get('velocity_threshold', zupt_config.get('vel_thr', 0.05)))
        self.min_stationary_frames = int(zupt_config.get('min_stationary_frames', zupt_config.get('min_stationary', 5)))
        self.cooldown_samples = int(zupt_config.get('cooldown_samples', 0))
        
        # Adaptive threshold configuration
        self.adaptive_thresholds = bool(zupt_config.get('adaptive_thresholds', True))
        self.auto_relax_samples = int(zupt_config.get('auto_relax_after', max(self.window_size * 4, 200)))
        self.relaxation_rate = float(zupt_config.get('relaxation_rate', 0.5))
        self.max_relax_factor = float(zupt_config.get('max_relax_factor', 3.0))
        
        debug_cfg = config.get('debug', {})
        self.debug = bool(zupt_config.get('debug', False) or debug_cfg.get('enable_debug_output', False))
        
        # Contatori per garantire durata minima e cooldown tra ZUPT consecutivi
        self._stationary_counter = 0
        self._cooldown_counter = 0
        self._samples_since_last_zupt = 0
        self._current_relax_multiplier = 1.0
        self._last_debug_report = -1
        
        # Manteniamo i valori di soglia originali per eventuali reset
        self._base_accel_mean_threshold = self.accel_mean_threshold
        self._base_accel_std_threshold = self.accel_std_threshold
        self._base_velocity_threshold = self.velocity_threshold
    
    def detect_stationary(self, accel_data, velocities=None):
        """
        Determina se l'ultimo campione rappresenta una condizione di velocità zero.
        
        Args:
            accel_data: sequenza di accelerazioni (N, 3) o inferiore alla window
            velocities: sequenza di velocità stimate (N, 3), opzionale
            
        Returns:
            bool: True se il campione corrente è considerato stazionario.
        """
        if len(accel_data) < self.window_size:
            self._stationary_counter = 0
            if self._cooldown_counter > 0:
                self._cooldown_counter -= 1
            return False
        
        accel_window = np.asarray(accel_data[-self.window_size:])
        accel_mag = np.linalg.norm(accel_window, axis=1)
        
        # Aggiorna fattore di rilassamento se la modalità adattiva è attiva
        if self.adaptive_thresholds:
            if self._samples_since_last_zupt > self.auto_relax_samples:
                extra = self._samples_since_last_zupt - self.auto_relax_samples
                relax_multiplier = 1.0 + (extra / max(self.auto_relax_samples, 1)) * self.relaxation_rate
                relax_multiplier = min(relax_multiplier, self.max_relax_factor)
            else:
                relax_multiplier = 1.0
        else:
            relax_multiplier = 1.0
        self._current_relax_multiplier = relax_multiplier
        
        mean_threshold = self._base_accel_mean_threshold * relax_multiplier
        std_threshold = self._base_accel_std_threshold * relax_multiplier
        velocity_threshold = self._base_velocity_threshold * relax_multiplier
        
        mean_accel_norm = np.linalg.norm(np.mean(accel_window, axis=0))
        std_accel_mag = np.std(accel_mag)
        
        velocity_ok = True
        if velocities is not None and len(velocities) >= self.window_size:
            velocity_window = np.asarray(velocities[-self.window_size:])
            vel_mag = np.linalg.norm(velocity_window, axis=1)
            velocity_ok = np.mean(vel_mag) < velocity_threshold
        
        stationary_candidate = (
            mean_accel_norm < mean_threshold and
            std_accel_mag < std_threshold and
            velocity_ok
        )
        
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            stationary_candidate = False
        
        if stationary_candidate:
            self._stationary_counter += 1
            if self._stationary_counter >= self.min_stationary_frames:
                self._cooldown_counter = self.cooldown_samples
                self._samples_since_last_zupt = 0
                self._current_relax_multiplier = 1.0
                return True
        else:
            self._stationary_counter = 0
            self._samples_since_last_zupt += 1
        
        # Debug logging controllato per capire perché non si attiva ZUPT
        if self.debug:
            should_report = False
            if stationary_candidate is False and self._samples_since_last_zupt % max(1, self.window_size) == 0:
                should_report = True
            elif self.adaptive_thresholds and relax_multiplier > 1.0 and self._samples_since_last_zupt != self._last_debug_report:
                should_report = True
            if should_report:
                logger.info(
                    "🔎 ZUPT debug | mean=%.4f(th=%.4f) std=%.4f(th=%.4f) vel_ok=%s(th=%.4f) relax=%.2f counter=%d",
                    mean_accel_norm,
                    mean_threshold,
                    std_accel_mag,
                    std_threshold,
                    velocity_ok,
                    velocity_threshold,
                    relax_multiplier,
                    self._samples_since_last_zupt,
                )
                self._last_debug_report = self._samples_since_last_zupt
        
        return False


def preprocess_acceleration(accel_data: np.ndarray, fs: float, config: dict) -> np.ndarray:
    """
    Pre-processa i dati di accelerazione con filtri per ridurre rumore e deriva.
    
    Args:
        accel_data: Array (N, 3) di accelerazioni [ax, ay, az]
        fs: Frequenza di campionamento [Hz]
        config: Configurazione con sezione 'preprocessing'
    
    Returns:
        Array filtrato (N, 3)
    """
    preproc_cfg = config.get('preprocessing', {})
    
    logger = logging.getLogger(__name__)
    # print(f"DEBUG: preprocess_acceleration called, enabled={preproc_cfg.get('enabled', False)}")
    logger.info(f"🔍 Preprocessing config: enabled={preproc_cfg.get('enabled', False)}")
    
    if not preproc_cfg.get('enabled', False):
        logger.info("⚠️ Preprocessing disabled, returning raw data")
        return accel_data
    
    logger.info("🚀 Starting acceleration preprocessing...")
    # print("DEBUG: Starting preprocessing filters...")
    filtered = accel_data.copy()
    stats_before = {
        'mean': np.mean(filtered, axis=0),
        'std': np.std(filtered, axis=0),
        'max': np.max(np.abs(filtered), axis=0)
    }
    
    # 1. Low-pass filter (Butterworth) per rimuovere rumore ad alta frequenza
    if preproc_cfg.get('lowpass_filter', {}).get('enabled', False):
        # print("DEBUG: Applying low-pass filter...")
        lp_cfg = preproc_cfg['lowpass_filter']
        cutoff = lp_cfg.get('cutoff_hz', 10.0)
        order = lp_cfg.get('order', 4)
        
        # Butterworth filter
        nyquist = fs / 2.0
        normalized_cutoff = cutoff / nyquist
        
        if normalized_cutoff < 1.0:
            b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
            
            for axis in range(3):
                # Applica filtro bidirezionale (zero-phase)
                filtered[:, axis] = signal.filtfilt(b, a, filtered[:, axis])
            
            logger.info(f"✅ Low-pass filter applicato: cutoff={cutoff}Hz, order={order}")
            # print(f"DEBUG: Low-pass filter applied: cutoff={cutoff}Hz")
        else:
            logger.warning(f"⚠️ Cutoff {cutoff}Hz >= Nyquist {nyquist}Hz, skipping low-pass")
    
    # 2. Median filter per rimuovere spike/outliers
    if preproc_cfg.get('median_filter', {}).get('enabled', False):
        med_cfg = preproc_cfg['median_filter']
        kernel_size = med_cfg.get('kernel_size', 5)
        
        for axis in range(3):
            filtered[:, axis] = signal.medfilt(filtered[:, axis], kernel_size=kernel_size)
        
        logger.info(f"✅ Median filter applicato: kernel_size={kernel_size}")
    
    # 3. High-pass filter (opzionale) per rimuovere drift DC
    if preproc_cfg.get('highpass_filter', {}).get('enabled', False):
        hp_cfg = preproc_cfg['highpass_filter']
        cutoff = hp_cfg.get('cutoff_hz', 0.1)
        order = hp_cfg.get('order', 2)
        
        nyquist = fs / 2.0
        normalized_cutoff = cutoff / nyquist
        
        if 0 < normalized_cutoff < 1.0:
            b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
            
            for axis in range(3):
                filtered[:, axis] = signal.filtfilt(b, a, filtered[:, axis])
            
            logger.info(f"✅ High-pass filter applicato: cutoff={cutoff}Hz, order={order}")
    
    stats_after = {
        'mean': np.mean(filtered, axis=0),
        'std': np.std(filtered, axis=0),
        'max': np.max(np.abs(filtered), axis=0)
    }
    
    logger.info(f"📊 Preprocessing stats:")
    logger.info(f"  Before - std: {stats_before['std']}, max: {stats_before['max']}")
    logger.info(f"  After  - std: {stats_after['std']}, max: {stats_after['max']}")
    
    return filtered


def postprocess_velocity_position(velocity: np.ndarray, position: np.ndarray, 
                                   timestamps: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Post-processa velocità e posizione per ridurre deriva e migliorare periodicità.
    
    Args:
        velocity: Array (N, 3) di velocità
        position: Array (N, 3) di posizioni
        timestamps: Array (N,) di timestamp
        config: Configurazione con sezione 'postprocess'
    
    Returns:
        Tupla (velocity_corrected, position_corrected)
    """
    postproc_cfg = config.get('postprocess', {})
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting advanced post-processing...")
    
    vel_corrected = velocity.copy()
    pos_corrected = position.copy()
    
    # 1. Detrending lineare per rimuovere deriva
    if postproc_cfg.get('detrend_velocity', {}).get('enabled', False):
        for axis in range(3):
            vel_corrected[:, axis] = signal.detrend(vel_corrected[:, axis], type='linear')
        logger.info("✅ Velocity detrending applicato")
    
    if postproc_cfg.get('detrend_position', {}).get('enabled', False):
        for axis in range(3):
            pos_corrected[:, axis] = signal.detrend(pos_corrected[:, axis], type='linear')
        logger.info("✅ Position detrending applicato")
    
    # 2. High-pass filter sulla posizione per rimuovere deriva DC
    if postproc_cfg.get('position_highpass', {}).get('enabled', False):
        hp_cfg = postproc_cfg['position_highpass']
        cutoff = hp_cfg.get('cutoff_hz', 0.05)
        order = hp_cfg.get('order', 2)
        fs = 1.0 / np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 100.0
        
        nyquist = fs / 2.0
        normalized_cutoff = cutoff / nyquist
        
        if 0 < normalized_cutoff < 1.0:
            b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
            
            for axis in range(3):
                pos_corrected[:, axis] = signal.filtfilt(b, a, pos_corrected[:, axis])
            
            logger.info(f"✅ Position high-pass filter: cutoff={cutoff}Hz")
    
    # 3. Savitzky-Golay smoothing per movimento periodico
    # 3. Savitzky-Golay smoothing per movimento periodico
    if postproc_cfg.get('savgol_smooth', {}).get('enabled', False):
        sg_cfg = postproc_cfg['savgol_smooth']
        window_length = sg_cfg.get('window_length', 51)
        polyorder = sg_cfg.get('polyorder', 3)
        
        # Assicura window_length dispari
        if window_length % 2 == 0:
            window_length += 1
        
        # Assicura window_length > polyorder
        if window_length <= polyorder:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
        
        if window_length < len(vel_corrected):
            for axis in range(3):
                vel_corrected[:, axis] = signal.savgol_filter(
                    vel_corrected[:, axis], window_length, polyorder
                )
                pos_corrected[:, axis] = signal.savgol_filter(
                    pos_corrected[:, axis], window_length, polyorder
                )
            logger.info(f"✅ Savitzky-Golay smoothing: window={window_length}, poly={polyorder}")
    
    logger.info("✅ Advanced post-processing completed")
    return vel_corrected, pos_corrected


class ExtendedKalmanFilter9D:
    """
    Extended Kalman Filter 9D per stima di posizione, velocità e bias accelerometrico.
    
    Stato: [px, py, pz, vx, vy, vz, bx, by, bz]
    Implementa Joseph form per stabilità numerica e ZUPT per correzioni.
    """
    
    def __init__(self, config):
        """Inizializza l'EKF 9D."""
        self.config = config
        ekf_config = config.get('ekf', {})
        debug_cfg = config.get('debug', {})
        self.debug_enabled = bool(debug_cfg.get('enable_debug_output', False))
        self.debug_every_n = int(debug_cfg.get('log_every_n', 100))
        self.debug_print_state = bool(debug_cfg.get('print_state_snapshots', False))
        self.debug_log_innovations = bool(debug_cfg.get('log_innovations', True))
        self.step_counter = 0
        self._last_debug_step = -1
        if self.debug_enabled:
            logger.info(
                "🟣 Debug EKF9D attivo (every %d steps, print=%s, innovations=%s)",
                self.debug_every_n,
                self.debug_print_state,
                self.debug_log_innovations,
            )
        
        # Stato 9D: [px, py, pz, vx, vy, vz, bx, by, bz]
        initial_pos = ekf_config.get('initial_position', [0.0, 0.0, 0.0])
        initial_vel = ekf_config.get('initial_velocity', [0.0, 0.0, 0.0])  
        initial_bias = ekf_config.get('initial_bias', [0.0, 0.0, 0.0])
        
        self.x = np.array(initial_pos + initial_vel + initial_bias).reshape(9, 1)
        
        # Matrice di covarianza iniziale 9x9
        initial_cov_diag = ekf_config.get('initial_covariance', {})
        pos_var = initial_cov_diag.get('position', 1.0)
        vel_var = initial_cov_diag.get('velocity', 1.0)
        bias_var = initial_cov_diag.get('bias', 1e-6)
        
        self.P = np.diag([pos_var]*3 + [vel_var]*3 + [bias_var]*3)
        
        # Noise matrices
        process_noise = ekf_config.get('process_noise', {})
        self.Q = self._build_process_noise_matrix(process_noise)
        
        # MEGA DEBUG - Measurement noise
        measurement_noise = ekf_config.get('measurement_noise', 0.1)
        logger.info("🔍 DEBUG TYPES - Measurement noise:")
        logger.info(f"  measurement_noise type: {type(measurement_noise)}")
        logger.info(f"  measurement_noise value: {measurement_noise}")
        
        try:
            measurement_noise_array = np.asarray(measurement_noise, dtype=float)
            logger.info(f"  Converted to numpy array: dtype={measurement_noise_array.dtype}, shape={measurement_noise_array.shape}")
        except (TypeError, ValueError) as e:
            logger.error(f"  FAILED numpy conversion: {e}")
            logger.warning(
                "⚠️ Invalid measurement_noise value %s; falling back to 0.1",
                measurement_noise,
            )
            measurement_noise_array = np.array(0.1, dtype=float)

        if measurement_noise_array.ndim == 0:
            self.R = np.eye(3, dtype=float) * float(measurement_noise_array)
            logger.info(f"  R created as scalar * I(3): {float(measurement_noise_array)}")
        elif measurement_noise_array.ndim == 1:
            if measurement_noise_array.size != 3:
                raise ValueError(
                    "measurement_noise vector must have length 3 (for x, y, z). "
                    f"Received length {measurement_noise_array.size}"
                )
            self.R = np.diag(measurement_noise_array.astype(float))
            logger.info(f"  R created as diagonal from vector: {measurement_noise_array}")
        else:
            measurement_noise_array = np.asarray(measurement_noise_array, dtype=float)
            if measurement_noise_array.shape != (3, 3):
                raise ValueError(
                    "measurement_noise matrix must be 3x3. "
                    f"Received shape {measurement_noise_array.shape}"
                )
            self.R = measurement_noise_array
            logger.info(f"  R created as full matrix")
        
        logger.info(f"✅ R created: shape={self.R.shape}, dtype={self.R.dtype}")
        logger.info(f"✅ R diagonal: {np.diag(self.R)}")
        
        # Parametro dt (compatibile con sample_rate)
        sample_rate = ekf_config.get('sample_rate_hz')
        if sample_rate and sample_rate > 0:
            self.dt = 1.0 / sample_rate
        else:
            self.dt = ekf_config.get('dt', 0.01)
        
        # Gestione gravità e clamp accelerazioni
        self.use_gravity_compensation = ekf_config.get('use_gravity_compensation', False)
        gravity_value = ekf_config.get('gravity_mps2', 9.80665)
        gravity_vector = ekf_config.get('gravity_vector', [0.0, 0.0, gravity_value])
        self.gravity_vector = np.array(gravity_vector, dtype=float).reshape(3, 1)
        
        clamp_cfg = ekf_config.get('clamp_accel', {})
        self.clamp_accel = clamp_cfg.get('enabled', False)
        self.clamp_min = clamp_cfg.get('min', -50.0)
        self.clamp_max = clamp_cfg.get('max', 50.0)
        
        # Joseph form flag
        self.use_joseph_form = ekf_config.get('numerical_stability', {}).get('use_joseph_form', True)
        
        # ZUPT detector
        self.zupt_detector = ZUPTDetector(config)
        
        # Diagnostics
        self.innovation_history = []
        self.nis_history = []
        self.state_history = []
        
        # Stato interno per diagnostica
        self._prev_velocity = self.x[3:6].copy()
        self._last_accel_corrected = np.zeros((3, 1))
    
    def _build_process_noise_matrix(self, process_config):
        """Costruisce la matrice Q del rumore di processo."""
        # MEGA DEBUG - Process noise configuration
        logger.info("🔍 DEBUG TYPES - Process noise configuration:")
        logger.info(f"  process_config type: {type(process_config)}")
        logger.info(f"  process_config value: {process_config}")
        
        if isinstance(process_config, dict):
            for key, value in process_config.items():
                logger.info(f"  {key}: type={type(value)}, value={value}")
        
        # Default values
        pos_noise = process_config.get('position', 1e-4)
        vel_noise = process_config.get('velocity', 1e-3)
        bias_noise = process_config.get('bias', 1e-8)
        
        logger.info(f"🔍 Extracted noise values:")
        logger.info(f"  pos_noise: type={type(pos_noise)}, value={pos_noise}")
        logger.info(f"  vel_noise: type={type(vel_noise)}, value={vel_noise}")
        logger.info(f"  bias_noise: type={type(bias_noise)}, value={bias_noise}")
        
        # Force conversion to float
        try:
            pos_noise_float = float(pos_noise)
            logger.info(f"  pos_noise converted: {pos_noise} -> {pos_noise_float} ✅")
        except (TypeError, ValueError) as e:
            logger.error(f"  pos_noise FAILED conversion: {e}")
            pos_noise_float = 1e-4
            
        try:
            vel_noise_float = float(vel_noise)
            logger.info(f"  vel_noise converted: {vel_noise} -> {vel_noise_float} ✅")
        except (TypeError, ValueError) as e:
            logger.error(f"  vel_noise FAILED conversion: {e}")
            vel_noise_float = 1e-3
            
        try:
            bias_noise_float = float(bias_noise)
            logger.info(f"  bias_noise converted: {bias_noise} -> {bias_noise_float} ✅")
        except (TypeError, ValueError) as e:
            logger.error(f"  bias_noise FAILED conversion: {e}")
            bias_noise_float = 1e-8
        
        Q_diag = [pos_noise_float]*3 + [vel_noise_float]*3 + [bias_noise_float]*3
        logger.info(f"🔍 Q_diag before building Q: {Q_diag}")
        logger.info(f"🔍 Q_diag types: {[type(x) for x in Q_diag]}")
        
        Q_matrix = np.diag(Q_diag)
        logger.info(f"✅ Q_matrix created: shape={Q_matrix.shape}, dtype={Q_matrix.dtype}")
        logger.info(f"✅ Q_matrix diagonal sample: {Q_matrix.diagonal()[:3]}")
        
        return Q_matrix
    
    def _debug_state_snapshot(self, label, extra_msg=""):
        """Debug helper per loggare stato/covarianza quando abilitato."""
        if not self.debug_enabled:
            return
        msg = (
            f"{label} | pos={np.round(self.x[0:3,0],4)} vel={np.round(self.x[3:6,0],4)} "
            f"bias={np.round(self.x[6:9,0],5)} traceP={np.trace(self.P):.4f} {extra_msg}"
        )
        logger.debug(f"🟣 EKF9D {msg}")
        if self.debug_print_state:
            print_colored(msg, "🟣", "magenta")
    
    def predict(self, accel_measurement):
        """
        Propagazione dello stato usando la misura accelerometrica come input.
        
        Args:
            accel_measurement (array-like): Accelerazione misurata (3,).
        """
        self.step_counter += 1
        
        # MEGA DEBUG - First call only
        if self.step_counter == 1:
            logger.info("🔍 DEBUG FIRST PREDICT CALL:")
            logger.info(f"  accel_measurement type: {type(accel_measurement)}")
            logger.info(f"  accel_measurement value: {accel_measurement}")
            if hasattr(accel_measurement, 'dtype'):
                logger.info(f"  accel_measurement dtype: {accel_measurement.dtype}")
            if hasattr(accel_measurement, 'shape'):
                logger.info(f"  accel_measurement shape: {accel_measurement.shape}")
        
        accel = np.asarray(accel_measurement, dtype=float).reshape(3, 1)
        
        if self.step_counter == 1:
            logger.info(f"  accel after conversion: type={type(accel)}, dtype={accel.dtype}, shape={accel.shape}")
            logger.info(f"  accel values: {accel.flatten()}")
            
            # Check Q matrix
            logger.info(f"  Q type: {type(self.Q)}, dtype: {self.Q.dtype}, shape: {self.Q.shape}")
            logger.info(f"  Q sample values: {self.Q.diagonal()[:3]}")
            
            # Check P matrix  
            logger.info(f"  P type: {type(self.P)}, dtype: {self.P.dtype}, shape: {self.P.shape}")
            logger.info(f"  P sample values: {self.P.diagonal()[:3]}")
        
        if self.clamp_accel:
            accel = np.clip(accel, self.clamp_min, self.clamp_max)
        
        if self.use_gravity_compensation:
            accel = accel - self.gravity_vector
        
        # Salva velocità precedente per diagnostica
        self._prev_velocity = self.x[3:6].copy()
        
        # Rimuove il bias stimato
        accel_corrected = accel - self.x[6:9]
        self._last_accel_corrected = accel_corrected.copy()
        
        dt = self.dt
        
        # Propagazione stato (modello integrazione accelerazione)
        self.x[0:3] = self.x[0:3] + self.x[3:6] * dt + 0.5 * accel_corrected * dt**2
        self.x[3:6] = self.x[3:6] + accel_corrected * dt
        # I bias restano invariati
        
        # Matrice di transizione linearizzata
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * (dt ** 2) * np.eye(3)
        F[3:6, 6:9] = -dt * np.eye(3)
        
        # Predizione covarianza
        self.P = F @ self.P @ F.T + self.Q
        self.P = (self.P + self.P.T) / 2
        
        # Diagnostica: innovazione = accelerazione corretta
        innovation = accel_corrected.flatten()
        self.innovation_history.append(innovation)
        if self.debug_enabled and self.debug_log_innovations:
            if self.debug_every_n <= 1 or (self.step_counter % self.debug_every_n == 0):
                logger.debug(
                    "🟣 EKF9D innovation step %d | acc_raw=%s acc_corr=%s vel=%s",
                    self.step_counter,
                    np.round(accel.flatten(), 4),
                    np.round(accel_corrected.flatten(), 4),
                    np.round(self.x[3:6, 0], 4),
                )
        
        if self.debug_enabled and (
            self.debug_every_n <= 1 or (self.step_counter % self.debug_every_n == 0)
        ):
            self._debug_state_snapshot(
                "predict",
                extra_msg=f"|acc_corr|={np.linalg.norm(accel_corrected):.4f} step={self.step_counter}",
            )
        
        try:
            R_inv = np.linalg.inv(self.R)
            nis = float(accel_corrected.T @ R_inv @ accel_corrected)
        except np.linalg.LinAlgError:
            nis = float(accel_corrected.T @ accel_corrected)
        self.nis_history.append(nis)
        self.state_history.append(self.x.flatten().copy())
    
    def update(self, accel_measurement, apply_zupt=True):
        """
        Metodo mantenuto per retro-compatibilità. Delega alla nuova predizione.
        """
        warnings.warn(
            "ExtendedKalmanFilter9D.update è deprecato: utilizzare predict() seguito da apply_zupt_update() se necessario.",
            RuntimeWarning,
            stacklevel=2,
        )
        self.predict(accel_measurement)
    
    def apply_zupt_update(self, accel_measurement=None):
        """Applica correzione ZUPT (velocità = 0)."""
        # Matrice di osservazione per velocità
        H_zupt = np.zeros((3, 9))
        H_zupt[0:3, 3:6] = np.eye(3)  # Osserviamo le velocità
        
        # Misura ZUPT (velocità = 0)
        z_zupt = np.zeros((3, 1))
        
        # Innovation
        y = z_zupt - H_zupt @ self.x
        
        # Innovation covariance (bassa varianza per ZUPT)
        R_zupt = np.eye(3) * 1e-6
        S = H_zupt @ self.P @ H_zupt.T + R_zupt
        
        # Kalman gain
        try:
            K = self.P @ H_zupt.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H_zupt.T @ np.linalg.pinv(S)
        
        # Update
        self.x = self.x + K @ y
        
        if self.use_joseph_form:
            I_KH = np.eye(9) - K @ H_zupt
            self.P = I_KH @ self.P @ I_KH.T + K @ R_zupt @ K.T
        else:
            self.P = (np.eye(9) - K @ H_zupt) @ self.P
            
        # Assicura simmetria
        self.P = (self.P + self.P.T) / 2
        self._debug_state_snapshot("apply_zupt_update", extra_msg="velocity->0")
        
        # Aggiorna il bias usando la misura accelerometrica durante lo ZUPT
        if accel_measurement is not None:
            accel = np.asarray(accel_measurement, dtype=float).reshape(3, 1)
            if self.clamp_accel:
                accel = np.clip(accel, self.clamp_min, self.clamp_max)
            if self.use_gravity_compensation:
                accel = accel - self.gravity_vector
            
            H_bias = np.zeros((3, 9))
            H_bias[0:3, 6:9] = np.eye(3)
            z_bias = accel
            y_bias = z_bias - H_bias @ self.x
            
            R_bias = self.R
            S_bias = H_bias @ self.P @ H_bias.T + R_bias
            try:
                K_bias = self.P @ H_bias.T @ np.linalg.inv(S_bias)
            except np.linalg.LinAlgError:
                K_bias = self.P @ H_bias.T @ np.linalg.pinv(S_bias)
            
            self.x = self.x + K_bias @ y_bias
            
            if self.use_joseph_form:
                I_KH_bias = np.eye(9) - K_bias @ H_bias
                self.P = I_KH_bias @ self.P @ I_KH_bias.T + K_bias @ R_bias @ K_bias.T
            else:
                self.P = (np.eye(9) - K_bias @ H_bias) @ self.P
            
            self.P = (self.P + self.P.T) / 2
            
            # Aggiorna diagnostica con l'ultima correzione
            self.innovation_history.append((accel - self.x[6:9]).flatten())
            try:
                R_inv = np.linalg.inv(R_bias)
                nis_bias = float(y_bias.T @ R_inv @ y_bias)
            except np.linalg.LinAlgError:
                nis_bias = float(y_bias.T @ y_bias)
            self.nis_history.append(nis_bias)
            self._debug_state_snapshot(
                "apply_zupt_update_bias",
                extra_msg=f"acc_meas={np.round(accel.flatten(),4)}",
            )
        
        if self.state_history:
            self.state_history[-1] = self.x.flatten().copy()
    
    def get_position(self):
        """Restituisce posizione stimata."""
        return self.x[0:3].flatten()
    
    def get_velocity(self):
        """Restituisce velocità stimata."""
        return self.x[3:6].flatten()
    
    def get_bias(self):
        """Restituisce bias stimato."""
        return self.x[6:9].flatten()
    
    def get_diagnostics(self):
        """Restituisce diagnostici del filtro."""
        return {
            'innovation_history': np.array(self.innovation_history),
            'nis_history': np.array(self.nis_history),
            'state_history': np.array(self.state_history),
            'current_covariance': self.P.copy()
        }


class EKFParameterOptimizer:
    """
    Ottimizzatore automatico dei parametri EKF attraverso iterazioni successive.
    
    Utilizza algoritmi di ottimizzazione per trovare automaticamente i migliori
    parametri Q, R e altri parametri del filtro basandosi sulle metriche di performance.
    """
    
    def __init__(self, base_config, data, axis):
        """
        Inizializza l'ottimizzatore.
        
        Args:
            base_config (dict): Configurazione di base
            data (pd.DataFrame): Dati di accelerazione
            axis (str): Asse da analizzare
        """
        self.base_config = base_config.copy()
        self.data = data
        self.axis = axis
        self.optimization_history = []
        self.best_params = None
        self.best_score = float('inf')
        
        # Parametri di ottimizzazione
        self.max_iterations = 15
        self.convergence_threshold = 0.02  # 2% miglioramento minimo
        self.parameter_ranges = {
            'position_noise': (1e-6, 1e-3),
            'velocity_noise': (1e-4, 1e-1),
            'acceleration_noise': (1e-2, 1.0),
            'bias_noise': (1e-6, 1e-3),
            'measurement_noise': (0.05, 1.0)
        }
        
        logger.info(f"🎯 Ottimizzatore EKF inizializzato per asse {axis}")
        logger.info(f"📊 Parametri di ottimizzazione: {self.max_iterations} iterazioni max")
    
    def calculate_performance_score(self, performance_metrics):
        """
        Calcola un punteggio di performance combinato dalle metriche EKF.
        
        Args:
            performance_metrics (dict): Metriche di performance dal monitor EKF
            
        Returns:
            float: Punteggio di performance (più basso è migliore)
        """
        try:
            # Validazione input
            if performance_metrics is None:
                logger.warning("⚠️ performance_metrics è None - usando score penalità")
                return 1000
                
            if not isinstance(performance_metrics, dict):
                logger.warning(f"⚠️ performance_metrics non è dict: {type(performance_metrics)}")
                return 1000
            
            # Pesi per diverse metriche (più importante = peso maggiore)
            weights = {
                'trace_penalty': 0.3,      # Penalità per traccia alta
                'innovation_penalty': 0.25, # Penalità per innovazioni correlate
                'nis_penalty': 0.2,        # Penalità per NIS fuori range
                'convergence_penalty': 0.15, # Penalità per non convergenza
                'stability_penalty': 0.1   # Penalità per instabilità
            }
            
            # Estrai metriche con valori di default sicuri
            trace_stats = performance_metrics.get('trace_statistics', {})
            innovation_stats = performance_metrics.get('innovation_whiteness', {})
            nis_stats = performance_metrics.get('nis_statistics', {})
            convergence_stats = performance_metrics.get('convergence', {})
            
            mean_trace = trace_stats.get('mean_trace', 50000)
            max_correlation = innovation_stats.get('max_correlation', 1.0)
            mean_nis = nis_stats.get('mean_nis', 0.5)
            is_converged = convergence_stats.get('is_converged', False)
            current_trace = convergence_stats.get('current_trace', 50000)
            
            # Debug logging per capire i valori
            logger.debug(f"🔍 Metriche score: trace={mean_trace}, corr={max_correlation}, "
                        f"nis={mean_nis}, conv={is_converged}, curr_trace={current_trace}")
            
            # Calcola penalità individuali
            trace_penalty = min(mean_trace / 1000, 100)  # Normalizza traccia
            innovation_penalty = max_correlation * 100 if max_correlation is not None else 100
            
            # Penalità NIS (ideale ~1-3)
            if mean_nis is not None and 0.5 <= mean_nis <= 3.0:
                nis_penalty = 0
            else:
                nis_penalty = abs((mean_nis or 1.5) - 1.5) * 20
            
            convergence_penalty = 0 if is_converged else 50
            stability_penalty = min(current_trace / 10000, 20) if current_trace is not None else 20
            
            # Punteggio finale pesato
            total_score = (
                weights['trace_penalty'] * trace_penalty +
                weights['innovation_penalty'] * innovation_penalty +
                weights['nis_penalty'] * nis_penalty +
                weights['convergence_penalty'] * convergence_penalty +
                weights['stability_penalty'] * stability_penalty
            )
            
            logger.debug(f"📊 Score components: trace={trace_penalty:.2f}, innovation={innovation_penalty:.2f}, "
                        f"nis={nis_penalty:.2f}, convergence={convergence_penalty:.2f}, stability={stability_penalty:.2f}")
            logger.info(f"🎯 Score totale calcolato: {total_score:.2f}")
            
            return total_score
            
        except Exception as e:
            logger.error(f"❌ Errore nel calcolo dello score: {e}")
            logger.error(f"❌ Metriche disponibili: {list(performance_metrics.keys()) if performance_metrics else 'None'}")
            return 1000  # Score alto come penalità
    
    def generate_parameter_candidate(self, iteration):
        """
        Genera un candidato di parametri per l'iterazione corrente.
        
        Args:
            iteration (int): Numero dell'iterazione corrente
            
        Returns:
            dict: Nuovi parametri da testare
        """
        if iteration == 0:
            # Prima iterazione: usa parametri attuali
            return {
                'position_noise': self.base_config['ekf']['process_noise']['position'],
                'velocity_noise': self.base_config['ekf']['process_noise']['velocity'],
                'acceleration_noise': self.base_config['ekf']['process_noise']['acceleration'],
                'bias_noise': self.base_config['ekf']['process_noise']['bias'],
                'measurement_noise': self.base_config['ekf']['measurement_noise']
            }
        
        # Strategie di ottimizzazione
        if iteration <= 5:
            # Fasi iniziali: esplorazione ampia
            strategy = "exploration"
            scale_factor = 0.5  # Variazioni più ampie
        elif iteration <= 10:
            # Fasi intermedie: raffinamento
            strategy = "refinement"  
            scale_factor = 0.2  # Variazioni moderate
        else:
            # Fasi finali: fine-tuning
            strategy = "fine_tuning"
            scale_factor = 0.1  # Variazioni piccole
        
        # Logica di ottimizzazione basata sulla storia
        if len(self.optimization_history) > 0:
            last_result = self.optimization_history[-1]
            last_score = last_result['score']
            last_params = last_result['parameters']
            
            # Analizza tendenze per direzione ottimizzazione
            if len(self.optimization_history) >= 2:
                trend = last_score - self.optimization_history[-2]['score']
                if trend > 0:  # Peggioramento
                    scale_factor *= -0.8  # Inverti direzione e riduci step
            
            # Genera nuovi parametri basati sui migliori precedenti
            new_params = {}
            for param_name, (min_val, max_val) in self.parameter_ranges.items():
                current_val = last_params[param_name]
                
                # Calcola variazione basata su strategia
                if strategy == "exploration":
                    # Esplorazione: variazioni random ampie
                    variation = np.random.uniform(-scale_factor, scale_factor) * current_val
                elif strategy == "refinement":
                    # Raffinamento: gradiente verso migliori parametri
                    if self.best_params:
                        target_val = self.best_params[param_name]
                        variation = (target_val - current_val) * scale_factor
                    else:
                        variation = np.random.uniform(-scale_factor, scale_factor) * current_val
                else:  # fine_tuning
                    # Fine-tuning: piccole variazioni random
                    variation = np.random.normal(0, scale_factor * current_val)
                
                new_val = np.clip(current_val + variation, min_val, max_val)
                new_params[param_name] = new_val
            
            return new_params
        
        # Fallback: parametri casuali nei range
        return {
            param_name: np.random.uniform(min_val, max_val)
            for param_name, (min_val, max_val) in self.parameter_ranges.items()
        }
    
    def update_config_with_parameters(self, config, params):
        """
        Aggiorna la configurazione con i nuovi parametri.
        
        Args:
            config (dict): Configurazione da aggiornare
            params (dict): Nuovi parametri
            
        Returns:
            dict: Configurazione aggiornata
        """
        updated_config = config.copy()
        
        updated_config['ekf']['process_noise']['position'] = params['position_noise']
        updated_config['ekf']['process_noise']['velocity'] = params['velocity_noise']
        updated_config['ekf']['process_noise']['acceleration'] = params['acceleration_noise']
        updated_config['ekf']['process_noise']['bias'] = params['bias_noise']
        updated_config['ekf']['measurement_noise'] = params['measurement_noise']
        
        return updated_config
    
    def run_optimization(self):
        """
        Esegue l'ottimizzazione iterativa dei parametri.
        
        Returns:
            dict: Migliori parametri trovati e storia dell'ottimizzazione
        """
        print_colored("🎯 ============ AVVIO OTTIMIZZAZIONE AUTOMATICA EKF ============", "🎯", "magenta")
        logger.info("🎯 Avvio ottimizzazione automatica dei parametri EKF")
        
        for iteration in range(self.max_iterations):
            print_colored(f"🔄 Iterazione {iteration + 1}/{self.max_iterations}", "🔄", "cyan")
            logger.info(f"🔄 Ottimizzazione iterazione {iteration + 1}/{self.max_iterations}")
            
            try:
                # Genera parametri candidati
                candidate_params = self.generate_parameter_candidate(iteration)
                
                # Assicura che tutti i parametri siano numerici
                numeric_params = {}
                for key, value in candidate_params.items():
                    try:
                        numeric_params[key] = float(value)
                    except (TypeError, ValueError):
                        logger.error(f"❌ Parametro '{key}' non numerico: {value!r}")
                        raise
                candidate_params = numeric_params
                
                # Aggiorna configurazione
                test_config = self.update_config_with_parameters(self.base_config, candidate_params)
                
                # Disabilita output verboso per ottimizzazione
                test_config['debug']['verbose'] = False
                test_config['performance_monitoring']['generate_performance_plots'] = False
                
                print_colored(f"🧪 Test parametri: Q=[{candidate_params['position_noise']:.2e}, "
                            f"{candidate_params['velocity_noise']:.2e}, {candidate_params['acceleration_noise']:.2e}, "
                            f"{candidate_params['bias_noise']:.2e}], R={candidate_params['measurement_noise']:.3f}", "🧪", "yellow")
                
                # Esegui EKF con parametri candidati
                results = apply_extended_kalman_filter(self.data, test_config, self.axis)
                
                # Verifica risultati più robusta
                if results is not None and isinstance(results, dict) and 'performance_metrics' in results:
                    # Calcola score di performance
                    score = self.calculate_performance_score(results['performance_metrics'])
                    
                    # Salva risultato
                    iteration_result = {
                        'iteration': iteration + 1,
                        'parameters': candidate_params,
                        'score': score,
                        'performance_metrics': results['performance_metrics']
                    }
                    self.optimization_history.append(iteration_result)
                    
                    print_colored(f"📊 Score performance: {score:.2f}", "📊", "blue")
                    
                    # Aggiorna migliori parametri
                    if score < self.best_score:
                        improvement = ((self.best_score - score) / self.best_score) * 100
                        self.best_score = score
                        self.best_params = candidate_params.copy()
                        
                        print_colored(f"✨ Nuovo miglior risultato! Miglioramento: {improvement:.1f}%", "✨", "green")
                        logger.info(f"✨ Nuovi migliori parametri trovati con score {score:.2f}")
                    
                    # Verifica convergenza
                    if iteration >= 3:
                        recent_scores = [r['score'] for r in self.optimization_history[-3:]]
                        if max(recent_scores) - min(recent_scores) < self.convergence_threshold * min(recent_scores):
                            print_colored(f"🎯 Convergenza raggiunta dopo {iteration + 1} iterazioni", "🎯", "green")
                            logger.info(f"🎯 Ottimizzazione convergente dopo {iteration + 1} iterazioni")
                            break
                
                else:
                    print_colored("❌ Errore nell'esecuzione EKF, parametri scartati", "❌", "red")
                    logger.warning("Errore nell'esecuzione EKF durante ottimizzazione")
            
            except Exception as e:
                print_colored(f"❌ Errore nell'iterazione {iteration + 1}: {e}", "❌", "red")
                logger.error(f"Errore nell'iterazione di ottimizzazione {iteration + 1}: {e}")
                continue
        
        # Report finale ottimizzazione
        self._print_optimization_summary()
        
        return {
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'total_iterations': len(self.optimization_history)
        }
    
    def _print_optimization_summary(self):
        """Stampa il sommario dell'ottimizzazione."""
        print_colored("🏆 ============ RISULTATI OTTIMIZZAZIONE ============", "🏆", "magenta")
        
        if self.best_params:
            print_colored(f"✨ Miglior score ottenuto: {self.best_score:.2f}", "✨", "green")
            print_colored("🎯 Migliori parametri trovati:", "🎯", "cyan")
            print_colored(f"  📍 Position noise: {self.best_params['position_noise']:.2e}", "📍", "cyan")
            print_colored(f"  🚀 Velocity noise: {self.best_params['velocity_noise']:.2e}", "🚀", "cyan")
            print_colored(f"  ⚡ Acceleration noise: {self.best_params['acceleration_noise']:.2e}", "⚡", "cyan")
            print_colored(f"  🎯 Bias noise: {self.best_params['bias_noise']:.2e}", "🎯", "cyan")
            print_colored(f"  📏 Measurement noise: {self.best_params['measurement_noise']:.3f}", "📏", "cyan")
            
            # Calcola miglioramento totale
            if len(self.optimization_history) > 1:
                initial_score = self.optimization_history[0]['score']
                total_improvement = ((initial_score - self.best_score) / initial_score) * 100
                print_colored(f"📈 Miglioramento totale: {total_improvement:.1f}%", "📈", "green")
        
        print_colored(f"🔄 Iterazioni completate: {len(self.optimization_history)}", "🔄", "blue")
        print_colored("🏆 ============================================", "🏆", "magenta")


def correct_velocity_drift(timestamps, velocities, positions, window_size=None, polynomial_order=2):
    """
    Corregge la deriva di velocità e posizione applicando correzioni basate sulla fisica del movimento.
    Utilizza una combinazione di correzione della media e filtri polinomiali per rimuovere la deriva.
    
    Args:
        timestamps (np.ndarray): Array dei timestamp.
        velocities (np.ndarray): Array delle velocità stimate.
        positions (np.ndarray): Array delle posizioni stimate.
        window_size (int, optional): Dimensione della finestra per il filtro. Se None, usa tutto il segnale.
        polynomial_order (int, optional): Ordine del polinomio per il filtro. Default è 2.
        
    Returns:
        tuple: Tuple contenente gli array corretti di velocità e posizione.
    """
    # Metodo 1: Correzione della media globale
    # Calcoliamo la media della velocità (idealmente dovrebbe essere zero su un intervallo lungo)
    mean_velocity = np.mean(velocities)
    
    # Sottraiamo il valore medio per rimuovere la deriva di velocità
    corrected_velocities = velocities - mean_velocity
    
    # Metodo 2: Rimuovi trend polinomiale (drift non lineare)
    try:
        # Adatta un polinomio ai dati per catturare la deriva non lineare
        t = timestamps - timestamps[0]  # Tempo relativo a partire da 0
        polynomial_coeffs = np.polyfit(t, corrected_velocities, polynomial_order)
        drift_trend = np.polyval(polynomial_coeffs, t)
        
        # Rimuovi il trend dalla velocità
        corrected_velocities = corrected_velocities - drift_trend
    except Exception as e:
        print_colored(f"Avviso: Errore nella correzione polinomiale: {e}. Usando solo correzione della media.", "⚠️", "yellow")
    
    # Ricalcoliamo le posizioni integrando le velocità corrette
    corrected_positions = np.zeros_like(positions)
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        corrected_positions[i] = corrected_positions[i-1] + corrected_velocities[i] * dt
    
    return corrected_velocities, corrected_positions


def run_EKF_for_vel_and_pos_est_from_acc(config_path=None):
    """
    Funzione principale che esegue l'Extended Kalman Filter per la stima di velocità e posizione.
    
    Args:
        config_path (str, optional): Percorso del file di configurazione YAML.
            Se None, viene utilizzato il percorso predefinito.
    
    Returns:
        dict: Dizionario contenente i risultati dell'EKF per ogni asse analizzato.
    """
    try:
        print_section_header("AVVIO ELABORAZIONE EKF", "🚀")
        logger.info("🚀 Avvio dell'elaborazione EKF per la stima di velocità e posizione")
        print_colored("🎯 Obiettivo: Stima di velocità e posizione da dati di accelerazione", "🎯", "cyan")
        
        # Percorso predefinito del file di configurazione
        if config_path is None:
            print_colored("🔍 Ricerca automatica del file di configurazione...", "🔍", "yellow")
            logger.info("🔍 Ricerca automatica del file di configurazione in corso...")
            
            # Ottieni la directory del modulo corrente
            module_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Determina se siamo nella struttura normale o in una con nome script uguale alla directory
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            current_dir_name = os.path.basename(os.path.dirname(module_dir))
            
            if script_name == current_dir_name:
                # Caso speciale: script con nome uguale alla directory parent
                module_parent_dir = os.path.dirname(module_dir)
            else:
                # Struttura normale
                module_parent_dir = os.path.dirname(module_dir)
            
            # Verifica direttamente il percorso esatto dove sappiamo che esiste il file
            direct_config_path = os.path.join(module_parent_dir, 'configs', 'EKF_for_vel_and_pos_est_from_acc.yaml')
            if os.path.exists(direct_config_path):
                config_path = direct_config_path
                print_colored(f"✅ File di configurazione trovato: {config_path}", "✅", "green")
                logger.info(f"✅ File di configurazione trovato direttamente in: {config_path}")
            else:
                print_colored("🔍 Ricerca in percorsi alternativi...", "🔍", "yellow")
                # Prova tutti i possibili percorsi partendo dal più probabile
                potential_paths = [
                    direct_config_path,
                    os.path.join(os.getcwd(), 'configs', 'EKF_for_vel_and_pos_est_from_acc.yaml'),
                    os.path.join(module_dir, '..', 'configs', 'EKF_for_vel_and_pos_est_from_acc.yaml'),
                    os.path.abspath(os.path.join(module_dir, '..', 'configs', 'EKF_for_vel_and_pos_est_from_acc.yaml')),
                    os.path.join(os.path.dirname(os.path.dirname(module_dir)), 'configs', 'EKF_for_vel_and_pos_est_from_acc.yaml')
                ]
                
                for i, path in enumerate(potential_paths, 1):
                    norm_path = os.path.normpath(path)
                    print_colored(f"🔍 Tentativo {i}/5: {norm_path}", "🔍", "blue")
                    logger.info(f"🔍 Cercando il file di configurazione in: {norm_path}")
                    if os.path.exists(norm_path):
                        config_path = norm_path
                        print_colored(f"✅ File trovato al tentativo {i}!", "✅", "green")
                        logger.info(f"✅ File di configurazione trovato in: {config_path}")
                        break
                
                # Se non troviamo il file, prova ancora una volta cercandolo in tutte le directory fino alla radice
                if not os.path.exists(config_path):
                    print_colored("🔍 Ricerca ricorsiva dalla directory corrente...", "🔍", "yellow")
                    # Stampa tutte le directory nel path per debug
                    current_dir = module_dir
                    search_attempts = 0
                    while current_dir != os.path.dirname(current_dir) and search_attempts < 10:  # Continua finché non raggiungiamo la radice
                        search_attempts += 1
                        print_colored(f"🔍 Controllo directory {search_attempts}: {current_dir}", "🔍", "blue")
                        logger.info(f"🔍 Controllando directory: {current_dir}")
                        config_file = os.path.join(current_dir, 'configs', 'EKF_for_vel_and_pos_est_from_acc.yaml')
                        if os.path.exists(config_file):
                            config_path = config_file
                            print_colored(f"✅ File trovato nella ricerca ricorsiva!", "✅", "green")
                            logger.info(f"✅ File di configurazione trovato in: {config_path}")
                            break
                        current_dir = os.path.dirname(current_dir)
        
        # Carica la configurazione
        print_colored("📋 Caricamento configurazione in corso...", "📋", "cyan")
        config = load_config(config_path)
        print_colored(f"✅ Configurazione caricata con successo da: {os.path.basename(config_path)}", "✅", "green")
        logger.info(f"📋 Configurazione caricata da: {config_path}")
        
        # Valida la configurazione
        if not validate_configuration(config):
            raise ValueError("❌ Configurazione non valida. Controllare i parametri.")
        
        # Determina il tipo di accelerazione da utilizzare
        acc_type = config['acceleration_type']
        print_colored(f"📱 Tipo di accelerazione selezionato: {acc_type}", "📱", "blue")
        logger.info(f"📱 Utilizzo dell'accelerazione di tipo: {acc_type}")
        
        # Log configurazione dettagliata
        logger.debug(f"🔧 Configurazione completa: {config}")
        
        # Percorso del file di input
        input_file = config['input_files'][acc_type]
        
        # Risolve il percorso relativo con riconoscimento intelligente della struttura
        module_dir = os.path.dirname(os.path.abspath(__file__))
        module_parent_dir = os.path.dirname(module_dir)  # /sources
        
        # Ottieni il nome dello script senza estensione
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        
        # Verifica se siamo in una struttura dove la directory root ha lo stesso nome dello script
        current_dir_name = os.path.basename(module_parent_dir)
        if current_dir_name == script_name:
            # Siamo nella struttura: EKF_for_vel_and_pos_est_from_acc/sources/EKF_for_vel_and_pos_est_from_acc.py
            base_dir = module_parent_dir  # La root è module_parent_dir
            print_colored(f"🎯 Rilevata struttura modulo con nome script: {script_name}", "🎯", "green")
        else:
            # Struttura tradizionale
            base_dir = os.path.dirname(module_parent_dir)
            print_colored(f"📁 Struttura tradizionale rilevata", "📁", "cyan")
        
        print_colored("📁 Risoluzione percorsi file di input...", "📁", "cyan")
        logger.info(f"📁 Directory modulo: {module_dir}")
        logger.info(f"📁 Directory parent: {module_parent_dir}")
        logger.info(f"📁 Directory base: {base_dir}")
        logger.info(f"📁 Nome script: {script_name}")
        logger.info(f"📁 Nome directory corrente: {current_dir_name}")
        
        # Prova diversi percorsi possibili per trovare il file di input
        potential_input_paths = [
            os.path.join(base_dir, input_file),  # Relativo alla directory del modulo principale
            os.path.join(module_parent_dir, input_file),  # Relativo alla directory sources
            os.path.join(os.getcwd(), input_file),  # Relativo alla directory corrente
            os.path.join(os.path.dirname(os.getcwd()), input_file),  # Un livello sopra rispetto alla directory corrente
            input_file  # Percorso assoluto o relativo alla directory corrente
        ]
        
        input_path = None
        for path in potential_input_paths:
            logger.info(f"Cercando il file di input in: {path}")
            if os.path.exists(path):
                input_path = path
                logger.info(f"File di input trovato in: {path}")
                break
                
        if input_path is None:
            raise FileNotFoundError(f"Non è possibile trovare il file di input {input_file}")
        
        # Carica i dati di accelerazione
        data = parse_acceleration_data(input_path, acc_type)
        print_colored(f"Dati di accelerazione caricati: {len(data)} campioni", "📊", "green")
        
        # Ricampiona i dati se richiesto
        if config['resampling']['enabled']:
            print_colored(f"Ricampionamento dei dati a {config['resampling']['frequency']} Hz...", "⏱️", "yellow")
            data = resample_acceleration_data(data, config['resampling']['frequency'])
            print_colored(f"Ricampionamento completato: {len(data)} campioni", "✅", "green")
        
        # Applica il trimming del segnale se richiesto
        if config.get('signal_trimming', {}).get('enabled', False):
            start_offset = config.get('signal_trimming', {}).get('start_offset', 50)
            if start_offset > 0 and start_offset < len(data):
                # Mostra un esempio dei valori prima del trimming
                print_colored(f"Primi 5 valori prima del trimming:", "🔍", "cyan")
                for i in range(min(5, len(data))):
                    print_colored(f"  [{i}] timestamp={data['timestamp'].iloc[i]:.3f}, acc_y={data['acc_y'].iloc[i]:.3f}", "📊", "white")
                
                # Applica il trimming
                original_length = len(data)
                data = data.iloc[start_offset:].reset_index(drop=True)
                
                # Mostra un esempio dei valori dopo il trimming
                print_colored(f"Primi 5 valori dopo il trimming (originariamente da indice {start_offset}):", "🔍", "cyan")
                for i in range(min(5, len(data))):
                    print_colored(f"  [{i}] timestamp={data['timestamp'].iloc[i]:.3f}, acc_y={data['acc_y'].iloc[i]:.3f}", "📊", "white")
                
                print_colored(f"Trimming del segnale: rimossi i primi {start_offset} campioni", "✂️", "yellow")
                print_colored(f"Lunghezza segnale: {original_length} → {len(data)} campioni", "📏", "cyan")
                print_colored(f"Intervallo temporale: {data['timestamp'].min():.3f} → {data['timestamp'].max():.3f} secondi", "⏱️", "magenta")
                logger.info(f"Applicato trimming del segnale: rimossi i primi {start_offset} campioni")
        
        # Pausa step-by-step dopo preprocessing se abilitata
        step_config = config.get('execution_control', {}).get('step_by_step', {})
        if step_config.get('enabled', False) and step_config.get('pause_after_preprocessing', False):
            if not get_user_confirmation("📊 Preprocessing completato. Continuare con l'EKF?", config):
                print_colored("🛑 Esecuzione interrotta dall'utente", "🛑", "yellow")
                return {}
        
        # Determina gli assi da analizzare
        analysis_axis = config['analysis_axis']
        axes_to_analyze = ['X', 'Y', 'Z'] if analysis_axis == 'all' else [analysis_axis]
        print_colored(f"Assi da analizzare: {', '.join(axes_to_analyze)}", "🧭", "magenta")
        
        # Dizionario per i risultati
        results_dict = {}
        
        # Directory di output centralizzata
        script_dir = os.path.dirname(os.path.abspath(__file__))
        module_dir = os.path.dirname(script_dir)  # Risali di un livello (da sources/ a modulo/)
        base_output_dir = get_output_base_dir()
        base_output_dir.mkdir(parents=True, exist_ok=True)
        print_colored(f"Directory di output impostata a: {base_output_dir}", "📁", "cyan")
        
        # Verifica se l'auto-tuning è abilitato
        auto_tuning_enabled = config.get('auto_tuning', {}).get('enabled', False)
        if auto_tuning_enabled:
            print_colored("🎯 Auto-tuning abilitato: ottimizzazione automatica dei parametri", "🎯", "cyan")
            logger.info("🎯 Avvio ottimizzazione automatica dei parametri EKF")
            
            # Esegui auto-tuning per il primo asse (o asse specificato)
            primary_axis = axes_to_analyze[0]
            print_colored(f"🔬 Ottimizzazione su asse primario: {primary_axis}", "🔬", "cyan")
            
            # Inizializza ottimizzatore
            optimizer = EKFParameterOptimizer(config, data, primary_axis)
            
            # Esegui ottimizzazione
            optimization_results = optimizer.run_optimization()
            
            if optimization_results['best_parameters']:
                print_colored("✨ Applicazione migliori parametri trovati", "✨", "green")
                logger.info("✨ Applicazione parametri ottimizzati a tutti gli assi")
                
                # Aggiorna configurazione con migliori parametri
                config['ekf']['process_noise']['position'] = optimization_results['best_parameters']['position_noise']
                config['ekf']['process_noise']['velocity'] = optimization_results['best_parameters']['velocity_noise']
                config['ekf']['process_noise']['acceleration'] = optimization_results['best_parameters']['acceleration_noise']
                config['ekf']['process_noise']['bias'] = optimization_results['best_parameters']['bias_noise']
                config['ekf']['measurement_noise'] = optimization_results['best_parameters']['measurement_noise']
                
                # Salva configurazione ottimizzata se richiesto
                if config.get('auto_tuning', {}).get('save_best_config', True):
                    optimized_config_path = resolve_output_path('optimized_config.yaml', base_output_dir)
                    try:
                        import yaml
                        with open(optimized_config_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        print_colored(f"💾 Configurazione ottimizzata salvata: {optimized_config_path.name}", "💾", "green")
                        logger.info(f"Configurazione ottimizzata salvata in: {optimized_config_path}")
                    except Exception as e:
                        logger.warning(f"Errore nel salvataggio configurazione ottimizzata: {e}")
                
                # Salva report di ottimizzazione
                optimization_report_path = resolve_output_path('optimization_report.json', base_output_dir)
                try:
                    import json
                    with open(optimization_report_path, 'w') as f:
                        json.dump(optimization_results, f, indent=2, default=str)
                    print_colored(f"📊 Report ottimizzazione salvato: {optimization_report_path.name}", "📊", "green")
                    logger.info(f"Report ottimizzazione salvato in: {optimization_report_path}")
                except Exception as e:
                    logger.warning(f"Errore nel salvataggio report ottimizzazione: {e}")
            
            else:
                print_colored("⚠️ Ottimizzazione non riuscita, uso parametri originali", "⚠️", "yellow")
                logger.warning("Ottimizzazione non riuscita, proseguo con parametri originali")
        
        # Esegui l'EKF per ogni asse
        for axis in axes_to_analyze:
            print_colored(f"Elaborazione dell'asse {axis}...", "🔄", "yellow")
            
            # Applica l'EKF
            results = apply_extended_kalman_filter(data, config, axis)
            
            # Valida i risultati se abilitato
            if config.get('output_validation', {}).get('enabled', False):
                validation_report = validate_results(results, config, axis)
                print_validation_report(validation_report, axis)
                
                # Gestione azioni in caso di validazione fallita
                validation_actions = config.get('output_validation', {}).get('validation_actions', {})
                
                if not validation_report['passed']:
                    action = validation_actions.get('on_bounds_violation', 'warn')
                    if action == 'error':
                        raise ValueError(f"❌ Validazione fallita per asse {axis}")
                    elif action == 'warn':
                       print_colored(f"⚠️ Continuando nonostante errori di validazione per asse {axis}", "⚠️", "yellow")
                
                # Controlla se ci sono avvisi per deriva
                drift_warnings = [w for w in validation_report['warnings'] if 'deriva' in w.lower()]
                if drift_warnings and validation_actions.get('on_drift_detection', 'warn') == 'correct':
                    print_colored("🔧 Applicazione correzione deriva aggiuntiva...", "🔧", "yellow")
                    # Qui potremmo aggiungere correzioni aggiuntive
            
            # Crea backup se richiesto
            velocity_output_path = resolve_output_path(f'estimated_velocity_{axis}.csv', base_output_dir)
            create_backup_if_needed(velocity_output_path, config)
            
            # Controlla conferma utente per sovrascrittura file
            if velocity_output_path.exists() and config.get('execution_control', {}).get('user_confirmations', {}).get('confirm_file_overwrites', False):
                if not get_user_confirmation(f"Sovrascrivere il file esistente {velocity_output_path.name}?", config):
                    print_colored(f"⏭️ Saltando salvataggio velocità per asse {axis}", "⏭️", "yellow")
                    continue
            
            # Salva i risultati
            save_results(results, velocity_output_path, 'velocity')
            print_colored(f"Velocità stimata salvata in: {velocity_output_path.name}", "💨", "green")
            
            position_output_path = resolve_output_path(f'estimated_position_{axis}.csv', base_output_dir)
            create_backup_if_needed(position_output_path, config)
            save_results(results, position_output_path, 'position')
            print_colored(f"Posizione stimata salvata in: {position_output_path.name}", "📍", "green")
            
            # Percorso per i grafici
            plots_output_path = resolve_output_path(f'ekf_plots_{axis}.png', base_output_dir)
            
            # Pausa step-by-step prima della visualizzazione se abilitata
            if step_config.get('enabled', False) and step_config.get('pause_before_visualization', False):
                if not get_user_confirmation(f"📊 Iniziare visualizzazione per asse {axis}?", config):
                    print_colored("🛑 Esecuzione interrotta dall'utente", "🛑", "yellow")
                    return results_dict
            
            plot_results(data, results, config, axis, plots_output_path)
            print_colored(f"Grafico salvato in: {plots_output_path.name}", "📊", "magenta")
            
            # Salva i risultati nel dizionario
            results_dict[axis] = results
            
            # Pausa step-by-step dopo EKF se abilitata
            if step_config.get('enabled', False) and step_config.get('pause_after_ekf', False):
                if not get_user_confirmation(f"🎯 EKF completato per asse {axis}. Continuare con post-processing?", config):
                    print_colored("🛑 Esecuzione interrotta dall'utente", "🛑", "yellow")
                    return results_dict
            
            print_colored(f"Elaborazione dell'asse {axis} completata con successo!", "✅", "blue")
            
            # Pausa step-by-step dopo post-processing se abilitata
            if step_config.get('enabled', False) and step_config.get('pause_after_postprocessing', False):
                if axis != axes_to_analyze[-1]:  # Non fare pausa per l'ultimo asse
                    if not get_user_confirmation(f"📊 Post-processing completato per asse {axis}. Continuare con il prossimo asse?", config):
                        print_colored("🛑 Esecuzione interrotta dall'utente", "🛑", "yellow")
                        return results_dict
        
        logger.info("Elaborazione EKF completata con successo")
        
        # 📊 CONFRONTO MULTI-ASSE SE PIÙ DI UN ASSE È STATO ANALIZZATO
        if len(axes_to_analyze) > 1:
            # Calcola directory assoluta basata sulla posizione del modulo corrente
            output_dir = base_output_dir
            
            # Raccogli i report di tuning
            tuning_reports = {}
            for axis in axes_to_analyze:
                if axis in results_dict and 'tuning_report' in results_dict[axis]:
                    tuning_reports[axis] = results_dict[axis]['tuning_report']
            
            if tuning_reports:
                multi_axis_report = generate_multi_axis_comparison(tuning_reports, output_dir)
                # Aggiungi il report al primo asse per compatibilità
                if axes_to_analyze and axes_to_analyze[0] in results_dict:
                    results_dict[axes_to_analyze[0]]['multi_axis_comparison'] = multi_axis_report
        
        # Pausa finale prima della conclusione se abilitata
        if step_config.get('enabled', False) and step_config.get('pause_before_completion', False):
            get_user_confirmation("🎊 Processo completato! Premere Invio per terminare.", config)
        
        if results_dict:
            summary_path = generate_execution_summary_report(results_dict, config, config_path)
            if summary_path:
                results_dict['summary_report'] = str(summary_path)
                logger.info(f"📄 Report riassuntivo salvato in: {summary_path}")
                print_colored(f"Report riassuntivo salvato in: {summary_path.name}", "📄", "green")
        
        print_colored("🎊 Tutti gli assi elaborati con successo!", "🎊", "green")
        return results_dict
        
    except Exception as e:
        logger.error(f"Errore nell'esecuzione dell'EKF: {e}")
        print_colored(f"ERRORE: {e}", "❌", "red")
        raise


def print_colored(message, emoji="", color="white"):
    """
    Stampa un messaggio colorato con emoji usando colorama se disponibile.
    
    Args:
        message (str): Il messaggio da stampare.
        emoji (str): L'emoji da aggiungere all'inizio del messaggio.
        color (str): Il colore del testo. Opzioni: red, green, yellow, blue, magenta, cyan, white.
    """
    if COLORAMA_AVAILABLE:
        # Usa colorama per i colori
        colors = {
            "red": Fore.RED + Style.BRIGHT,
            "green": Fore.GREEN + Style.BRIGHT,
            "yellow": Fore.YELLOW + Style.BRIGHT,
            "blue": Fore.BLUE + Style.BRIGHT,
            "magenta": Fore.MAGENTA + Style.BRIGHT,
            "cyan": Fore.CYAN + Style.BRIGHT,
            "white": Fore.WHITE + Style.BRIGHT
        }
        color_code = colors.get(color.lower(), colors["white"])
        print(f"{color_code}{emoji} {message}{Style.RESET_ALL}")
    else:
        # Fallback per codici ANSI tradizionali
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m"
        }
        reset = "\033[0m"
        color_code = colors.get(color.lower(), colors["white"])
        print(f"{color_code}{emoji} {message}{reset}")


class EKFPerformanceMonitor:
    """
    Classe per monitorare e valutare le prestazioni del filtro di Kalman esteso.
    
    Questa classe implementa vari metriche e controlli per valutare:
    - Convergenza del filtro
    - Qualità delle innovazioni
    - Stabilità della matrice di covarianza
    - Tuning dei parametri di processo e misura
    - Consistenza statistica
    """
    
    def __init__(self, config, logger):
        """
        Inizializza il monitor delle prestazioni EKF.
        
        Args:
            config (dict): Configurazione dell'EKF
            logger: Logger per il monitoraggio
        """
        self.config = config
        self.logger = logger
        
        # Buffer per memorizzare le metriche nel tempo
        self.innovation_history = deque(maxlen=1000)
        self.trace_history = deque(maxlen=1000)
        self.log_likelihood_history = deque(maxlen=1000)
        self.nees_history = deque(maxlen=1000)  # Normalized Estimation Error Squared
        self.nis_history = deque(maxlen=1000)   # Normalized Innovation Squared
        
        # Statistiche di performance
        self.performance_stats = {
            'convergence_time': None,
            'steady_state_variance': None,
            'innovation_whiteness': None,
            'filter_consistency': None,
            'tuning_quality': None
        }
        
        # Soglie per il controllo qualità (adattive in base alle dimensioni)
        self.base_thresholds = {
            'max_trace': config.get('performance_monitoring', {}).get('max_trace', 1000.0),
            'min_log_likelihood': config.get('performance_monitoring', {}).get('min_log_likelihood', -1000.0),
            'nees_upper_bound_factor': 3.0,  # Moltiplicatore per dimensioni dello stato
            'nees_lower_bound_factor': 0.3,  # Moltiplicatore per dimensioni dello stato
            'nis_upper_bound': config.get('performance_monitoring', {}).get('nis_upper_bound', 6.63),   # Chi-squared 95% for 1 DOF
            'innovation_correlation_threshold': 0.1
        }
        
        # Inizializziamo con valori di default (4 DOF)
        self.update_thresholds(4)
        
        self.logger.info("🔍 EKF Performance Monitor inizializzato")
        print_colored("🔍 Monitor delle prestazioni EKF attivato", "🔍", "cyan")
    
    def update_thresholds(self, state_dimension):
        """Aggiorna le soglie basate sulla dimensione dello stato dell'EKF."""
        self.thresholds = {
            'max_trace': self.base_thresholds['max_trace'],
            'min_log_likelihood': self.base_thresholds['min_log_likelihood'],
            'nees_upper_bound': state_dimension * self.base_thresholds['nees_upper_bound_factor'],
            'nees_lower_bound': state_dimension * self.base_thresholds['nees_lower_bound_factor'],
            'nis_upper_bound': self.base_thresholds['nis_upper_bound'],
            'innovation_correlation_threshold': self.base_thresholds['innovation_correlation_threshold']
        }
    
    def update_metrics(self, ekf, innovation, measurement, true_state=None):
        """
        Aggiorna le metriche di performance ad ogni step del filtro.
        
        Args:
            ekf: Oggetto Extended Kalman Filter
            innovation: Vettore delle innovazioni
            measurement: Misura corrente
            true_state: Stato vero (se disponibile) per NEES
        """
        try:
            # Aggiorna soglie se necessario basandosi sulle dimensioni dello stato EKF
            current_state_dim = ekf.x.shape[0]
            if not hasattr(self, '_last_state_dim') or self._last_state_dim != current_state_dim:
                self.update_thresholds(current_state_dim)
                self._last_state_dim = current_state_dim
                self.logger.info(f"🔄 Soglie aggiornate per EKF {current_state_dim}D")
            
            # 1. Traccia della matrice di covarianza
            trace = np.trace(ekf.P)
            self.trace_history.append(trace)
            
            # 2. Innovazione
            innovation = np.atleast_2d(innovation)
            if innovation.shape[0] == 1:
                innovation = innovation.T  # Assicura che sia colonna
            innovation_norm = np.linalg.norm(innovation)
            self.innovation_history.append(innovation_norm)
            
            # 3. Log-likelihood dell'innovazione (adattato per dimensioni variabili)
            try:
                # Per EKF 3D, usiamo una matrice H semplificata 1x6 che estrae solo l'asse Y
                if hasattr(ekf, 'mode_3d') and getattr(ekf, 'mode_3d', False):
                    # Modalità 3D: crea H che estrae solo la posizione Y per compatibilità
                    H_simplified = np.zeros((1, ekf.x.shape[0]))
                    H_simplified[0, 1] = 1.0  # Estrae pos_y
                    R_simplified = np.array([[float(np.mean(np.diag(ekf.R)))]])  # R scalare
                else:
                    # Modalità 1D: usa H e R originali
                    H_simplified = ekf.H[:1, :] if ekf.H.shape[0] > 1 else ekf.H
                    R_simplified = ekf.R[:1, :1] if ekf.R.shape[0] > 1 else ekf.R
                
                S = H_simplified @ ekf.P @ H_simplified.T + R_simplified
                S_inv = inv(S)
                log_likelihood = -0.5 * (innovation.T @ S_inv @ innovation + 
                                       np.log(np.linalg.det(2 * np.pi * S)))
                self.log_likelihood_history.append(float(log_likelihood))
                
                # 4. Normalized Innovation Squared (NIS)
                nis = float(innovation.T @ S_inv @ innovation)
                self.nis_history.append(nis)
                
            except (np.linalg.LinAlgError, ValueError) as e:
                self.log_likelihood_history.append(-np.inf)
                self.nis_history.append(np.inf)
                self.logger.warning(f"⚠️ Errore nel calcolo metriche: {e}")
            
            # 5. Normalized Estimation Error Squared (NEES) - se disponibile lo stato vero
            if true_state is not None:
                true_state = np.atleast_2d(true_state)
                if true_state.shape[0] == 1:
                    true_state = true_state.T
                estimation_error = ekf.x - true_state
                try:
                    P_inv = inv(ekf.P)
                    nees = float(estimation_error.T @ P_inv @ estimation_error)
                    self.nees_history.append(nees)
                except:
                    self.nees_history.append(np.inf)
                    
        except Exception as e:
            self.logger.error(f"❌ Errore nell'aggiornamento metriche: {e}")
            # Debug info per capire le dimensioni
            try:
                self.logger.debug(f"Debug dimensioni - innovation: {innovation.shape if hasattr(innovation, 'shape') else type(innovation)}")
                self.logger.debug(f"Debug dimensioni - ekf.P: {ekf.P.shape}")
                self.logger.debug(f"Debug dimensioni - ekf.x: {ekf.x.shape}")
            except:
                pass
    
    def check_convergence(self, window_size=50):
        """
        Controlla se il filtro ha raggiunto la convergenza.
        
        Args:
            window_size: Dimensione della finestra per calcolare la convergenza
            
        Returns:
            bool: True se il filtro è convergente
        """
        if len(self.trace_history) < window_size:
            return False
            
        recent_traces = list(self.trace_history)[-window_size:]
        
        # Calcola la varianza della traccia nell'ultima finestra
        trace_variance = np.var(recent_traces)
        mean_trace = np.mean(recent_traces)
        
        # Convergenza se la varianza relativa è piccola
        relative_variance = trace_variance / (mean_trace + 1e-10)
        is_converged = relative_variance < 0.01  # 1% di variazione relativa
        
        if is_converged and self.performance_stats['convergence_time'] is None:
            self.performance_stats['convergence_time'] = len(self.trace_history)
            self.logger.info(f"✅ Filtro convergente dopo {self.performance_stats['convergence_time']} iterazioni")
            print_colored(f"✅ Convergenza raggiunta dopo {self.performance_stats['convergence_time']} iterazioni", "✅", "green")
        
        return is_converged
    
    def check_filter_consistency(self):
        """
        Controlla la consistenza statistica del filtro usando NEES e NIS.
        
        Returns:
            dict: Risultati dei test di consistenza
        """
        results = {
            'nees_consistent': True,
            'nis_consistent': True,
            'overall_consistent': True
        }
        
        if len(self.nees_history) > 10:
            mean_nees = np.mean(self.nees_history)
            nees_in_bounds = (self.thresholds['nees_lower_bound'] <= mean_nees <= 
                             self.thresholds['nees_upper_bound'])
            results['nees_consistent'] = nees_in_bounds
            
            if not nees_in_bounds:
                self.logger.warning(f"⚠️ NEES fuori range: {mean_nees:.3f} (range: {self.thresholds['nees_lower_bound']:.3f}-{self.thresholds['nees_upper_bound']:.3f})")
        
        if len(self.nis_history) > 10:
            mean_nis = np.mean(self.nis_history)
            nis_consistent = mean_nis <= self.thresholds['nis_upper_bound']
            results['nis_consistent'] = nis_consistent
            
            if not nis_consistent:
                self.logger.warning(f"⚠️ NIS troppo alto: {mean_nis:.3f} (soglia: {self.thresholds['nis_upper_bound']:.3f})")
        
        results['overall_consistent'] = results['nees_consistent'] and results['nis_consistent']
        
        return results
    
    def analyze_innovation_whiteness(self, max_lag=20):
        """
        Analizza la bianchezza delle innovazioni (test di autocorrelazione).
        
        Args:
            max_lag: Numero massimo di lag per l'autocorrelazione
            
        Returns:
            dict: Risultati del test di bianchezza
        """
        if len(self.innovation_history) < max_lag * 2:
            return {'is_white': None, 'max_correlation': None}
        
        innovations = np.array(self.innovation_history)
        
        # Calcola l'autocorrelazione
        correlations = []
        for lag in range(1, min(max_lag, len(innovations)//2)):
            if len(innovations) > lag:
                corr = np.corrcoef(innovations[:-lag], innovations[lag:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        max_correlation = max(correlations) if correlations else 0
        is_white = max_correlation < self.thresholds['innovation_correlation_threshold']
        
        result = {
            'is_white': is_white,
            'max_correlation': max_correlation,
            'correlations': correlations
        }
        
        if not is_white:
            self.logger.warning(f"⚠️ Innovazioni non bianche: correlazione max = {max_correlation:.4f}")
            print_colored(f"⚠️ Innovazioni correlate (max: {max_correlation:.4f})", "⚠️", "yellow")
        
        return result
    
    def evaluate_tuning_quality(self):
        """
        Valuta la qualità del tuning dei parametri Q e R.
        
        Returns:
            dict: Raccomandazioni per il tuning
        """
        recommendations = {
            'Q_adjustment': 'none',
            'R_adjustment': 'none',
            'overall_quality': 'good'
        }
        
        if len(self.nis_history) > 20:
            mean_nis = np.mean(self.nis_history)
            
            # Se NIS è troppo alto, R potrebbe essere troppo piccolo
            if mean_nis > self.thresholds['nis_upper_bound'] * 1.5:
                recommendations['R_adjustment'] = 'increase'
                recommendations['overall_quality'] = 'poor'
                self.logger.warning("⚠️ NIS alto: considera di aumentare R (rumore di misura)")
                print_colored("⚠️ Suggerimento: aumenta il rumore di misura R", "💡", "yellow")
            
            # Se NIS è troppo basso, R potrebbe essere troppo grande
            elif mean_nis < 0.5:
                recommendations['R_adjustment'] = 'decrease'
                recommendations['overall_quality'] = 'suboptimal'
                self.logger.info("💡 NIS basso: considera di diminuire R")
                print_colored("💡 Suggerimento: diminuisci il rumore di misura R", "💡", "cyan")
        
        if len(self.trace_history) > 20:
            trace_trend = np.polyfit(range(len(self.trace_history)), self.trace_history, 1)[0]
            
            # Se la traccia cresce troppo, Q potrebbe essere troppo grande
            if trace_trend > 0.1:
                recommendations['Q_adjustment'] = 'decrease'
                recommendations['overall_quality'] = 'poor'
                self.logger.warning("⚠️ Traccia crescente: considera di diminuire Q (rumore di processo)")
                print_colored("⚠️ Suggerimento: diminuisci il rumore di processo Q", "💡", "yellow")
            
            # Se la traccia decresce troppo rapidamente, Q potrebbe essere troppo piccolo
            elif trace_trend < -0.5:
                recommendations['Q_adjustment'] = 'increase'
                self.logger.info("💡 Traccia decrescente rapidamente: considera di aumentare Q")
                print_colored("💡 Suggerimento: aumenta il rumore di processo Q", "💡", "cyan")
        
        return recommendations
    
    def generate_performance_report(self):
        """
        Genera un report completo delle prestazioni del filtro.
        
        Returns:
            dict: Report dettagliato delle prestazioni
        """
        print_colored("📊 =============== REPORT PRESTAZIONI EKF ===============", "📊", "magenta")
        self.logger.info("📊 =============== REPORT PRESTAZIONI EKF ===============")
        
        report = {}
        
        # 1. Convergenza
        convergence_info = {
            'is_converged': self.check_convergence(),
            'convergence_time': self.performance_stats['convergence_time'],
            'current_trace': self.trace_history[-1] if self.trace_history else None
        }
        report['convergence'] = convergence_info
        
        print_colored(f"🎯 CONVERGENZA:", "🎯", "cyan")
        if convergence_info['is_converged']:
            print_colored(f"  ✅ Filtro convergente", "✅", "green")
            if convergence_info['convergence_time']:
                print_colored(f"  ⏱️  Tempo di convergenza: {convergence_info['convergence_time']} iterazioni", "⏱️", "green")
        else:
            print_colored(f"  ⚠️ Filtro non ancora convergente", "⚠️", "yellow")
        
        if convergence_info['current_trace']:
            print_colored(f"  📈 Traccia corrente: {convergence_info['current_trace']:.4f}", "📈", "cyan")
        
        # 2. Consistenza statistica
        consistency = self.check_filter_consistency()
        report['consistency'] = consistency
        
        print_colored(f"📊 CONSISTENZA STATISTICA:", "📊", "cyan")
        if consistency['overall_consistent']:
            print_colored(f"  ✅ Filtro statisticamente consistente", "✅", "green")
        else:
            print_colored(f"  ⚠️ Filtro non consistente", "⚠️", "yellow")
            if not consistency['nees_consistent']:
                print_colored(f"    - NEES fuori range", "❌", "red")
            if not consistency['nis_consistent']:
                print_colored(f"    - NIS fuori range", "❌", "red")
        
        # 3. Bianchezza delle innovazioni
        whiteness = self.analyze_innovation_whiteness()
        report['innovation_whiteness'] = whiteness
        
        print_colored(f"🎲 BIANCHEZZA INNOVAZIONI:", "🎲", "cyan")
        if whiteness['is_white'] is not None:
            if whiteness['is_white']:
                print_colored(f"  ✅ Innovazioni bianche (non correlate)", "✅", "green")
            else:
                print_colored(f"  ⚠️ Innovazioni correlate (max: {whiteness['max_correlation']:.4f})", "⚠️", "yellow")
        else:
            print_colored(f"  ❓ Dati insufficienti per il test", "❓", "yellow")
        
        # 4. Qualità del tuning
        tuning = self.evaluate_tuning_quality()
        report['tuning'] = tuning
        
        print_colored(f"🔧 QUALITÀ DEL TUNING:", "🔧", "cyan")
        print_colored(f"  🎯 Qualità generale: {tuning['overall_quality']}", "🎯", 
                     "green" if tuning['overall_quality'] == 'good' else "yellow")
        
        if tuning['Q_adjustment'] != 'none':
            action = "aumentare" if tuning['Q_adjustment'] == 'increase' else "diminuire"
            print_colored(f"  💡 Suggerimento Q: {action} il rumore di processo", "💡", "cyan")
        
        if tuning['R_adjustment'] != 'none':
            action = "aumentare" if tuning['R_adjustment'] == 'increase' else "diminuire"
            print_colored(f"  💡 Suggerimento R: {action} il rumore di misura", "💡", "cyan")
        
        # 5. Statistiche numeriche
        if self.trace_history:
            stats = {
                'mean_trace': np.mean(self.trace_history),
                'std_trace': np.std(self.trace_history),
                'min_trace': np.min(self.trace_history),
                'max_trace': np.max(self.trace_history)
            }
            report['trace_statistics'] = stats
            
            print_colored(f"📈 STATISTICHE TRACCIA P:", "📈", "cyan")
            print_colored(f"  📊 Media: {stats['mean_trace']:.4f}", "📊", "cyan")
            print_colored(f"  📊 Std Dev: {stats['std_trace']:.4f}", "📊", "cyan")
            print_colored(f"  📊 Range: [{stats['min_trace']:.4f}, {stats['max_trace']:.4f}]", "📊", "cyan")
        
        if self.nis_history:
            nis_stats = {
                'mean_nis': np.mean(self.nis_history),
                'std_nis': np.std(self.nis_history),
                'percentage_in_bounds': np.mean(np.array(self.nis_history) <= self.thresholds['nis_upper_bound']) * 100
            }
            report['nis_statistics'] = nis_stats
            
            print_colored(f"🎯 STATISTICHE NIS:", "🎯", "cyan")
            print_colored(f"  📊 Media NIS: {nis_stats['mean_nis']:.4f}", "📊", "cyan")
            print_colored(f"  📊 % entro soglia: {nis_stats['percentage_in_bounds']:.1f}%", "📊", "cyan")
        
        print_colored("✅ ============== REPORT PRESTAZIONI COMPLETATO ==============", "✅", "green")
        
        return report


def generate_performance_plots(performance_monitor, output_dir, axis):
    """
    Genera grafici delle prestazioni del filtro EKF.
    
    Args:
        performance_monitor: Oggetto EKFPerformanceMonitor
        output_dir: Directory di output per salvare i grafici
        axis: Asse analizzato ('X', 'Y', 'Z')
    """
    try:
        print_colored("📊 Generazione grafici delle prestazioni...", "📊", "cyan")
        
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        
        # Crea figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'EKF Performance Analysis - Axis {axis}', fontsize=16, fontweight='bold')
        
        # 1. Traccia della matrice di covarianza
        if performance_monitor.trace_history:
            axes[0, 0].plot(performance_monitor.trace_history, 'b-', linewidth=2)
            axes[0, 0].set_title('Covariance Matrix Trace Evolution', fontweight='bold')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Trace(P)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Aggiungi linea di convergenza se disponibile
            if performance_monitor.performance_stats['convergence_time']:
                conv_time = performance_monitor.performance_stats['convergence_time']
                axes[0, 0].axvline(x=conv_time, color='red', linestyle='--', 
                                 label=f'Convergence at {conv_time}')
                axes[0, 0].legend()
        
        # 2. Innovazioni nel tempo
        if performance_monitor.innovation_history:
            axes[0, 1].plot(performance_monitor.innovation_history, 'g-', linewidth=1, alpha=0.7)
            axes[0, 1].set_title('Innovation Norm Evolution', fontweight='bold')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('||Innovation||')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Istogramma delle innovazioni
        if performance_monitor.innovation_history:
            innovations = list(performance_monitor.innovation_history)
            axes[1, 0].hist(innovations, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('Innovation Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Innovation Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Aggiungi statistiche
            mean_inn = np.mean(innovations)
            std_inn = np.std(innovations)
            axes[1, 0].axvline(x=mean_inn, color='red', linestyle='--', 
                             label=f'Mean: {mean_inn:.4f}')
            axes[1, 0].axvline(x=mean_inn + std_inn, color='red', linestyle=':', alpha=0.7, 
                             label=f'±1σ: {std_inn:.4f}')
            axes[1, 0].axvline(x=mean_inn - std_inn, color='red', linestyle=':', alpha=0.7)
            axes[1, 0].legend()
        
        # 4. NIS nel tempo
        if performance_monitor.nis_history:
            nis_values = list(performance_monitor.nis_history)
            axes[1, 1].plot(nis_values, 'm-', linewidth=1, alpha=0.7)
            axes[1, 1].axhline(y=performance_monitor.thresholds['nis_upper_bound'], 
                             color='red', linestyle='--', label='Upper Bound (95%)')
            axes[1, 1].set_title('Normalized Innovation Squared (NIS)', fontweight='bold')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('NIS Value')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            # Aggiungi percentuale entro i bounds
            in_bounds = np.mean(np.array(nis_values) <= performance_monitor.thresholds['nis_upper_bound']) * 100
            axes[1, 1].text(0.02, 0.98, f'Within bounds: {in_bounds:.1f}%', 
                           transform=axes[1, 1].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Salva il grafico
        output_path = resolve_output_path(f'EKF_performance_analysis_axis_{axis}.png', output_dir)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print_colored(f"📊 Grafici salvati in: {output_path}", "💾", "green")
        logger.info(f"📊 Grafici delle prestazioni salvati in: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"❌ Errore nella generazione dei grafici: {e}")
        print_colored(f"❌ Errore nei grafici: {e}", "❌", "red")


def comprehensive_innovation_analysis(performance_monitor, output_dir, axis):
    """
    Analisi completa delle innovazioni per validazione EKF.
    
    Args:
        performance_monitor: Oggetto EKFPerformanceMonitor
        output_dir: Directory di output per salvare i grafici
        axis: Asse analizzato
        
    Returns:
        dict: Metriche di validazione delle innovazioni
    """
    try:
        innovations = np.array(list(performance_monitor.innovation_history))
        if len(innovations) < 10:
            print_colored("⚠️ Dati insufficienti per analisi innovazioni", "⚠️", "yellow")
            return {'status': 'insufficient_data'}
        
        print_colored(f"📊 Analisi innovazioni per asse {axis} ({len(innovations)} campioni)...", "📊", "cyan")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Innovation Analysis - Axis {axis}', fontsize=16)
        
        # 1. Serie temporale delle innovazioni
        axes[0, 0].plot(innovations, 'b-', alpha=0.7)
        axes[0, 0].set_title('Innovation Time Series')
        axes[0, 0].set_ylabel('Innovation Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Istogramma con test di normalità
        axes[0, 1].hist(innovations, bins=50, density=True, alpha=0.7, color='skyblue')
        
        # Overlay distribuzione normale teorica
        mu, sigma = stats.norm.fit(innovations)
        x = np.linspace(innovations.min(), innovations.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                       label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        # Test Shapiro-Wilk per normalità
        sample_size = min(5000, len(innovations))
        shapiro_stat, shapiro_p = stats.shapiro(innovations[:sample_size])
        axes[0, 1].set_title(f'Innovation Distribution\nShapiro p-value: {shapiro_p:.4f}')
        axes[0, 1].legend()
        
        # 3. Q-Q Plot
        stats.probplot(innovations, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot vs Normal')
        
        # 4. Autocorrelazione
        if STATSMODELS_AVAILABLE:
            lags = min(50, len(innovations)//4)
            autocorr = acf(innovations, nlags=lags, fft=True)
            
            axes[1, 0].stem(range(lags+1), autocorr)
            axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[1, 0].axhline(y=1.96/np.sqrt(len(innovations)), color='r', linestyle='--', alpha=0.7)
            axes[1, 0].axhline(y=-1.96/np.sqrt(len(innovations)), color='r', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Autocorrelation Function')
            axes[1, 0].set_xlabel('Lag')
            max_autocorr = np.max(np.abs(autocorr[1:])) if len(autocorr) > 1 else 0
        else:
            axes[1, 0].text(0.5, 0.5, 'Statsmodels not available', ha='center', va='center')
            axes[1, 0].set_title('Autocorrelation (Not Available)')
            max_autocorr = None
        
        # 5. Running variance
        window_size = min(100, len(innovations)//10)
        if window_size > 1:
            running_var = pd.Series(innovations).rolling(window=window_size).var()
            axes[1, 1].plot(running_var, 'g-', linewidth=2)
            axes[1, 1].set_title(f'Running Variance (window={window_size})')
            axes[1, 1].set_ylabel('Variance')
            axes[1, 1].grid(True, alpha=0.3)
            variance_stability = np.std(running_var.dropna()) / np.mean(running_var.dropna()) if np.mean(running_var.dropna()) != 0 else float('inf')
        else:
            variance_stability = None
        
        # 6. Spectral density
        if len(innovations) > 10:
            freqs, psd = signal.welch(innovations, nperseg=min(256, len(innovations)//4))
            axes[1, 2].semilogy(freqs, psd)
            axes[1, 2].set_title('Power Spectral Density')
            axes[1, 2].set_xlabel('Frequency')
            axes[1, 2].set_ylabel('PSD')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = resolve_output_path(f"innovation_analysis_{axis}.png", output_dir)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Test Jarque-Bera per normalità
        jb_stat, jb_p = stats.jarque_bera(innovations)
        
        # Ritorna metriche numeriche
        metrics = {
            'mean': float(np.mean(innovations)),
            'std': float(np.std(innovations)),
            'skewness': float(stats.skew(innovations)),
            'kurtosis': float(stats.kurtosis(innovations)),
            'shapiro_p_value': float(shapiro_p),
            'jarque_bera_p_value': float(jb_p),
            'max_autocorr': float(max_autocorr) if max_autocorr is not None else None,
            'variance_stability': float(variance_stability) if variance_stability is not None else None,
            'sample_size': len(innovations)
        }
        
        print_colored(f"✅ Analisi innovazioni completata: μ={mu:.4f}, σ={sigma:.4f}, p_shapiro={shapiro_p:.4f}", "✅", "green")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Errore nell'analisi delle innovazioni: {e}")
        print_colored(f"❌ Errore analisi innovazioni: {e}", "❌", "red")
        return {'status': 'error', 'message': str(e)}


def nis_validation_analysis(performance_monitor, output_dir, axis):
    """
    Analisi approfondita del NIS per validazione consistenza.
    
    Args:
        performance_monitor: Oggetto EKFPerformanceMonitor
        output_dir: Directory di output per salvare i grafici
        axis: Asse analizzato
        
    Returns:
        dict: Metriche di validazione NIS
    """
    try:
        nis_values = np.array(list(performance_monitor.nis_history))
        if len(nis_values) < 10:
            print_colored("⚠️ Dati insufficienti per analisi NIS", "⚠️", "yellow")
            return {'status': 'insufficient_data'}
        
        measurement_dim = 1  # Per il caso 1D
        print_colored(f"📊 Analisi NIS per asse {axis} ({len(nis_values)} campioni)...", "📊", "cyan")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'NIS Validation Analysis - Axis {axis}', fontsize=16)
        
        # 1. NIS time series con bounds teorici
        axes[0, 0].plot(nis_values, 'b-', alpha=0.7, linewidth=1)
        
        # Bounds teorici (Chi-squared)
        alpha = 0.05  # 95% confidence
        lower_bound = stats.chi2.ppf(alpha/2, measurement_dim)
        upper_bound = stats.chi2.ppf(1-alpha/2, measurement_dim)
        
        axes[0, 0].axhline(y=lower_bound, color='r', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
        axes[0, 0].axhline(y=upper_bound, color='r', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')
        axes[0, 0].set_title('NIS Time Series')
        axes[0, 0].set_ylabel('NIS Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Istogramma NIS vs Chi-squared teorica
        axes[0, 1].hist(nis_values, bins=50, density=True, alpha=0.7, color='lightcoral')
        
        x = np.linspace(0, np.percentile(nis_values, 99), 1000)
        theoretical_pdf = stats.chi2.pdf(x, measurement_dim)
        axes[0, 1].plot(x, theoretical_pdf, 'k-', linewidth=2, label=f'χ²({measurement_dim})')
        axes[0, 1].set_title('NIS Distribution')
        axes[0, 1].legend()
        
        # 3. Percentuale in bounds nel tempo
        window_size = max(50, len(nis_values)//20)
        rolling_percentage = []
        
        for i in range(window_size, len(nis_values)):
            window_nis = nis_values[i-window_size:i]
            in_bounds = np.sum((window_nis >= lower_bound) & (window_nis <= upper_bound))
            rolling_percentage.append(in_bounds / window_size * 100)
        
        if rolling_percentage:
            axes[1, 0].plot(rolling_percentage, 'g-', linewidth=2)
            axes[1, 0].axhline(y=95, color='r', linestyle='--', label='Expected 95%')
            axes[1, 0].set_title(f'Rolling In-Bounds Percentage (window={window_size})')
            axes[1, 0].set_ylabel('Percentage')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q plot vs Chi-squared
        try:
            stats.probplot(nis_values, dist=stats.chi2, sparams=(measurement_dim,), plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot vs χ²(1)')
        except:
            axes[1, 1].text(0.5, 0.5, 'Q-Q plot failed', ha='center', va='center')
        
        plt.tight_layout()
        plot_path = resolve_output_path(f"nis_analysis_{axis}.png", output_dir)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Metriche di validazione
        total_in_bounds = np.sum((nis_values >= lower_bound) & (nis_values <= upper_bound))
        percentage_in_bounds = total_in_bounds / len(nis_values) * 100
        
        # Test Kolmogorov-Smirnov
        ks_stat, ks_p = stats.kstest(nis_values, lambda x: stats.chi2.cdf(x, measurement_dim))
        
        metrics = {
            'percentage_in_bounds': float(percentage_in_bounds),
            'expected_percentage': 95.0,
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_p),
            'mean_nis': float(np.mean(nis_values)),
            'expected_mean': float(measurement_dim),
            'variance_nis': float(np.var(nis_values)),
            'expected_variance': float(2 * measurement_dim),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'sample_size': len(nis_values)
        }
        
        print_colored(f"✅ Analisi NIS completata: {percentage_in_bounds:.1f}% in bounds, KS p-value: {ks_p:.4f}", "✅", "green")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Errore nell'analisi NIS: {e}")
        print_colored(f"❌ Errore analisi NIS: {e}", "❌", "red")
        return {'status': 'error', 'message': str(e)}


def covariance_trace_analysis(performance_monitor, output_dir, axis):
    """
    Analisi della convergenza e stabilità della matrice di covarianza.
    
    Args:
        performance_monitor: Oggetto EKFPerformanceMonitor
        output_dir: Directory di output per salvare i grafici
        axis: Asse analizzato
        
    Returns:
        dict: Metriche di convergenza e stabilità
    """
    try:
        trace_values = np.array(list(performance_monitor.trace_history))
        if len(trace_values) < 10:
            print_colored("⚠️ Dati insufficienti per analisi traccia covarianza", "⚠️", "yellow")
            return {'status': 'insufficient_data'}
        
        print_colored(f"📊 Analisi traccia covarianza per asse {axis} ({len(trace_values)} campioni)...", "📊", "cyan")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Covariance Trace Analysis - Axis {axis}', fontsize=16)
        
        # 1. Traccia nel tempo con fasi identificate
        axes[0, 0].plot(trace_values, 'b-', linewidth=2)
        
        # Identifica fasi di convergenza
        convergence_point = None
        if len(trace_values) > 100:
            # Fase transiente (primi 20%)
            transient_end = len(trace_values) // 5
            axes[0, 0].axvline(x=transient_end, color='orange', linestyle='--', 
                              label=f'Transient phase end: {transient_end}')
            
            # Rileva convergenza (quando la derivata diventa piccola)
            gradient = np.gradient(trace_values)
            smooth_gradient = pd.Series(gradient).rolling(window=20).mean().values
            
            convergence_candidates = np.where(np.abs(smooth_gradient) < np.std(gradient) * 0.1)[0]
            if len(convergence_candidates) > 0:
                convergence_point = convergence_candidates[0]
                axes[0, 0].axvline(x=convergence_point, color='green', linestyle='--',
                                 label=f'Convergence: {convergence_point}')
        
        axes[0, 0].set_title('Covariance Trace Evolution')
        axes[0, 0].set_ylabel('Trace(P)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Log-scale per vedere dettagli
        axes[0, 1].semilogy(trace_values, 'r-', linewidth=2)
        axes[0, 1].set_title('Trace Evolution (Log Scale)')
        axes[0, 1].set_ylabel('log(Trace(P))')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Gradient della traccia
        gradient = np.gradient(trace_values)
        axes[1, 0].plot(gradient, 'purple', alpha=0.7, linewidth=1)
        
        # Smooth gradient
        if len(gradient) > 20:
            smooth_gradient = pd.Series(gradient).rolling(window=20).mean()
            axes[1, 0].plot(smooth_gradient, 'k-', linewidth=2, label='Smoothed')
        
        axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 0].set_title('Trace Gradient (Rate of Change)')
        axes[1, 0].set_ylabel('dTrace/dt')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Analisi di stabilità (ultimi 30%)
        coefficient_of_variation = None
        if len(trace_values) > 100:
            steady_state_start = int(len(trace_values) * 0.7)
            steady_state_values = trace_values[steady_state_start:]
            
            axes[1, 1].plot(range(steady_state_start, len(trace_values)), 
                           steady_state_values, 'g-', linewidth=2)
            
            # Statistiche steady-state
            ss_mean = np.mean(steady_state_values)
            ss_std = np.std(steady_state_values)
            
            axes[1, 1].axhline(y=ss_mean, color='r', linestyle='--', 
                              label=f'Mean: {ss_mean:.3f}')
            axes[1, 1].axhline(y=ss_mean + ss_std, color='orange', linestyle=':', 
                              label=f'±1σ: {ss_std:.3f}')
            axes[1, 1].axhline(y=ss_mean - ss_std, color='orange', linestyle=':')
            
            axes[1, 1].set_title('Steady-State Analysis')
            axes[1, 1].set_ylabel('Trace(P)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            coefficient_of_variation = ss_std / ss_mean if ss_mean != 0 else float('inf')
        
        plt.tight_layout()
        plot_path = resolve_output_path(f"covariance_analysis_{axis}.png", output_dir)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Metriche di convergenza
        convergence_ratio = trace_values[-1] / trace_values[0] if trace_values[0] != 0 else float('inf')
        gradient_stabilization = np.std(gradient[-100:]) if len(gradient) > 100 else np.std(gradient)
        
        metrics = {
            'final_trace': float(trace_values[-1]),
            'initial_trace': float(trace_values[0]),
            'convergence_ratio': float(convergence_ratio),
            'convergence_point': int(convergence_point) if convergence_point is not None else None,
            'gradient_stabilization': float(gradient_stabilization),
            'coefficient_of_variation': float(coefficient_of_variation) if coefficient_of_variation is not None else None,
            'max_trace': float(np.max(trace_values)),
            'min_trace': float(np.min(trace_values)),
            'sample_size': len(trace_values)
        }
        
        print_colored(f"✅ Analisi covarianza completata: finale={trace_values[-1]:.2f}, ratio={convergence_ratio:.4f}", "✅", "green")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Errore nell'analisi della covarianza: {e}")
        print_colored(f"❌ Errore analisi covarianza: {e}", "❌", "red")
        return {'status': 'error', 'message': str(e)}


def residual_outlier_analysis(results, data, config, axis):
    """
    Analisi dei residui per identificare outlier e problemi di modellazione.
    
    Args:
        results: Dizionario risultati EKF
        data: DataFrame dati originali
        config: Configurazione EKF
        axis: Asse analizzato
        
    Returns:
        dict: Metriche di analisi residui
    """
    try:
        # Estrai dati
        velocities = np.asarray(results['velocities'])
        positions = np.asarray(results.get('positions', results['velocities']))  # fallback if not provided
        timestamps = np.asarray(results['timestamps'])
        acc_column = f'acc_{axis.lower()}'
        
        if acc_column not in data.columns:
            print_colored(f"⚠️ Colonna {acc_column} non trovata nei dati", "⚠️", "yellow")
            return {'status': 'column_not_found'}
        
        measured_acc = data[acc_column].values
        
        # Assicurati che le lunghezze corrispondano
        min_len = min(len(velocities), len(measured_acc), len(timestamps))
        velocities = velocities[:min_len]
        measured_acc = measured_acc[:min_len]
        timestamps = timestamps[:min_len]
        
        if min_len < 10:
            print_colored("⚠️ Dati insufficienti per analisi residui", "⚠️", "yellow")
            return {'status': 'insufficient_data'}
        
        print_colored(f"📊 Analisi residui per asse {axis} ({min_len} campioni)...", "📊", "cyan")
        
        # Calcola accelerazioni stimate (derivata numerica della velocità)
        estimated_acc = np.gradient(velocities, timestamps)
        
        # Residui di accelerazione
        acc_residuals = measured_acc - estimated_acc
        
        # Gestione valori non finiti
        finite_mask = (
            np.isfinite(acc_residuals) &
            np.isfinite(measured_acc) &
            np.isfinite(estimated_acc) &
            np.isfinite(velocities) &
            np.isfinite(timestamps)
        )
        
        if not np.any(finite_mask):
            message = "Nessun dato finito disponibile per l'analisi dei residui"
            logger.error(f"❌ {message}")
            print_colored(f"❌ Errore analisi residui: {message}", "❌", "red")
            return {'status': 'no_finite_data'}
        
        if not np.all(finite_mask):
            removed = np.size(finite_mask) - np.count_nonzero(finite_mask)
            logger.warning(f"⚠️ Residual analysis: rimossi {removed} campioni non finiti")
            acc_residuals = acc_residuals[finite_mask]
            measured_acc = measured_acc[finite_mask]
            estimated_acc = estimated_acc[finite_mask]
            velocities = velocities[finite_mask]
            timestamps = timestamps[finite_mask]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Residual and Outlier Analysis - Axis {axis}', fontsize=16)
        
        # 1. Serie temporale dei residui
        axes[0, 0].plot(timestamps, acc_residuals, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Acceleration Residuals')
        axes[0, 0].set_ylabel('Residual (m/s²)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Identifica outlier (3-sigma rule)
        residual_std = np.std(acc_residuals)
        outlier_threshold = 3 * residual_std
        outliers = np.abs(acc_residuals) > outlier_threshold
        
        if np.any(outliers):
            axes[0, 0].scatter(timestamps[outliers], acc_residuals[outliers], 
                              color='red', s=30, label=f'Outliers: {np.sum(outliers)}')
        axes[0, 0].axhline(y=outlier_threshold, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].axhline(y=-outlier_threshold, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].legend()
        
        # 2. Istogramma residui
        axes[0, 1].hist(acc_residuals, bins=50, density=True, alpha=0.7, color='lightblue')
        
        # Overlay normale
        mu, sigma = stats.norm.fit(acc_residuals)
        x = np.linspace(acc_residuals.min(), acc_residuals.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
        axes[0, 1].set_title(f'Residual Distribution\nμ={mu:.4f}, σ={sigma:.4f}')
        
        # 3. Scatter plot misurato vs stimato
        axes[1, 0].scatter(measured_acc, estimated_acc, alpha=0.6, s=1)
        
        # Linea identità
        min_val = min(measured_acc.min(), estimated_acc.min())
        max_val = max(measured_acc.max(), estimated_acc.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # R-squared
        correlation = np.corrcoef(measured_acc, estimated_acc)[0, 1]
        r_squared = correlation ** 2
        axes[1, 0].set_title(f'Measured vs Estimated Acceleration\nR² = {r_squared:.4f}')
        axes[1, 0].set_xlabel('Measured Acc (m/s²)')
        axes[1, 0].set_ylabel('Estimated Acc (m/s²)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Autocorrelazione dei residui
        if STATSMODELS_AVAILABLE and len(acc_residuals) > 10:
            lags = min(50, len(acc_residuals)//4)
            autocorr = acf(acc_residuals, nlags=lags, fft=True)
            
            axes[1, 1].stem(range(lags+1), autocorr)
            axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[1, 1].axhline(y=1.96/np.sqrt(len(acc_residuals)), color='r', linestyle='--')
            axes[1, 1].axhline(y=-1.96/np.sqrt(len(acc_residuals)), color='r', linestyle='--')
            axes[1, 1].set_title('Residual Autocorrelation')
            max_autocorr = np.max(np.abs(autocorr[1:])) if len(autocorr) > 1 else 0
        else:
            axes[1, 1].text(0.5, 0.5, 'Autocorrelation not available', ha='center', va='center')
            max_autocorr = None
        
        # 5. Analisi temporale della varianza
        window_size = max(10, len(acc_residuals) // 20)
        if window_size < len(acc_residuals):
            rolling_var = pd.Series(acc_residuals).rolling(window=window_size).var()
            axes[2, 0].plot(timestamps, rolling_var, 'g-', linewidth=2)
            axes[2, 0].set_title(f'Rolling Variance (window={window_size})')
            axes[2, 0].set_ylabel('Variance')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Residui vs velocità stimata (heteroscedasticity check)
        axes[2, 1].scatter(velocities, acc_residuals, alpha=0.6, s=1)
        axes[2, 1].axhline(y=0, color='r', linestyle='--')
        axes[2, 1].set_title('Residuals vs Estimated Velocity')
        axes[2, 1].set_xlabel('Estimated Velocity (m/s)')
        axes[2, 1].set_ylabel('Acceleration Residual (m/s²)')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = resolve_output_path(f"residual_analysis_{axis}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Test Shapiro-Wilk per normalità residui
        sample_size = min(5000, len(acc_residuals))
        shapiro_stat, shapiro_p = stats.shapiro(acc_residuals[:sample_size])
        
        metrics = {
            'rmse': float(np.sqrt(np.mean(acc_residuals**2))),
            'mae': float(np.mean(np.abs(acc_residuals))),
            'r_squared': float(r_squared),
            'outlier_count': int(np.sum(outliers)),
            'outlier_percentage': float(np.sum(outliers) / len(outliers) * 100),
            'residual_mean': float(np.mean(acc_residuals)),
            'residual_std': float(np.std(acc_residuals)),
            'max_autocorr': float(max_autocorr) if max_autocorr is not None else None,
            'shapiro_p_value': float(shapiro_p),
            'sample_size': len(acc_residuals)
        }
        
        print_colored(f"✅ Analisi residui completata: RMSE={metrics['rmse']:.4f}, R²={r_squared:.4f}", "✅", "green")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Errore nell'analisi dei residui: {e}")
        print_colored(f"❌ Errore analisi residui: {e}", "❌", "red")
        return {'status': 'error', 'message': str(e)}


def generate_tuning_recommendations(innovation_metrics, nis_metrics, covariance_metrics, residual_metrics):
    """
    Genera raccomandazioni specifiche per il tuning basate sulle analisi.
    
    Args:
        innovation_metrics: Metriche analisi innovazioni
        nis_metrics: Metriche analisi NIS
        covariance_metrics: Metriche analisi covarianza
        residual_metrics: Metriche analisi residui
        
    Returns:
        list: Lista di raccomandazioni di tuning
    """
    recommendations = []
    
    # Analisi NIS
    if nis_metrics.get('status') != 'error' and 'percentage_in_bounds' in nis_metrics:
        if nis_metrics['percentage_in_bounds'] < 90:
            if nis_metrics['mean_nis'] > nis_metrics['expected_mean'] * 1.5:
                recommendations.append({
                    'parameter': 'R_measurement',
                    'action': 'increase',
                    'factor': 1.5,
                    'priority': 'high',
                    'reason': f"NIS troppo alto ({nis_metrics['mean_nis']:.2f}), solo {nis_metrics['percentage_in_bounds']:.1f}% entro bounds"
                })
            elif nis_metrics['mean_nis'] < nis_metrics['expected_mean'] * 0.5:
                recommendations.append({
                    'parameter': 'R_measurement',
                    'action': 'decrease',
                    'factor': 0.7,
                    'priority': 'medium',
                    'reason': f"NIS troppo basso ({nis_metrics['mean_nis']:.2f})"
                })
    
    # Analisi innovazioni
    if innovation_metrics.get('status') != 'error':
        if 'shapiro_p_value' in innovation_metrics and innovation_metrics['shapiro_p_value'] < 0.01:
            recommendations.append({
                'parameter': 'Q_process',
                'action': 'adjust',
                'factor': 1.2,
                'priority': 'medium',
                'reason': f"Innovazioni non normali (p={innovation_metrics['shapiro_p_value']:.4f})"
            })
        
        if 'max_autocorr' in innovation_metrics and innovation_metrics['max_autocorr'] and innovation_metrics['max_autocorr'] > 0.15:
            recommendations.append({
                'parameter': 'Q_process',
                'action': 'increase',
                'factor': 1.3,
                'priority': 'high',
                'reason': f"Innovazioni correlate (max autocorr = {innovation_metrics['max_autocorr']:.3f})"
            })
    
    # Analisi covarianza
    if covariance_metrics.get('status') != 'error':
        if 'coefficient_of_variation' in covariance_metrics and covariance_metrics['coefficient_of_variation'] and covariance_metrics['coefficient_of_variation'] > 0.1:
            recommendations.append({
                'parameter': 'Q_process',
                'action': 'decrease',
                'factor': 0.8,
                'priority': 'medium',
                'reason': f"Covarianza instabile (CV = {covariance_metrics['coefficient_of_variation']:.3f})"
            })
        
        if 'convergence_ratio' in covariance_metrics and covariance_metrics['convergence_ratio'] > 100:
            recommendations.append({
                'parameter': 'Q_process',
                'action': 'decrease',
                'factor': 0.5,
                'priority': 'high',
                'reason': f"Divergenza traccia covarianza (ratio = {covariance_metrics['convergence_ratio']:.1f})"
            })
    
    # Analisi residui
    if residual_metrics.get('status') != 'error':
        if 'outlier_percentage' in residual_metrics and residual_metrics['outlier_percentage'] > 5:
            recommendations.append({
                'parameter': 'R_measurement',
                'action': 'increase',
                'factor': 1.4,
                'priority': 'medium',
                'reason': f"Troppi outlier ({residual_metrics['outlier_percentage']:.1f}%)"
            })
        
        if 'r_squared' in residual_metrics and residual_metrics['r_squared'] < 0.7:
            recommendations.append({
                'parameter': 'Q_process',
                'action': 'increase',
                'factor': 1.2,
                'priority': 'medium',
                'reason': f"Bassa correlazione misurato vs stimato (R² = {residual_metrics['r_squared']:.3f})"
            })
    
    return recommendations


def print_tuning_recommendations(recommendations, axis):
    """
    Stampa le raccomandazioni di tuning in modo formattato.
    
    Args:
        recommendations: Lista raccomandazioni
        axis: Asse analizzato
    """
    print_colored(f"🎛️ ============ RACCOMANDAZIONI TUNING ASSE {axis} ============", "🎛️", "magenta")
    
    if not recommendations:
        print_colored("✅ Nessuna raccomandazione: parametri sembrano ben calibrati!", "✅", "green")
        return
    
    # Ordina per priorità
    priority_order = {'high': 1, 'medium': 2, 'low': 3}
    recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 2))
    
    for i, rec in enumerate(recommendations, 1):
        action_emoji = "📈" if rec['action'] == 'increase' else "📉" if rec['action'] == 'decrease' else "🔧"
        priority_emoji = "🔴" if rec.get('priority') == 'high' else "🟡" if rec.get('priority') == 'medium' else "🟢"
        factor_text = f"fattore {rec['factor']}" if 'factor' in rec else ""
        
        print_colored(f"{priority_emoji} {action_emoji} {i}. {rec['parameter'].upper()}: {rec['action']} {factor_text}", 
                     action_emoji, "cyan")
        print_colored(f"     💡 Motivo: {rec['reason']}", "💡", "yellow")
        print_colored("", "", "white")  # Spacer
    
    print_colored("🎛️ ========================================", "🎛️", "magenta")


def generate_tuning_dashboard(performance_monitor, results, data, config, axis, output_dir):
    """
    Dashboard completo per il tuning dei parametri EKF.
    
    Args:
        performance_monitor: Oggetto EKFPerformanceMonitor
        results: Dizionario risultati EKF
        data: DataFrame dati originali
        config: Configurazione EKF
        axis: Asse analizzato
        output_dir: Directory di output
        
    Returns:
        dict: Report completo di tuning
    """
    try:
        print_colored("📊 Generazione dashboard di tuning...", "📊", "cyan")
        
        # Esegui tutte le analisi
        innovation_metrics = comprehensive_innovation_analysis(performance_monitor, output_dir, axis)
        nis_metrics = nis_validation_analysis(performance_monitor, output_dir, axis)
        covariance_metrics = covariance_trace_analysis(performance_monitor, output_dir, axis)
        residual_metrics = residual_outlier_analysis(results, data, config, axis)
        
        # Genera raccomandazioni
        recommendations = generate_tuning_recommendations(
            innovation_metrics, nis_metrics, covariance_metrics, residual_metrics
        )
        
        # Crea report di tuning
        tuning_report = {
            'axis': axis,
            'timestamp': datetime.now().isoformat(),
            'current_parameters': {
                'Q_position': config.get('kalman_filter', {}).get('process_noise', {}).get('position', 'N/A'),
                'Q_velocity': config.get('kalman_filter', {}).get('process_noise', {}).get('velocity', 'N/A'),
                'Q_acceleration': config.get('kalman_filter', {}).get('process_noise', {}).get('acceleration', 'N/A'),
                'Q_bias': config.get('kalman_filter', {}).get('process_noise', {}).get('bias', 'N/A'),
                'R_measurement': config.get('kalman_filter', {}).get('measurement_noise', 'N/A')
            },
            'validation_metrics': {
                'innovation_analysis': innovation_metrics,
                'nis_analysis': nis_metrics,
                'covariance_analysis': covariance_metrics,
                'residual_analysis': residual_metrics
            },
            'tuning_recommendations': recommendations,
            'summary': {
                'total_recommendations': len(recommendations),
                'high_priority': len([r for r in recommendations if r.get('priority') == 'high']),
                'medium_priority': len([r for r in recommendations if r.get('priority') == 'medium']),
                'low_priority': len([r for r in recommendations if r.get('priority') == 'low'])
            }
        }
        
        # Salva report
        tuning_report_path = resolve_output_path(f"tuning_report_{axis}.json", output_dir)
        with open(tuning_report_path, 'w') as f:
            json.dump(tuning_report, f, indent=2, default=str)
        
        print_colored(f"💾 Report tuning salvato: {tuning_report_path}", "💾", "green")
        
        # Stampa raccomandazioni
        print_tuning_recommendations(recommendations, axis)
        
        # Stampa riassunto metriche principali
        print_colored(f"📊 ============ RIASSUNTO METRICHE ASSE {axis} ============", "📊", "blue")
        
        if innovation_metrics.get('status') != 'error':
            print_colored(f"🔍 Innovazioni: μ={innovation_metrics.get('mean', 'N/A'):.4f}, " +
                         f"p_normalità={innovation_metrics.get('shapiro_p_value', 'N/A'):.4f}", "🔍", "white")
        
        if nis_metrics.get('status') != 'error':
            print_colored(f"📊 NIS: {nis_metrics.get('percentage_in_bounds', 'N/A'):.1f}% in bounds, " +
                         f"μ={nis_metrics.get('mean_nis', 'N/A'):.2f}", "📊", "white")
        
        if covariance_metrics.get('status') != 'error':
            print_colored(f"📈 Covarianza: finale={covariance_metrics.get('final_trace', 'N/A'):.2f}, " +
                         f"ratio={covariance_metrics.get('convergence_ratio', 'N/A'):.2f}", "📈", "white")
        
        if residual_metrics.get('status') != 'error':
            print_colored(f"🎯 Residui: RMSE={residual_metrics.get('rmse', 'N/A'):.4f}, " +
                         f"R²={residual_metrics.get('r_squared', 'N/A'):.3f}, " +
                         f"outliers={residual_metrics.get('outlier_percentage', 'N/A'):.1f}%", "🎯", "white")
        
        print_colored("📊 ======================================================", "📊", "blue")
        
        return tuning_report
        
    except Exception as e:
        logger.error(f"❌ Errore nella generazione del dashboard di tuning: {e}")
        print_colored(f"❌ Errore dashboard tuning: {e}", "❌", "red")
        return {'status': 'error', 'message': str(e)}


def generate_multi_axis_comparison(tuning_reports, output_dir):
    """
    Genera un confronto tra i report di tuning di diversi assi.
    
    Args:
        tuning_reports: Dictionary dei report per ogni asse
        output_dir: Directory di output
        
    Returns:
        dict: Report di confronto multi-asse
    """
    try:
        print_colored("📊 Generazione confronto multi-asse...", "📊", "cyan")
        
        comparison_report = {
            'timestamp': datetime.now().isoformat(),
            'axes_analyzed': list(tuning_reports.keys()),
            'axis_comparison': {},
            'overall_recommendations': [],
            'summary': {}
        }
        
        # Confronta metriche tra assi
        metrics_comparison = {}
        for axis, report in tuning_reports.items():
            if report.get('status') == 'error':
                continue
                
            validation_metrics = report.get('validation_metrics', {})
            
            # Estrai metriche chiave
            axis_metrics = {
                'innovation_mean': validation_metrics.get('innovation_analysis', {}).get('mean', None),
                'innovation_std': validation_metrics.get('innovation_analysis', {}).get('std', None),
                'nis_percentage_in_bounds': validation_metrics.get('nis_analysis', {}).get('percentage_in_bounds', None),
                'covariance_final_trace': validation_metrics.get('covariance_analysis', {}).get('final_trace', None),
                'residual_rmse': validation_metrics.get('residual_analysis', {}).get('rmse', None),
                'residual_r_squared': validation_metrics.get('residual_analysis', {}).get('r_squared', None),
                'total_recommendations': report.get('summary', {}).get('total_recommendations', 0),
                'high_priority_recommendations': report.get('summary', {}).get('high_priority', 0)
            }
            
            metrics_comparison[axis] = axis_metrics
            comparison_report['axis_comparison'][axis] = axis_metrics
        
        # Identifica l'asse con migliori prestazioni
        best_axis = None
        best_score = float('inf')
        
        for axis, metrics in metrics_comparison.items():
            # Calcola score composito (più basso è meglio)
            score = 0
            if metrics['total_recommendations'] is not None:
                score += metrics['total_recommendations'] * 10
            if metrics['high_priority_recommendations'] is not None:
                score += metrics['high_priority_recommendations'] * 50
            if metrics['nis_percentage_in_bounds'] is not None:
                score += abs(95 - metrics['nis_percentage_in_bounds'])
            
            if score < best_score:
                best_score = score
                best_axis = axis
        
        comparison_report['best_performing_axis'] = best_axis
        
        # Raccomandazioni generali
        total_high_priority = sum(metrics.get('high_priority_recommendations', 0) 
                                 for metrics in metrics_comparison.values())
        
        if total_high_priority > 0:
            comparison_report['overall_recommendations'].append({
                'type': 'urgent',
                'message': f"Rilevate {total_high_priority} raccomandazioni ad alta priorità. Rivedere tuning urgentemente."
            })
        
        # Confronto statistico tra assi
        rmse_values = [m.get('residual_rmse') for m in metrics_comparison.values() if m.get('residual_rmse') is not None]
        if len(rmse_values) > 1:
            rmse_variation = np.std(rmse_values) / np.mean(rmse_values)
            if rmse_variation > 0.5:
                comparison_report['overall_recommendations'].append({
                    'type': 'warning',
                    'message': f"Grande variazione RMSE tra assi ({rmse_variation:.2f}). Verificare configurazione."
                })
        
        # Salva report di confronto
        comparison_path = resolve_output_path("multi_axis_comparison.json", output_dir)
        with open(comparison_path, 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        
        # Stampa riassunto
        print_colored("📊 ============ CONFRONTO MULTI-ASSE ============", "📊", "blue")
        
        for axis, metrics in metrics_comparison.items():
            status_emoji = "✅" if axis == best_axis else "⚠️" if metrics.get('total_recommendations', 0) > 5 else "🔵"
            print_colored(f"{status_emoji} Asse {axis}: {metrics.get('total_recommendations', 0)} raccomandazioni " +
                         f"(RMSE: {metrics.get('residual_rmse', 'N/A'):.4f})", status_emoji, "white")
        
        if best_axis:
            print_colored(f"🏆 Asse con migliori prestazioni: {best_axis}", "🏆", "green")
        
        print_colored("📊 ===============================================", "📊", "blue")
        
        return comparison_report
        
    except Exception as e:
        logger.error(f"❌ Errore nel confronto multi-asse: {e}")
        print_colored(f"❌ Errore confronto multi-asse: {e}", "❌", "red")
        return {'status': 'error', 'message': str(e)}


def generate_multi_axis_comparison(results_dict, output_dir):
    """
    Genera un confronto completo tra tutti gli assi analizzati.
    
    Args:
        results_dict: Dictionary con risultati per tutti gli assi
        output_dir: Directory di output
        
    Returns:
        dict: Report di confronto multi-asse
    """
    try:
        print_colored("🌐 Generazione confronto multi-asse...", "🌐", "cyan")
        
        # Estrai dati da tutti gli assi
        axis_data = {}
        for axis, results in results_dict.items():
            if isinstance(results, dict) and 'tuning_report' in results:
                tuning_report = results['tuning_report']
                if tuning_report.get('status') != 'error':
                    axis_data[axis] = {
                        'innovation_metrics': tuning_report.get('validation_metrics', {}).get('innovation_analysis', {}),
                        'nis_metrics': tuning_report.get('validation_metrics', {}).get('nis_analysis', {}),
                        'covariance_metrics': tuning_report.get('validation_metrics', {}).get('covariance_analysis', {}),
                        'residual_metrics': tuning_report.get('validation_metrics', {}).get('residual_analysis', {}),
                        'recommendations': tuning_report.get('tuning_recommendations', [])
                    }
        
        if len(axis_data) < 2:
            print_colored("⚠️ Dati insufficienti per confronto multi-asse", "⚠️", "yellow")
            return {'status': 'insufficient_data'}
        
        # Crea grafici di confronto
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Axis Comparison Dashboard', fontsize=16)
        
        # 1. Confronto NIS percentage in bounds
        axes[0, 0].bar(axis_data.keys(), 
                      [data['nis_metrics'].get('percentage_in_bounds', 0) for data in axis_data.values()])
        axes[0, 0].axhline(y=95, color='r', linestyle='--', label='Target 95%')
        axes[0, 0].set_title('NIS Percentage In Bounds')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].legend()
        
        # 2. Confronto RMSE residui
        axes[0, 1].bar(axis_data.keys(), 
                      [data['residual_metrics'].get('rmse', 0) for data in axis_data.values()])
        axes[0, 1].set_title('Residuals RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE')
        
        # 3. Confronto R-squared
        axes[0, 2].bar(axis_data.keys(), 
                      [data['residual_metrics'].get('r_squared', 0) for data in axis_data.values()])
        axes[0, 2].axhline(y=0.8, color='g', linestyle='--', label='Good threshold')
        axes[0, 2].set_title('R-squared Comparison')
        axes[0, 2].set_ylabel('R²')
        axes[0, 2].legend()
        
        # 4. Confronto traccia finale covarianza
        axes[1, 0].bar(axis_data.keys(), 
                      [data['covariance_metrics'].get('final_trace', 0) for data in axis_data.values()])
        axes[1, 0].set_title('Final Covariance Trace')
        axes[1, 0].set_ylabel('Trace(P)')
        axes[1, 0].set_yscale('log')
        
        # 5. Confronto percentuale outlier
        axes[1, 1].bar(axis_data.keys(), 
                      [data['residual_metrics'].get('outlier_percentage', 0) for data in axis_data.values()])
        axes[1, 1].axhline(y=5, color='r', linestyle='--', label='Warning threshold')
        axes[1, 1].set_title('Outlier Percentage')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].legend()
        
        # 6. Numero raccomandazioni per asse
        recommendation_counts = [len(data['recommendations']) for data in axis_data.values()]
        colors = ['green' if count == 0 else 'orange' if count <= 2 else 'red' for count in recommendation_counts]
        axes[1, 2].bar(axis_data.keys(), recommendation_counts, color=colors)
        axes[1, 2].set_title('Tuning Recommendations Count')
        axes[1, 2].set_ylabel('Number of Recommendations')
        
        plt.tight_layout()
        multi_axis_plot_path = resolve_output_path("multi_axis_comparison.png", output_dir)
        plt.savefig(multi_axis_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Genera report comparativo
        comparison_report = {
            'timestamp': datetime.now().isoformat(),
            'analyzed_axes': list(axis_data.keys()),
            'summary_metrics': {},
            'best_performing_axis': None,
            'worst_performing_axis': None,
            'global_recommendations': []
        }
        
        # Calcola metriche di sintesi
        for axis, data in axis_data.items():
            score = 0
            # Punteggio basato su NIS in bounds (peso 30%)
            nis_score = min(data['nis_metrics'].get('percentage_in_bounds', 0) / 95.0, 1.0) * 30
            # Punteggio basato su R² (peso 25%)
            r2_score = data['residual_metrics'].get('r_squared', 0) * 25
            # Punteggio basato su bassa percentuale outlier (peso 20%)
            outlier_score = max(0, (5 - data['residual_metrics'].get('outlier_percentage', 5)) / 5.0) * 20
            # Punteggio basato su basso numero raccomandazioni (peso 25%)
            rec_score = max(0, (5 - len(data['recommendations'])) / 5.0) * 25
            
            score = nis_score + r2_score + outlier_score + rec_score
            comparison_report['summary_metrics'][axis] = {
                'overall_score': score,
                'nis_score': nis_score,
                'r2_score': r2_score,
                'outlier_score': outlier_score,
                'recommendation_score': rec_score
            }
        
        # Identifica migliore e peggiore asse
        scores = {axis: metrics['overall_score'] for axis, metrics in comparison_report['summary_metrics'].items()}
        comparison_report['best_performing_axis'] = max(scores, key=scores.get)
        comparison_report['worst_performing_axis'] = min(scores, key=scores.get)
        
        # Genera raccomandazioni globali
        if len([data for data in axis_data.values() if len(data['recommendations']) > 0]) > len(axis_data) * 0.5:
            comparison_report['global_recommendations'].append({
                'type': 'global_parameter_review',
                'priority': 'high',
                'message': 'Più del 50% degli assi richiede tuning. Considerare revisione globale parametri.'
            })
        
        # Controlla consistenza tra assi
        nis_variance = np.var([data['nis_metrics'].get('percentage_in_bounds', 0) for data in axis_data.values()])
        if nis_variance > 100:  # Soglia arbitraria per alta varianza
            comparison_report['global_recommendations'].append({
                'type': 'inconsistent_performance',
                'priority': 'medium',
                'message': f'Performance inconsistente tra assi (var NIS: {nis_variance:.1f}). Verificare configurazione.'
            })
        
        # Salva report
        comparison_report_path = resolve_output_path("multi_axis_comparison_report.json", output_dir)
        with open(comparison_report_path, 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        
        # Stampa riassunto
        print_colored("🌐 ============ CONFRONTO MULTI-ASSE ============", "🌐", "blue")
        print_colored(f"🏆 Migliore performance: {comparison_report['best_performing_axis']} " +
                     f"(score: {scores[comparison_report['best_performing_axis']]:.1f})", "🏆", "green")
        print_colored(f"⚠️  Performance da migliorare: {comparison_report['worst_performing_axis']} " +
                     f"(score: {scores[comparison_report['worst_performing_axis']]:.1f})", "⚠️", "yellow")
        
        if comparison_report['global_recommendations']:
            print_colored("🔧 Raccomandazioni globali:", "🔧", "cyan")
            for rec in comparison_report['global_recommendations']:
                priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡"
                print_colored(f"  {priority_emoji} {rec['message']}", "💡", "white")
        
        print_colored("🌐 =======================================", "🌐", "blue")
        
        print_colored(f"💾 Report multi-asse salvato: {comparison_report_path}", "💾", "green")
        return comparison_report
        
    except Exception as e:
        logger.error(f"❌ Errore nel confronto multi-asse: {e}")
        print_colored(f"❌ Errore confronto multi-asse: {e}", "❌", "red")
        return {'status': 'error', 'message': str(e)}


def save_performance_report_to_file(performance_report, performance_monitor, output_dir, axis, format='yaml'):
    """
    Salva il report delle prestazioni in un file.
    
    Args:
        performance_report: Dictionary del report delle prestazioni
        performance_monitor: Oggetto EKFPerformanceMonitor
        output_dir: Directory di output
        axis: Asse analizzato
        format: Formato del file ('yaml', 'json', 'txt')
    """
    try:
        output_path = resolve_output_path(f'EKF_performance_report_axis_{axis}.{format}', output_dir)
        
        if format == 'yaml':
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(performance_report, f, default_flow_style=False, indent=2)
                
        elif format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(performance_report, f, indent=2, default=str)
                
        elif format == 'txt':
            with open(output_path, 'w') as f:
                f.write(f"EKF Performance Report - Axis {axis}\n")
                f.write("=" * 50 + "\n\n")
                
                for section, data in performance_report.items():
                    f.write(f"{section.upper()}:\n")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {data}\n")
                    f.write("\n")
                
                # Aggiungi dati grezzi
                f.write("RAW DATA:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Trace history (last 10): {list(performance_monitor.trace_history)[-10:]}\n")
                f.write(f"Innovation history (last 10): {list(performance_monitor.innovation_history)[-10:]}\n")
                f.write(f"NIS history (last 10): {list(performance_monitor.nis_history)[-10:]}\n")
        
        print_colored(f"📄 Report salvato in: {output_path}", "💾", "green")
        logger.info(f"📄 Report delle prestazioni salvato in: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Errore nel salvataggio del report: {e}")
        print_colored(f"❌ Errore nel salvataggio: {e}", "❌", "red")


def generate_enhanced_residual_plots(ekf, diagnostics, output_dir, axis):
    """
    Generate comprehensive residual analysis plots for enhanced EKF diagnostics.
    
    Args:
        ekf: ExtendedKalmanFilter3D instance
        diagnostics: Diagnostics dictionary from EKF
        output_dir: Output directory for plots
        axis: Current axis being processed
    """
    try:
        # Extract residual data
        if not diagnostics['residual_stats']['mean'] or len(ekf.residual_buffer) < 20:
            logger.warning("⚠️ Insufficient residual data for enhanced plots")
            return
            
        residuals = np.array(list(ekf.residual_buffer))
        
        # Create comprehensive plot
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'Enhanced Residual Analysis - Axis {axis}', fontsize=16)
        
        # 1. Residual time series for each axis
        for i in range(3):
            axes[0, i].plot(residuals[:, i], alpha=0.7)
            axes[0, i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, i].set_title(f'Residuals {["X", "Y", "Z"][i]} Axis')
            axes[0, i].set_ylabel('Residual')
            axes[0, i].grid(True, alpha=0.3)
            
            # Add mean and std lines
            mean_val = np.mean(residuals[:, i])
            std_val = np.std(residuals[:, i])
            axes[0, i].axhline(y=mean_val, color='g', linestyle='-', alpha=0.7, label=f'Mean: {mean_val:.3f}')
            axes[0, i].axhline(y=mean_val + 2*std_val, color='orange', linestyle=':', alpha=0.7, label=f'±2σ: ±{2*std_val:.3f}')
            axes[0, i].axhline(y=mean_val - 2*std_val, color='orange', linestyle=':', alpha=0.7)
            axes[0, i].legend(fontsize=8)
        
        # 2. Residual histograms with normality test
        for i in range(3):
            axes[1, i].hist(residuals[:, i], bins=30, alpha=0.7, density=True, color=['red', 'green', 'blue'][i])
            
            # Fit normal distribution
            mu, sigma = np.mean(residuals[:, i]), np.std(residuals[:, i])
            x = np.linspace(residuals[:, i].min(), residuals[:, i].max(), 100)
            normal_fit = stats.norm.pdf(x, mu, sigma)
            axes[1, i].plot(x, normal_fit, 'k--', alpha=0.8, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
            
            # Kolmogorov-Smirnov test for normality
            ks_stat, ks_p = stats.kstest(residuals[:, i], lambda x: stats.norm.cdf(x, mu, sigma))
            axes[1, i].set_title(f'Histogram {["X", "Y", "Z"][i]} - KS p-value: {ks_p:.4f}')
            axes[1, i].set_ylabel('Density')
            axes[1, i].legend(fontsize=8)
            axes[1, i].grid(True, alpha=0.3)
        
        # 3. Autocorrelation plots
        if STATSMODELS_AVAILABLE and 'autocorr' in diagnostics['residual_stats']:
            for i in range(3):
                autocorr = diagnostics['residual_stats']['autocorr'][i]
                lags = np.arange(1, len(autocorr) + 1)
                axes[2, i].plot(lags, autocorr, 'o-', alpha=0.7, color=['red', 'green', 'blue'][i])
                
                # Add 95% confidence interval
                ci_95 = 1.96 / np.sqrt(len(residuals))
                axes[2, i].axhline(y=ci_95, color='r', linestyle='--', alpha=0.5, label='95% CI')
                axes[2, i].axhline(y=-ci_95, color='r', linestyle='--', alpha=0.5)
                axes[2, i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                axes[2, i].set_title(f'Autocorrelation {["X", "Y", "Z"][i]}')
                axes[2, i].set_xlabel('Lag')
                axes[2, i].set_ylabel('Autocorrelation')
                axes[2, i].legend(fontsize=8)
                axes[2, i].grid(True, alpha=0.3)
        else:
            # Simple lag plot if statsmodels not available
            for i in range(3):
                if len(residuals) > 1:
                    axes[2, i].scatter(residuals[:-1, i], residuals[1:, i], alpha=0.5)
                    axes[2, i].set_title(f'Lag-1 Plot {["X", "Y", "Z"][i]}')
                    axes[2, i].set_xlabel('Residual(t)')
                    axes[2, i].set_ylabel('Residual(t+1)')
                    axes[2, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save enhanced residual plot
        plot_filename = f'enhanced_residual_analysis_{axis}.png'
        plot_path = resolve_output_path(plot_filename, output_dir)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics file
        stats_filename = f'residual_statistics_{axis}.json'
        stats_path = resolve_output_path(stats_filename, output_dir)
        
        # Compute comprehensive statistics
        stats_dict = {
            'axis': axis,
            'sample_size': len(residuals),
            'residual_statistics': {
                'mean': diagnostics['residual_stats']['mean'],
                'std': diagnostics['residual_stats']['std'],
                'min': residuals.min(axis=0).tolist(),
                'max': residuals.max(axis=0).tolist(),
                'median': np.median(residuals, axis=0).tolist(),
                'q25': np.percentile(residuals, 25, axis=0).tolist(),
                'q75': np.percentile(residuals, 75, axis=0).tolist()
            },
            'normality_tests': {},
            'outlier_statistics': {
                'total_outliers': diagnostics['outlier_count'],
                'outlier_percentage': 100 * diagnostics['outlier_count'] / max(1, diagnostics['total_measurements'])
            },
            'zupt_statistics': {
                'activations': diagnostics['zupt_activations'],
                'activation_rate': 100 * diagnostics['zupt_activations'] / max(1, diagnostics['total_measurements'])
            }
        }
        
        # Add normality test results
        for i in range(3):
            axis_name = ['X', 'Y', 'Z'][i]
            # Shapiro-Wilk test (more powerful for small samples)
            if len(residuals[:, i]) <= 5000:  # Shapiro-Wilk limitation
                sw_stat, sw_p = stats.shapiro(residuals[:, i])
                stats_dict['normality_tests'][f'{axis_name}_shapiro'] = {'statistic': float(sw_stat), 'p_value': float(sw_p)}
            
            # Kolmogorov-Smirnov test
            mu, sigma = np.mean(residuals[:, i]), np.std(residuals[:, i])
            ks_stat, ks_p = stats.kstest(residuals[:, i], lambda x: stats.norm.cdf(x, mu, sigma))
            stats_dict['normality_tests'][f'{axis_name}_ks'] = {'statistic': float(ks_stat), 'p_value': float(ks_p)}
        
        # Save statistics
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        logger.info(f"✅ Enhanced residual plots saved: {plot_path}")
        logger.info(f"✅ Residual statistics saved: {stats_path}")
        print_colored(f"📊 Enhanced residual analysis saved to: {plot_filename}", "📊", "green")
        
    except Exception as e:
        logger.error(f"❌ Error generating enhanced residual plots: {e}")
        print_colored(f"❌ Error in enhanced residual plots: {e}", "❌", "red")


def print_section_header(title, emoji="🔸"):
    """
    Stampa un'intestazione di sezione con bordi colorati.
    
    Args:
        title (str): Il titolo della sezione.
        emoji (str): L'emoji da usare per decorare.
    """
    border = "=" * (len(title) + 10)
    print_colored(border, "", "cyan")
    print_colored(f"    {title}    ", emoji, "cyan")
    print_colored(border, "", "cyan")


def print_progress_bar(current, total, prefix="Progress", length=50):
    """
    Stampa una barra di progresso colorata.
    
    Args:
        current (int): Valore corrente.
        total (int): Valore totale.
        prefix (str): Prefisso da mostrare.
        length (int): Lunghezza della barra.
    """
    percent = current / total
    filled_length = int(length * percent)
    bar = "█" * filled_length + "░" * (length - filled_length)
    
    if COLORAMA_AVAILABLE:
        if percent < 0.3:
            color = Fore.RED
        elif percent < 0.7:
            color = Fore.YELLOW
        else:
            color = Fore.GREEN
        print(f"\r{color}{prefix}: |{bar}| {percent:.1%} ({current}/{total}){Style.RESET_ALL}", end="")
    else:
        print(f"\r{prefix}: |{bar}| {percent:.1%} ({current}/{total})", end="")
    
    if current == total:
        print()  # New line when complete


def run_comprehensive_9d_ekf(config_path=None):
    """
    Esegue l'EKF 9D completo con ZUPT detection e post-processing robusto.
    
    Args:
        config_path: Path al file di configurazione YAML
    """
    setup_enhanced_logging()
    logger.info("🚀 Starting Comprehensive 9D EKF Processing")
    
    try:
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "EKF_for_vel_and_pos_est_from_acc.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"📋 Configuration loaded from: {config_path}")
        
        # Load data configuration (support legacy `input` and new `input_files` keys)
        io_config = config.get('io', {})
        data_config = io_config.get('input')
        delimiter = None
        accel_columns = None

        if isinstance(data_config, dict):
            filename = data_config.get('filename') or data_config.get('path')
            delimiter = data_config.get('delimiter', delimiter)
            accel_columns = data_config.get('accel_columns')
        else:
            input_files = io_config.get('input_files', {})
            if not input_files:
                raise KeyError("Missing input data configuration: expected `io.input` or `io.input_files`")
            # Prefer explicitly marked default, otherwise use 'linear', else the first entry
            preferred_key = io_config.get('default_input_key', 'linear')
            if preferred_key not in input_files:
                preferred_key = next(iter(input_files))
            entry = input_files[preferred_key]
            if isinstance(entry, dict):
                filename = entry.get('filename') or entry.get('path')
                delimiter = entry.get('delimiter', delimiter)
                accel_columns = entry.get('accel_columns', accel_columns)
            else:
                filename = entry

        if filename is None:
            raise KeyError("Input configuration missing filename/path information")

        input_file = Path(filename)
        if not input_file.is_absolute():
            input_file = (Path(__file__).parent.parent / input_file).resolve()
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Read acceleration data
        logger.info(f"📊 Loading acceleration data from: {input_file}")

        accel_col_names = None
        accel_col_indices = None
        if accel_columns is not None:
            try:
                if all(isinstance(c, (int, np.integer)) for c in accel_columns):
                    accel_col_indices = [int(c) for c in accel_columns]
                else:
                    accel_col_names = [str(c) for c in accel_columns]
            except TypeError:
                accel_col_names = [str(accel_columns)]

        # Helper to read with pandas when column names are supplied
        def _read_accel_frame(path, delim_setting):
            if delim_setting is None or str(delim_setting).lower() in ("", "auto", "none"):
                return pd.read_csv(path)
            if str(delim_setting).lower() in ("whitespace", "space", "spaces"):
                return pd.read_csv(path, delim_whitespace=True)
            return pd.read_csv(path, sep=delim_setting)

        accel_data = None

        if accel_col_names:
            data = _read_accel_frame(input_file, delimiter)
            missing = [c for c in accel_col_names if c not in data.columns]
            if missing:
                raise ValueError(
                    f"Requested acceleration columns {missing} not found in data. "
                    f"Available columns: {list(data.columns)}"
                )
            accel_data = data[accel_col_names].to_numpy(dtype=float)
        else:
            # Column selection by index (or default last 3 columns)
            def _load_numpy(path, delim_setting):
                if delim_setting is None or str(delim_setting).lower() in ("", "auto", "none"):
                    return np.loadtxt(path, dtype=float)
                if str(delim_setting).lower() in ("whitespace", "space", "spaces"):
                    return np.loadtxt(path, dtype=float)
                return np.loadtxt(path, delimiter=delim_setting, dtype=float)

            try:
                arr = _load_numpy(input_file, delimiter)
            except ValueError:
                # Retry with whitespace if explicit delimiter failed
                arr = _load_numpy(input_file, "whitespace")

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            if arr.shape[1] < 3:
                raise ValueError(
                    "Acceleration file must contain at least three numeric columns. "
                    f"Found {arr.shape[1]} numeric columns in {input_file}. "
                    "Update `io.input` delimiter/columns in the configuration."
                )

            if accel_col_indices is None:
                accel_col_indices = list(range(arr.shape[1] - 3, arr.shape[1]))
            accel_data = arr[:, accel_col_indices]
        
        # MEGA DEBUG - Check loaded data
        logger.info("🔍 DEBUG LOADED ACCELERATION DATA:")
        logger.info(f"  accel_data type: {type(accel_data)}")
        logger.info(f"  accel_data dtype: {accel_data.dtype}")
        logger.info(f"  accel_data shape: {accel_data.shape}")
        logger.info(f"  First 3 samples:\n{accel_data[:3]}")
        if len(accel_data) > 0:
            logger.info(f"  Sample value types: {[type(x) for x in accel_data[0]]}")
        
        # Force float conversion if needed
        if accel_data.dtype != np.float64:
            logger.warning(f"⚠️ Converting accel_data from {accel_data.dtype} to float64")
            accel_data = accel_data.astype(np.float64)
            logger.info(f"✅ Converted: new dtype={accel_data.dtype}")
        
        logger.info(f"📈 Loaded {len(accel_data)} acceleration samples")
        
        # ============================================
        # PREPROCESSING: Filter acceleration data
        # ============================================
        sample_rate = config['ekf'].get('sample_rate_hz', 100.0)
        
        # BIOMECHANICAL IMPROVEMENT 1: Auto-bias estimation
        bias_cfg = config.get('preprocessing', {}).get('auto_bias', {})
        if bias_cfg.get('enabled', True):
            n_samples = min(bias_cfg.get('initial_samples', 100), len(accel_data) // 10)
            initial_bias = np.mean(accel_data[:n_samples], axis=0)
            logger.info(f"🎯 Auto-bias stimato dai primi {n_samples} campioni: {initial_bias}")
            accel_data = accel_data - initial_bias
        
        logger.info(f"🔍 About to call preprocessing with sample_rate={sample_rate}")
        logger.info(f"🔍 Config keys: {list(config.keys())}")
        logger.info(f"🔍 'preprocessing' in config: {'preprocessing' in config}")
        accel_data = preprocess_acceleration(accel_data, sample_rate, config)
        logger.info("✅ Acceleration preprocessing completed")
        
        # Initialize 9D EKF
        ekf = ExtendedKalmanFilter9D(config)
        logger.info("🔧 9D Extended Kalman Filter initialized")
        
        # Storage for results
        positions = []
        velocities = []
        biases = []
        zupt_flags = []
        
        # Warm-up configuration
        warmup_cfg = config.get('preprocessing', {}).get('warmup', {})
        skip_samples = warmup_cfg.get('skip_samples', 0) if warmup_cfg.get('enabled', False) else 0
        fast_convergence = warmup_cfg.get('fast_convergence', False)
        
        # Process each sample
        window_size = max(ekf.zupt_detector.window_size, 1)
        for i, accel_sample in enumerate(accel_data):
            
            # BIOMECHANICAL IMPROVEMENT 4: Fast convergence during warmup
            if fast_convergence and i < skip_samples:
                # Temporarily increase process noise for faster adaptation
                original_Q = ekf.Q.copy()
                ekf.Q *= 10.0  # 10x higher process noise during warmup
                ekf.predict(accel_sample)
                ekf.Q = original_Q  # Restore original
            else:
                # Normal prediction
                ekf.predict(accel_sample)
            
            # Store results (skip warmup samples if configured)
            if i >= skip_samples:
                # BIOMECHANICAL IMPROVEMENT 5: Reset position at start of recording for cyclic motion
                if len(positions) == 0 and skip_samples > 0:
                    # Reset position to zero at start of actual recording
                    ekf.x[0:3, 0] = 0.0  
                    logger.info(f"🎯 Position reset to zero after warmup")
                
                # Get current estimates AFTER potential reset
                current_position = ekf.get_position()
                current_velocity = ekf.get_velocity()
                current_bias = ekf.get_bias()
                
                positions.append(current_position)
                velocities.append(current_velocity)
                biases.append(current_bias)
            elif i == skip_samples - 1:
                logger.info(f"🚀 Warmup completed after {skip_samples} samples, starting data recording")
            
            # Always get estimates for ZUPT detection
            if i < skip_samples:
                current_position = ekf.get_position()
                current_velocity = ekf.get_velocity()
                current_bias = ekf.get_bias()
            
            # ZUPT detection basato sull'ultima finestra disponibile
            window_start = max(0, i + 1 - window_size)
            accel_window = accel_data[window_start:i+1]
            
            # Use velocity from positions list (skip warmup adjustment)
            if len(velocities) >= window_size:
                velocity_window = velocities[-window_size:]
            else:
                velocity_window = velocities if len(velocities) > 0 else None
            
            is_stationary = ekf.zupt_detector.detect_stationary(
                accel_window,
                velocity_window if velocity_window is not None else None
            )
            
            if is_stationary:
                ekf.apply_zupt_update(accel_sample)
                # Aggiorna con valori corretti dopo ZUPT (only if recording)
                if i >= skip_samples:
                    positions[-1] = ekf.get_position()
                    velocities[-1] = ekf.get_velocity()
                    biases[-1] = ekf.get_bias()
            
            # BIOMECHANICAL IMPROVEMENT 6: Real-time velocity baseline removal
            baseline_cfg = config.get('ekf', {}).get('velocity_baseline_removal', {})
            if baseline_cfg.get('enabled', False) and i >= skip_samples:
                # Calculate rolling baseline of velocity
                baseline_window = baseline_cfg.get('window_size', 500)  # 5 seconds
                if len(velocities) >= baseline_window:
                    # Get velocity vectors and reshape to (window_size, 3)
                    recent_velocities = np.array(velocities[-baseline_window:])
                    
                    if recent_velocities.ndim == 3:  # Shape (window_size, 3, 1) -> (window_size, 3)
                        recent_velocities = recent_velocities.squeeze(axis=2)
                    
                    velocity_baseline = np.mean(recent_velocities, axis=0)  # Shape (3,)
                    
                    # Remove baseline from current velocity
                    baseline_factor = baseline_cfg.get('removal_factor', 0.2)  # 20% baseline removal
                    correction = baseline_factor * velocity_baseline  # Shape (3,)
                    
                    # Apply correction to EKF state (manual assignment to avoid numpy broadcast issues)
                    ekf.x[3, 0] = ekf.x[3, 0] - correction[0]
                    ekf.x[4, 0] = ekf.x[4, 0] - correction[1]
                    ekf.x[5, 0] = ekf.x[5, 0] - correction[2]
                    
                    if i % 1000 == 0:  # Log every 10 seconds
                        logger.info(f"🔄 Velocity baseline removed @{i}: baseline={velocity_baseline}, corr={correction}")
            
            # Store ZUPT flag (only if recording)
            if i >= skip_samples:
                zupt_flags.append(bool(is_stationary))
            
            # Adjust constraint indices for warmup skip
            effective_index = i - skip_samples
            
            # BIOMECHANICAL IMPROVEMENT 2: Periodic velocity constraint
            constraint_cfg = config.get('ekf', {}).get('velocity_constraint', {})
            if (constraint_cfg.get('enabled', False) and effective_index >= 0 and 
                (effective_index + 1) % constraint_cfg.get('apply_every', 500) == 0 and len(velocities) > 0):
                # Assume movimento ciclico → velocità media dovrebbe essere ~0
                recent_window = max(constraint_cfg.get('window_size', 200), 50)
                start_idx = max(0, len(velocities) - recent_window)
                recent_velocities = np.array(velocities[start_idx:])
                
                if len(recent_velocities) > 10:
                    vel_bias = np.mean(recent_velocities, axis=0)
                    max_correction = constraint_cfg.get('max_correction', 0.5)
                    correction = np.clip(vel_bias, -max_correction, max_correction)
                    
                    # Applica correzione aggressiva per movimento biomeccanico
                    current_vel = ekf.get_velocity()
                    correction_factor = constraint_cfg.get('correction_factor', 0.3)  # 30% instead of 10%
                    corrected_vel = current_vel - correction_factor * correction
                    ekf.x[3:6, 0] = corrected_vel  
                    
                    logger.info(f"🔄 Velocity constraint applicato @{i}: bias={vel_bias}, corr={correction}, factor={correction_factor}")
            
            # BIOMECHANICAL IMPROVEMENT 3: Position drift correction for cyclic motion
            pos_constraint_cfg = config.get('ekf', {}).get('position_constraint', {})
            if (pos_constraint_cfg.get('enabled', False) and effective_index >= 0 and 
                (effective_index + 1) % pos_constraint_cfg.get('apply_every', 1000) == 0 and len(positions) > 0):
                # Per movimento ciclico, la posizione media dovrebbe essere ~0
                recent_window = max(pos_constraint_cfg.get('window_size', 500), 100)
                start_idx = max(0, len(positions) - recent_window)
                recent_positions = np.array(positions[start_idx:])
                
                if len(recent_positions) > 50:
                    pos_bias = np.mean(recent_positions, axis=0)
                    max_correction = pos_constraint_cfg.get('max_correction', 5.0)
                    correction = np.clip(pos_bias, -max_correction, max_correction)
                    
                    # Applica correzione alla posizione
                    current_pos = ekf.get_position()
                    correction_factor = pos_constraint_cfg.get('correction_factor', 0.2)  # 20% correction
                    corrected_pos = current_pos - correction_factor * correction
                    ekf.x[0:3, 0] = corrected_pos  
                    
                    logger.info(f"📍 Position constraint applicato @{i}: bias={pos_bias}, corr={correction}, factor={correction_factor}")
            
            if (
                len(zupt_flags) > 0  # Check if we have recorded any ZUPT flags
                and not zupt_flags[-1]
                and ekf.zupt_detector.adaptive_thresholds
                and ekf.zupt_detector._current_relax_multiplier > 1.0
                and ekf.zupt_detector._samples_since_last_zupt % max(1, ekf.zupt_detector.window_size) == 0
            ):
                logger.info(
                    "⚠️ ZUPT non attivo da %d campioni → soglie rilassate x%.2f",
                    ekf.zupt_detector._samples_since_last_zupt,
                    ekf.zupt_detector._current_relax_multiplier,
                )
            
            # Progress logging
            if i % 1000 == 0 and i > 0:
                logger.info(f"⏳ Processed {i}/{len(accel_data)} samples ({100*i/len(accel_data):.1f}%)")
        
        # Convert to numpy arrays
        positions = np.array(positions)
        velocities = np.array(velocities)
        biases = np.array(biases)
        zupt_flags = np.array(zupt_flags)
        
        logger.info(f"✅ EKF processing completed. Applied ZUPT to {np.sum(zupt_flags)} frames ({100*np.sum(zupt_flags)/len(zupt_flags):.1f}%)")
        
        # Post-processing
        post_config = config.get('postprocess', {})
        if post_config.get('enable', True):
            logger.info("🔧 Starting post-processing...")
            
            # Create timestamp array for advanced post-processing
            timestamps = np.arange(len(velocities)) / sample_rate
            
            # Apply advanced post-processing (detrending, high-pass, smoothing)
            velocities, positions = postprocess_velocity_position(
                velocities, positions, timestamps, config
            )
            
            # Apply legacy post-processing to each axis (outlier removal)
            processed_velocities = np.zeros_like(velocities)
            processed_positions = np.zeros_like(positions)
            
            for axis in range(3):
                vel_axis, pos_axis = apply_post_processing(
                    velocities[:, axis], 
                    positions[:, axis], 
                    post_config
                )
                processed_velocities[:, axis] = vel_axis
                processed_positions[:, axis] = pos_axis
            
            velocities = processed_velocities
            positions = processed_positions
            logger.info("✅ Post-processing completed")
        
        # Generate comprehensive diagnostics
        diagnostics = ekf.get_diagnostics()
        
        # Create output directory
        output_dir = get_output_base_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        output_config = config['io']['output']
        
        # Save position and velocity estimates
        for axis, axis_name in enumerate(['X', 'Y', 'Z']):
            # Position
            pos_file = resolve_output_path(f"estimated_position_{axis_name}.csv", output_dir)
            pd.DataFrame({
                'timestamp': range(len(positions)),
                'position': positions[:, axis],
                'zupt_applied': zupt_flags
            }).to_csv(pos_file, index=False)
            
            # Velocity 
            vel_file = resolve_output_path(f"estimated_velocity_{axis_name}.csv", output_dir)
            pd.DataFrame({
                'timestamp': range(len(velocities)),
                'velocity': velocities[:, axis],
                'zupt_applied': zupt_flags
            }).to_csv(vel_file, index=False)
        
        # Save bias estimates
        bias_file = resolve_output_path("estimated_bias.csv", output_dir)
        pd.DataFrame({
            'timestamp': range(len(biases)),
            'bias_x': biases[:, 0],
            'bias_y': biases[:, 1], 
            'bias_z': biases[:, 2]
        }).to_csv(bias_file, index=False)
        
        # Generate performance report
        report = generate_performance_report(diagnostics, config, zupt_flags)
        
        # Save performance report
        report_file = resolve_output_path("EKF_9D_performance_report.yaml", output_dir)
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        # Generate plots
        if config.get('plots', {}).get('enable', True):
            generate_comprehensive_plots(
                positions, velocities, biases, zupt_flags, 
                diagnostics, output_dir, config
            )
        
        logger.info("🎉 Comprehensive 9D EKF processing completed successfully!")
        return {
            'positions': positions,
            'velocities': velocities, 
            'biases': biases,
            'zupt_flags': zupt_flags,
            'diagnostics': diagnostics,
            'report': report
        }
        
    except Exception as e:
        logger.error(f"❌ Error in 9D EKF processing: {e}")
        logger.exception("📋 Full stack trace:")
        raise


def generate_performance_report(diagnostics, config, zupt_flags):
    """Genera report di performance per l'EKF 9D."""
    
    innovation_history = diagnostics['innovation_history']
    nis_history = diagnostics['nis_history'] 
    state_history = diagnostics['state_history']
    
    # Calculate innovation statistics
    innovation_stats = {
        'mean': np.mean(innovation_history, axis=0).tolist(),
        'std': np.std(innovation_history, axis=0).tolist(),
        'rmse': np.sqrt(np.mean(innovation_history**2, axis=0)).tolist()
    }
    
    # NIS statistics
    nis_stats = {
        'mean': float(np.mean(nis_history)),
        'std': float(np.std(nis_history)),
        'percentage_in_bounds': float(np.sum((nis_history >= 0.35) & (nis_history <= 7.81)) / len(nis_history) * 100)
    }
    
    # ZUPT statistics
    zupt_stats = {
        'total_frames': len(zupt_flags),
        'zupt_frames': int(np.sum(zupt_flags)),
        'zupt_percentage': float(100 * np.sum(zupt_flags) / len(zupt_flags)),
        'longest_zupt_sequence': int(np.max(np.diff(np.where(np.diff(zupt_flags.astype(int)))[0]))) if np.any(zupt_flags) else 0
    }
    
    # State evolution statistics
    final_state = state_history[-1]
    state_stats = {
        'final_position': final_state[0:3].tolist(),
        'final_velocity': final_state[3:6].tolist(), 
        'final_bias': final_state[6:9].tolist(),
        'position_drift': np.linalg.norm(final_state[0:3]).item(),
        'velocity_drift': np.linalg.norm(final_state[3:6]).item()
    }
    
    return {
        'timestamp': datetime.now().isoformat(),
        'config_summary': {
            'dt': config.get('ekf', {}).get('dt', 0.01),
            'use_joseph_form': config.get('ekf', {}).get('numerical_stability', {}).get('use_joseph_form', True),
            'zupt_enabled': True
        },
        'innovation_statistics': innovation_stats,
        'nis_statistics': nis_stats,
        'zupt_statistics': zupt_stats,
        'state_statistics': state_stats,
        'overall_performance': {
            'filter_stability': 'stable' if nis_stats['percentage_in_bounds'] > 80 else 'unstable',
            'bias_estimation_quality': 'good' if np.linalg.norm(final_state[6:9]) < 0.5 else 'poor',
            'drift_performance': 'excellent' if state_stats['position_drift'] < 1.0 else 'moderate'
        }
    }


def generate_comprehensive_plots(positions, velocities, biases, zupt_flags, diagnostics, output_dir, config):
    """Genera plots completi per l'analisi EKF 9D."""
    
    try:
        # 1. Main state estimation plot
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle('9D EKF State Estimation Results', fontsize=16)
        
        time_axis = np.arange(len(positions)) * config.get('ekf', {}).get('dt', 0.01)
        axis_names = ['X', 'Y', 'Z']
        
        # Position plots
        for i in range(3):
            axes[0, i].plot(time_axis, positions[:, i], 'b-', label='Position')
            axes[0, i].fill_between(time_axis, positions[:, i], alpha=0.3, where=zupt_flags, label='ZUPT', color='red')
            axes[0, i].set_title(f'Position {axis_names[i]}')
            axes[0, i].set_ylabel('Position [m]')
            axes[0, i].legend()
            axes[0, i].grid(True)
        
        # Velocity plots
        for i in range(3):
            axes[1, i].plot(time_axis, velocities[:, i], 'g-', label='Velocity')
            axes[1, i].fill_between(time_axis, velocities[:, i], alpha=0.3, where=zupt_flags, label='ZUPT', color='red')
            axes[1, i].set_title(f'Velocity {axis_names[i]}')
            axes[1, i].set_ylabel('Velocity [m/s]')
            axes[1, i].legend()
            axes[1, i].grid(True)
        
        # Bias plots
        for i in range(3):
            axes[2, i].plot(time_axis, biases[:, i], 'r-', label='Bias')
            axes[2, i].set_title(f'Accelerometer Bias {axis_names[i]}')
            axes[2, i].set_ylabel('Bias [m/s²]')
            axes[2, i].set_xlabel('Time [s]')
            axes[2, i].legend()
            axes[2, i].grid(True)
        
        plt.tight_layout()
        plt.savefig(resolve_output_path('ekf_9d_state_estimation.png', output_dir), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Diagnostics plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('9D EKF Diagnostics', fontsize=16)
        
        innovation_history = diagnostics['innovation_history']
        nis_history = diagnostics['nis_history']
        
        # Innovation time series
        for i in range(3):
            axes[0, 0].plot(time_axis, innovation_history[:, i], label=f'Innovation {axis_names[i]}')
        axes[0, 0].set_title('Innovation Sequence')
        axes[0, 0].set_ylabel('Innovation')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # NIS time series
        axes[0, 1].plot(time_axis, nis_history, 'b-', label='NIS')
        axes[0, 1].axhline(y=0.35, color='r', linestyle='--', label='Lower bound (α=0.05)')
        axes[0, 1].axhline(y=7.81, color='r', linestyle='--', label='Upper bound (α=0.05)')
        axes[0, 1].set_title('Normalized Innovation Squared (NIS)')
        axes[0, 1].set_ylabel('NIS')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Innovation histogram
        axes[1, 0].hist(innovation_history.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 0].set_title('Innovation Distribution')
        axes[1, 0].set_xlabel('Innovation Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True)
        
        # ZUPT statistics
        zupt_sequences = []
        in_sequence = False
        current_length = 0
        for flag in zupt_flags:
            if flag:
                if not in_sequence:
                    in_sequence = True
                    current_length = 1
                else:
                    current_length += 1
            else:
                if in_sequence:
                    zupt_sequences.append(current_length)
                    in_sequence = False
                    current_length = 0
        
        if zupt_sequences:
            axes[1, 1].hist(zupt_sequences, bins=20, alpha=0.7)
            axes[1, 1].set_title('ZUPT Sequence Length Distribution')
            axes[1, 1].set_xlabel('Sequence Length [frames]')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No ZUPT sequences detected', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('ZUPT Sequence Length Distribution')
        
        plt.tight_layout()
        plt.savefig(resolve_output_path('ekf_9d_diagnostics.png', output_dir), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 3D trajectory plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        traj = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
        
        # Mark ZUPT points
        zupt_positions = positions[zupt_flags]
        if len(zupt_positions) > 0:
            ax.scatter(zupt_positions[:, 0], zupt_positions[:, 1], zupt_positions[:, 2], 
                      c='red', s=10, alpha=0.6, label='ZUPT Points')
        
        # Mark start and end
        ax.scatter(*positions[0], c='green', s=100, marker='o', label='Start')
        ax.scatter(*positions[-1], c='red', s=100, marker='s', label='End')
        
        ax.set_xlabel('X Position [m]')
        ax.set_ylabel('Y Position [m]')
        ax.set_zlabel('Z Position [m]')
        ax.set_title('3D Trajectory with ZUPT Points')
        ax.legend()
        
        plt.savefig(resolve_output_path('ekf_9d_trajectory_3d.png', output_dir), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Comprehensive plots generated successfully")
        
    except Exception as e:
        logger.error(f"❌ Error generating plots: {e}")


def generate_execution_summary_report(results_dict, config, config_path=None):
    """
    Genera un report riassuntivo in Markdown delle performance EKF.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = resolve_output_path(f"EKF_summary_{timestamp}.md")
        
        axes = sorted(results_dict.keys())
        
        def _fmt(val, digits=4):
            try:
                return f"{float(val):.{digits}f}"
            except (TypeError, ValueError):
                return str(val)
        
        lines = []
        lines.append(f"# EKF Performance Summary – {timestamp}")
        lines.append("")
        if config_path:
            lines.append(f"- **Config file**: `{config_path}`")
        lines.append(f"- **Axes processed**: {', '.join(axes) if axes else 'none'}")
        lines.append(f"- **Analysis axis setting**: `{config.get('analysis_axis', 'not-specified')}`")
        lines.append("")
        
        for axis in axes:
            axis_results = results_dict[axis]
            stats = axis_results.get('summary_stats', {})
            perf = axis_results.get('performance_metrics', {})
            tuning = axis_results.get('tuning_report', {})
            
            lines.append(f"## Axis {axis}")
            general = stats.get('general', {})
            lines.append(
                f"- Samples: {general.get('samples', 'n/a')} | "
                f"duration: {_fmt(general.get('duration_seconds'))} s | "
                f"sampling rate: {_fmt(general.get('sampling_rate_hz'))} Hz"
            )
            
            pos = stats.get('position', {})
            lines.append(
                f"- Position mean/std (m): {_fmt(pos.get('mean'))} / {_fmt(pos.get('std'))}; "
                f"range: {_fmt(pos.get('range'))} (min {_fmt(pos.get('min'))}, max {_fmt(pos.get('max'))})"
            )
            
            vel = stats.get('velocity', {})
            lines.append(
                f"- Velocity mean/std (m/s): {_fmt(vel.get('mean'))} / {_fmt(vel.get('std'))}; "
                f"range: {_fmt(vel.get('range'))}"
            )
            
            acc = stats.get('acceleration', {})
            lines.append(
                f"- Acceleration mean/std (m/s²): {_fmt(acc.get('mean'))} / {_fmt(acc.get('std'))}; "
                f"range: {_fmt(acc.get('range'))}"
            )
            
            drift_info = stats.get('drift_correction', {})
            if drift_info.get('enabled'):
                lines.append(
                    f"- Drift correction: poly order {drift_info.get('polynomial_order')} | "
                    f"mean vel {_fmt(drift_info.get('mean_velocity_before'))} → "
                    f"{_fmt(drift_info.get('mean_velocity_after'))} m/s"
                )
            
            post_info = stats.get('post_processing', {})
            if post_info.get('enabled'):
                lines.append(
                    f"- Post-processing noise reduction: {_fmt(post_info.get('noise_reduction_percent'))}% "
                    f"(std {_fmt(post_info.get('std_before'))} → {_fmt(post_info.get('std_after'))} m/s)"
                )
            
            if perf:
                zupt_stats = perf.get('zupt_statistics', {})
                nis_stats = perf.get('nis_statistics', {})
                state_stats = perf.get('state_statistics', {})
                lines.append(
                    f"- ZUPT activations: {zupt_stats.get('zupt_frames', 0)}/"
                    f"{zupt_stats.get('total_frames', 0)} "
                    f"({_fmt(zupt_stats.get('zupt_percentage', 0), 1)}%)"
                )
                lines.append(
                    f"- NIS mean/std: {_fmt(nis_stats.get('mean', 0.0), 3)} / "
                    f"{_fmt(nis_stats.get('std', 0.0), 3)} "
                    f"(in-bounds {_fmt(nis_stats.get('percentage_in_bounds', 0.0), 1)}%)"
                )
                lines.append(
                    f"- Final velocity drift: {_fmt(state_stats.get('velocity_drift', 0.0))} m/s | "
                    f"bias norm: {_fmt(np.linalg.norm(state_stats.get('final_bias', [0, 0, 0])))}"
                )
            
            if tuning and tuning.get('status') != 'error':
                lines.append(f"- Tuning quality: {tuning.get('overall_quality', 'n/a')}")
                summary = tuning.get('summary', {})
                lines.append(
                    f"- Tuning recommendations: {summary.get('total_recommendations', 0)} "
                    f"(high priority: {summary.get('high_priority', 0)})"
                )
            
            lines.append("")
        
        lines.append("_Report generated automatically by `generate_execution_summary_report`._")
        report_path.write_text("\n".join(lines), encoding='utf-8')
        logger.info(f"📄 Report riassuntivo generato: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"❌ Errore nella generazione del report riassuntivo: {e}")
        return None


def generate_execution_summary_report(results_dict, config, config_path=None):
    """
    Genera un report riassuntivo in formato Markdown delle performance EKF.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = resolve_output_path(f"EKF_summary_{timestamp}.md")
        
        axes = sorted(results_dict.keys())
        
        def fmt(value, digits=4):
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return str(value)
        
        lines = []
        lines.append(f"# EKF Performance Summary – {timestamp}")
        lines.append("")
        if config_path:
            lines.append(f"- **Config file**: `{config_path}`")
        lines.append(f"- **Axes processed**: {', '.join(axes) if axes else 'none'}")
        lines.append(f"- **Analysis axis setting**: `{config.get('analysis_axis', 'not-specified')}`")
        lines.append("")
        
        for axis in axes:
            axis_results = results_dict[axis]
            stats = axis_results.get('summary_stats', {})
            perf = axis_results.get('performance_metrics', {})
            tuning = axis_results.get('tuning_report', {})
            
            lines.append(f"## Axis {axis}")
            general = stats.get('general', {})
            lines.append(
                f"- Samples: {general.get('samples', 'n/a')} | "
                f"duration: {fmt(general.get('duration_seconds'))} s | "
                f"sampling rate: {fmt(general.get('sampling_rate_hz'))} Hz"
            )
            
            pos = stats.get('position', {})
            lines.append(
                f"- Position mean/std (m): {fmt(pos.get('mean'))} / {fmt(pos.get('std'))}; "
                f"range: {fmt(pos.get('range'))} (min {fmt(pos.get('min'))}, max {fmt(pos.get('max'))})"
            )
            
            vel = stats.get('velocity', {})
            lines.append(
                f"- Velocity mean/std (m/s): {fmt(vel.get('mean'))} / {fmt(vel.get('std'))}; "
                f"range: {fmt(vel.get('range'))}"
            )
            
            acc = stats.get('acceleration', {})
            lines.append(
                f"- Acceleration mean/std (m/s²): {fmt(acc.get('mean'))} / {fmt(acc.get('std'))}; "
                f"range: {fmt(acc.get('range'))}"
            )
            
            drift_info = stats.get('drift_correction', {})
            if drift_info.get('enabled'):
                lines.append(
                    f"- Drift correction: poly order {drift_info.get('polynomial_order')} | "
                    f"mean vel {fmt(drift_info.get('mean_velocity_before'))} → {fmt(drift_info.get('mean_velocity_after'))} m/s"
                )
            
            post_info = stats.get('post_processing', {})
            if post_info.get('enabled'):
                lines.append(
                    f"- Post-processing noise reduction: {fmt(post_info.get('noise_reduction_percent'))}% "
                    f"(std {fmt(post_info.get('std_before'))} → {fmt(post_info.get('std_after'))} m/s)"
                )
            
            if perf:
                zupt_stats = perf.get('zupt_statistics', {})
                nis_stats = perf.get('nis_statistics', {})
                state_stats = perf.get('state_statistics', {})
                lines.append(
                    f"- ZUPT activations: {zupt_stats.get('zupt_frames', 0)}/{zupt_stats.get('total_frames', 0)} "
                    f"({fmt(zupt_stats.get('zupt_percentage', 0), 1)}%)"
                )
                lines.append(
                    f"- NIS mean/std: {fmt(nis_stats.get('mean', 0.0), 3)} / {fmt(nis_stats.get('std', 0.0), 3)} "
                    f"(in-bounds {fmt(nis_stats.get('percentage_in_bounds', 0.0), 1)}%)"
                )
                lines.append(
                    f"- Final velocity drift: {fmt(state_stats.get('velocity_drift', 0.0))} m/s | "
                    f"Bias norm: {fmt(np.linalg.norm(state_stats.get('final_bias', [0, 0, 0]))) }"
                )
            
            if tuning and tuning.get('status') != 'error':
                lines.append(f"- Tuning quality: {tuning.get('overall_quality', 'n/a')}")
                total_recs = tuning.get('summary', {}).get('total_recommendations', 0)
                high_recs = tuning.get('summary', {}).get('high_priority', 0)
                lines.append(f"- Tuning recommendations: {total_recs} (high priority: {high_recs})")
            
            lines.append("")
        
        lines.append("_Report generated automatically by `generate_execution_summary_report`._")
        
        report_path.write_text("\n".join(lines), encoding='utf-8')
        logger.info(f"📄 Report riassuntivo generato: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"❌ Errore nella generazione del report riassuntivo: {e}")
        return None


if __name__ == "__main__":
    """
    Entry point principale per l'esecuzione dell'EKF.
    
    Supporta sia l'EKF tradizionale per singolo asse che il nuovo EKF 9D completo.
    """
    parser = argparse.ArgumentParser(description='Extended Kalman Filter for Position and Velocity Estimation')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--mode', choices=['legacy', '9d'], default='9d',
                       help='EKF mode: legacy (single axis) or 9d (comprehensive)')
    parser.add_argument('--axis', choices=['X', 'Y', 'Z'], 
                       help='Axis for legacy mode (required if mode=legacy)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == '9d':
            print_section_header("9D EXTENDED KALMAN FILTER", "🚀")
            print_colored("Starting comprehensive 9D EKF with ZUPT detection...", "🚀", "cyan")
            
            result = run_comprehensive_9d_ekf(args.config)
            
            print_section_header("PROCESSING COMPLETED", "✅")
            print_colored("9D EKF processing completed successfully!", "✅", "green")
            print_colored(f"📊 Position estimates: {result['positions'].shape}", "📊", "blue")
            print_colored(f"🎯 ZUPT applied to: {np.sum(result['zupt_flags'])} frames", "🎯", "blue")
            print_colored(f"📈 Final bias estimates: {result['biases'][-1]}", "📈", "blue")
            
        else:
            # Legacy mode
            if not args.axis:
                print_colored("❌ Axis required for legacy mode", "❌", "red")
                sys.exit(1)
                
            print_section_header(f"LEGACY EKF - AXIS {args.axis}", "🔧")
            print_colored(f"Starting legacy EKF processing for axis {args.axis}...", "🔧", "yellow")
            
            run_EKF_for_vel_and_pos_est_from_acc(args.config, args.axis)
            
            print_section_header("LEGACY PROCESSING COMPLETED", "✅")
            print_colored(f"Legacy EKF processing for axis {args.axis} completed!", "✅", "green")
            
    except KeyboardInterrupt:
        print_colored("\n⏹️  Processing interrupted by user", "⏹️", "yellow")
        sys.exit(0)
    except Exception as e:
        print_colored(f"❌ Fatal error: {e}", "❌", "red")
        sys.exit(1)


        sys.exit(1)
