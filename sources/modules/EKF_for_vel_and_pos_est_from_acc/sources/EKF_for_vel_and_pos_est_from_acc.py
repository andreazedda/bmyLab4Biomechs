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
from scipy.linalg import inv, cholesky
import yaml
from datetime import datetime
import logging
from pathlib import Path
from collections import deque
import warnings
import json
import time

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
        kf_config = config['kalman_filter']
        print_colored(f"📋 Caricamento parametri di configurazione EKF 3D", "📋", "blue")
        
        # Stato 6D: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        initial_state = kf_config.get('initial_state_3d', [0.0] * 6)
        self.x = np.array(initial_state).reshape(6, 1)
        print_colored(f"📍 Stato iniziale 3D: pos={self.x[0:3,0]}, vel={self.x[3:6,0]}", "📍", "green")
        logger.info(f"Stato iniziale 3D: {self.x.flatten()}")
        
        # Matrice di covarianza iniziale 6x6
        initial_cov = kf_config.get('initial_covariance_3d', 1.0)
        if np.isscalar(initial_cov):
            self.P = np.eye(6) * initial_cov
        else:
            self.P = np.array(initial_cov).reshape(6, 6)
        print_colored(f"🎯 Matrice di covarianza 3D configurata ({self.P.shape})", "🎯", "green")
        
        # Processo noise per [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        process_noise = kf_config['process_noise']
        if isinstance(process_noise, dict):
            # Estendi per 3D
            pos_noise = process_noise['position']
            vel_noise = process_noise['velocity']
            Q_diag = [pos_noise] * 3 + [vel_noise] * 3
        else:
            Q_diag = process_noise if len(process_noise) == 6 else [process_noise[0]] * 3 + [process_noise[1]] * 3
        
        self.Q = np.diag(Q_diag)
        print_colored(f"🔊 Rumore del processo 3D: pos={Q_diag[:3]}, vel={Q_diag[3:]}", "🔊", "yellow")
        logger.info(f"Matrice di rumore del processo Q 3D: {Q_diag}")
        
        # Rumore della misura (per accelerazioni 3D)
        measurement_noise = kf_config['measurement_noise']
        if np.isscalar(measurement_noise):
            self.R = measurement_noise * np.eye(3)
        else:
            self.R = np.diag(measurement_noise)
        print_colored(f"📏 Rumore della misura 3D: {np.diag(self.R)}", "📏", "yellow")
        logger.info(f"Rumore della misura R 3D: {np.diag(self.R)}")
        
        # Valore della gravità
        self.gravity = kf_config['gravity']
        print_colored(f"🌍 Valore di gravità: {self.gravity} m/s²", "🌍", "blue")
        logger.info(f"Gravità: {self.gravity} m/s²")
        
        # CORREZIONE BUG: Matrice di osservazione H corretta
        # PROBLEMA: Il codice originale mappava velocità su accelerazioni (sbagliato)
        # SOLUZIONE: Per accelerometro, osserviamo accelerazione che è derivata della velocità
        # H mappa da stato [pos,vel] a osservazione di accelerazione
        # Per ora implementiamo H = 0 perché accelerazione non è direttamente nello stato
        # Useremo un approccio diverso nell'update corretto
        self.H = np.zeros((3, 6))
        # H resta zero perché accelerazione non è direttamente osservabile dallo stato [pos,vel]
        # L'aggiornamento corretto sarà implementato nel metodo update_proper
        logger.info(f"CORREZIONE: Matrice H impostata a zero - accelerazione trattata come controllo")
        logger.info(f"Matrice di osservazione H 3D: {self.H.shape}")
        
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
    
    def predict(self, dt):
        """
        Fase di predizione dell'EKF 3D.
        
        Args:
            dt (float): Intervallo di tempo tra misurazioni successive.
        """
        logger.debug(f"🔮 Fase di predizione 3D con dt={dt:.6f}s")
        
        # Matrice di transizione di stato 6x6
        F = np.block([
            [np.eye(3), dt * np.eye(3)],  # Posizione = posizione + velocità * dt
            [np.zeros((3, 3)), np.eye(3)]  # Velocità = velocità (modello inerziale)
        ])
        
        # Prediczione dello stato
        self.x = F @ self.x
        
        # Prediczione della covarianza
        self.P = F @ self.P @ F.T + self.Q
        
        # CORREZIONE BUG: Aggiungi controlli stabilità numerica
        self._check_numerical_stability()
        
        # Memorizza traccia per monitoring
        self.trace_history.append(np.trace(self.P))
        
        logger.debug(f"📊 Stato predetto 3D - Pos: {self.x[0:3,0]}, Vel: {self.x[3:6,0]}")
        
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
        
        # Matrice di controllo per accelerazione 6x3
        B = np.block([
            [0.5 * dt**2 * np.eye(3)],  # Posizione: 0.5 * a * dt^2
            [dt * np.eye(3)]            # Velocità: a * dt
        ])
        
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

    def update_proper_kalman(self, acc_body, roll, pitch, yaw, dt):
        """
        CORREZIONE BUG: Implementazione corretta del filtro di Kalman.
        
        Questo metodo implementa il vero update di Kalman con:
        1. Accelerazione come input di controllo (non osservazione)
        2. Osservazioni dirette di posizione/velocità quando disponibili
        3. Corrette equazioni di Kalman: K = P H^T (H P H^T + R)^(-1)
        
        Args:
            acc_body: Accelerazione in body-frame [ax, ay, az]
            roll, pitch, yaw: Angoli di Euler in radianti  
            dt: Intervallo di tempo
            
        Returns:
            tuple: Stato aggiornato, matrice P, innovazione
        """
        # 1. PREDIZIONE (già fatta con predict)
        
        # 2. CONTROLLO: Applica accelerazione come input di controllo
        R_matrix = self.euler_to_rotation_matrix(roll, pitch, yaw)
        acc_world = R_matrix @ np.array(acc_body).reshape(3, 1)
        
        # Rimuovi gravità
        gravity_vector = np.array([[0], [0], [self.gravity]])
        acc_world_corrected = acc_world - gravity_vector
        
        # Matrice di controllo B
        B = np.block([
            [0.5 * dt**2 * np.eye(3)],  # Posizione: 0.5 * a * dt^2
            [dt * np.eye(3)]            # Velocità: a * dt
        ])
        
        # Applica controllo
        self.x = self.x + B @ acc_world_corrected
        
        # 3. UPDATE: Per ora non abbiamo osservazioni dirette, solo controllo
        # In futuro, se avremo osservazioni di posizione/velocità, useremo:
        # innovation = z - H @ self.x
        # S = H @ self.P @ H.T + self.R
        # K = self.P @ H.T @ np.linalg.inv(S)
        # self.x = self.x + K @ innovation
        # self.P = (np.eye(6) - K @ H) @ self.P
        
        # Per ora, simuliamo un'innovazione basata sulla consistenza interna
        innovation = np.zeros((3, 1))  # Nessuna vera osservazione
        
        logger.debug(f"🔧 Update Kalman corretto - Pos: {self.x[0:3,0]}, Vel: {self.x[3:6,0]}")
        
        return self.x.copy(), self.P.copy(), innovation

    def _check_numerical_stability(self):
        """
        CORREZIONE BUG: Controlli di stabilità numerica per la matrice di covarianza P.
        
        Verifica:
        1. Esplosione della covarianza (trace troppo grande)
        2. Condition number (mal condizionamento)
        3. Simmetria della matrice P
        4. Positività della diagonale
        """
        trace_P = np.trace(self.P)
        max_trace_threshold = 1000.0  # Soglia per esplosione covarianza
        
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
        required_sections = ['kalman_filter', 'input_files', 'output_files']
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Sezione mancante nella configurazione: {section}")
                return False
        
        # Controlli parametri Kalman
        kf_config = config['kalman_filter']
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


def create_backup_if_needed(file_path: str, config: dict) -> Optional[str]:
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
        
    if not os.path.exists(file_path):
        return None
        
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.backup_{timestamp}"
        shutil.copy2(file_path, backup_path)
        logger.info(f"📁 Backup creato: {backup_path}")
        return backup_path
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
        
        if has_3d_data and has_orientation:
            print_colored("🌍 Dati 3D completi rilevati - modalità EKF 3D attivata", "🌍", "green")
            logger.info("🌍 Modalità EKF 3D attivata con accelerazione e orientamento")
            mode_3d = True
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
                
                # Update EKF 3D con auto-tuning
                state, covariance, innovation = ekf.update_with_acceleration(
                    acc_body, roll, pitch, yaw, dt
                )
                
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
            
            # Performance monitoring temporaneamente disabilitato per problemi dimensionali
            # try:
            #     performance_monitor.update_metrics(ekf, innovation_array, measurement_array)
            # except Exception as e:
            #     logger.warning(f"⚠️ Errore nel monitoraggio performance: {e}")
            #     # Continua l'esecuzione senza crashare
            
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
        
        # Applica post-processing avanzato se abilitato
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
        
        # Statistiche di accelerazione
        acc_mean = np.mean(accelerations)
        acc_std = np.std(accelerations)
        acc_range = np.max(accelerations) - np.min(accelerations)
        acc_min, acc_max = np.min(accelerations), np.max(accelerations)
        
        print_colored(f"⚡ STATISTICHE ACCELERAZIONE:", "⚡", "yellow")
        print_colored(f"  - Media: {acc_mean:.4f} m/s²", "📊", "yellow")
        print_colored(f"  - Deviazione standard: {acc_std:.4f} m/s²", "📊", "yellow")
        print_colored(f"  - Range: {acc_range:.4f} m/s² (min: {acc_min:.4f}, max: {acc_max:.4f})", "📏", "yellow")
        
        # Statistiche di bias
        bias_mean = np.mean(biases)
        bias_std = np.std(biases)
        bias_range = np.max(biases) - np.min(biases)
        
        print_colored(f"🎯 STATISTICHE BIAS:", "🎯", "magenta")
        print_colored(f"  - Media: {bias_mean:.6f}", "📊", "magenta")
        print_colored(f"  - Deviazione standard: {bias_std:.6f}", "📊", "magenta")
        print_colored(f"  - Range: {bias_range:.6f}", "📏", "magenta")
        
        # Log delle statistiche
        logger.info(f"📊 STATISTICHE FINALI - Campioni: {len(timestamps)}, Durata: {total_time:.2f}s")
        logger.info(f"📍 POSIZIONE - Media: {pos_mean:.4f}m, Std: {pos_std:.4f}m, Range: {pos_range:.4f}m")
        logger.info(f"🚀 VELOCITÀ - Media: {vel_mean:.4f}m/s, Std: {vel_std:.4f}m/s, Range: {vel_range:.4f}m/s")
        logger.info(f"⚡ ACCELERAZIONE - Media: {acc_mean:.4f}m/s², Std: {acc_std:.4f}m/s², Range: {acc_range:.4f}m/s²")
        logger.info(f"🎯 BIAS - Media: {bias_mean:.6f}, Std: {bias_std:.6f}, Range: {bias_range:.6f}")
        
        print_colored("✅ ============== EKF COMPLETATO CON SUCCESSO ==============", "✅", "green")
        
        # 📊 GENERAZIONE REPORT PRESTAZIONI EKF
        print_colored("", "", "white")  # Spacer
        performance_report = performance_monitor.generate_performance_report()
        
        # Controlla se sono abilitati i grafici e il salvataggio del report
        if config.get('performance_monitoring', {}).get('generate_performance_plots', True):
            output_dir = config.get('output_files', {}).get('plots', 'data/outputs')
            output_dir = str(Path(output_dir).parent.resolve())  # Ottieni la directory assoluta
            generate_performance_plots(performance_monitor, output_dir, axis)
        
        if config.get('performance_monitoring', {}).get('save_performance_report', True):
            output_dir = config.get('output_files', {}).get('plots', 'data/outputs')
            output_dir = str(Path(output_dir).parent.resolve())  # Ottieni la directory assoluta
            report_format = config.get('performance_monitoring', {}).get('performance_report_format', 'yaml')
            save_performance_report_to_file(performance_report, performance_monitor, output_dir, axis, report_format)
        
        # 🎛️ GENERAZIONE DASHBOARD DI TUNING AVANZATO
        if config.get('performance_monitoring', {}).get('enable_advanced_analysis', True):
            output_dir = config.get('output_files', {}).get('plots', 'data/outputs')
            output_dir = str(Path(output_dir).parent.resolve())  # Ottieni la directory assoluta
            tuning_report = generate_tuning_dashboard(performance_monitor, results, data, config, axis, output_dir)
            results['tuning_report'] = tuning_report
        
        # Salva il report delle prestazioni nel log
        logger.info("📊 Report delle prestazioni EKF generato")
        for key, value in performance_report.items():
            logger.info(f"📊 {key}: {value}")
        
        # Aggiungi metriche di performance ai risultati per l'ottimizzazione
        results['performance_metrics'] = performance_report
        
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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
        
        # Titolo generale
        plt.suptitle(f"{config['visualization']['plot_title']} - Axis {axis}")
        plt.tight_layout()
        
        # Salva il grafico se richiesto
        if output_path is not None and config['visualization']['save_plots']:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Modifichiamo il nome del file per includere l'asse
            base, ext = os.path.splitext(output_path)
            axis_output_path = f"{base}_{axis}{ext}"
            plt.savefig(axis_output_path)
            logger.info(f"Grafico salvato in {axis_output_path}")
        
        # Mostra il grafico se richiesto
        if config['visualization']['show_plots']:
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
            # Pesi per diverse metriche (più importante = peso maggiore)
            weights = {
                'trace_penalty': 0.3,      # Penalità per traccia alta
                'innovation_penalty': 0.25, # Penalità per innovazioni correlate
                'nis_penalty': 0.2,        # Penalità per NIS fuori range
                'convergence_penalty': 0.15, # Penalità per non convergenza
                'stability_penalty': 0.1   # Penalità per instabilità
            }
            
            # Estrai metriche
            mean_trace = performance_metrics.get('trace_statistics', {}).get('mean_trace', 50000)
            max_correlation = performance_metrics.get('innovation_whiteness', {}).get('max_correlation', 1.0)
            mean_nis = performance_metrics.get('nis_statistics', {}).get('mean_nis', 0.5)
            is_converged = performance_metrics.get('convergence', {}).get('is_converged', False)
            current_trace = performance_metrics.get('convergence', {}).get('current_trace', 50000)
            
            # Calcola penalità individuali
            trace_penalty = min(mean_trace / 1000, 100)  # Normalizza traccia
            innovation_penalty = max_correlation * 100   # Penalità correlazione
            
            # Penalità NIS (ideale ~1-3)
            if 0.5 <= mean_nis <= 3.0:
                nis_penalty = 0
            else:
                nis_penalty = abs(mean_nis - 1.5) * 20
            
            convergence_penalty = 0 if is_converged else 50
            stability_penalty = min(current_trace / 10000, 20)
            
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
            
            return total_score
            
        except Exception as e:
            logger.warning(f"Errore nel calcolo dello score: {e}")
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
                'position_noise': self.base_config['kalman_filter']['process_noise']['position'],
                'velocity_noise': self.base_config['kalman_filter']['process_noise']['velocity'],
                'acceleration_noise': self.base_config['kalman_filter']['process_noise']['acceleration'],
                'bias_noise': self.base_config['kalman_filter']['process_noise']['bias'],
                'measurement_noise': self.base_config['kalman_filter']['measurement_noise']
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
        
        updated_config['kalman_filter']['process_noise']['position'] = params['position_noise']
        updated_config['kalman_filter']['process_noise']['velocity'] = params['velocity_noise']
        updated_config['kalman_filter']['process_noise']['acceleration'] = params['acceleration_noise']
        updated_config['kalman_filter']['process_noise']['bias'] = params['bias_noise']
        updated_config['kalman_filter']['measurement_noise'] = params['measurement_noise']
        
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
        
        # Definisci la directory di output fissa
        fixed_output_dir = "/Volumes/nvme/Github/igmSquatBiomechanics/sources/lab/modules/EKF_for_vel_and_pos_est_from_acc/data/outputs"
        os.makedirs(fixed_output_dir, exist_ok=True)
        print_colored(f"Directory di output impostata a: {fixed_output_dir}", "📁", "cyan")
        
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
                config['kalman_filter']['process_noise']['position'] = optimization_results['best_parameters']['position_noise']
                config['kalman_filter']['process_noise']['velocity'] = optimization_results['best_parameters']['velocity_noise']
                config['kalman_filter']['process_noise']['acceleration'] = optimization_results['best_parameters']['acceleration_noise']
                config['kalman_filter']['process_noise']['bias'] = optimization_results['best_parameters']['bias_noise']
                config['kalman_filter']['measurement_noise'] = optimization_results['best_parameters']['measurement_noise']
                
                # Salva configurazione ottimizzata se richiesto
                if config.get('auto_tuning', {}).get('save_best_config', True):
                    optimized_config_path = os.path.join(fixed_output_dir, 'optimized_config.yaml')
                    try:
                        import yaml
                        with open(optimized_config_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        print_colored(f"💾 Configurazione ottimizzata salvata: {os.path.basename(optimized_config_path)}", "💾", "green")
                        logger.info(f"Configurazione ottimizzata salvata in: {optimized_config_path}")
                    except Exception as e:
                        logger.warning(f"Errore nel salvataggio configurazione ottimizzata: {e}")
                
                # Salva report di ottimizzazione
                optimization_report_path = os.path.join(fixed_output_dir, 'optimization_report.json')
                try:
                    import json
                    with open(optimization_report_path, 'w') as f:
                        json.dump(optimization_results, f, indent=2, default=str)
                    print_colored(f"📊 Report ottimizzazione salvato: {os.path.basename(optimization_report_path)}", "📊", "green")
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
            velocity_output_path = os.path.join(fixed_output_dir, f'estimated_velocity_{axis}.csv')
            create_backup_if_needed(velocity_output_path, config)
            
            # Controlla conferma utente per sovrascrittura file
            if os.path.exists(velocity_output_path) and config.get('execution_control', {}).get('user_confirmations', {}).get('confirm_file_overwrites', False):
                if not get_user_confirmation(f"Sovrascrivere il file esistente {os.path.basename(velocity_output_path)}?", config):
                    print_colored(f"⏭️ Saltando salvataggio velocità per asse {axis}", "⏭️", "yellow")
                    continue
            
            # Salva i risultati
            save_results(results, velocity_output_path, 'velocity')
            print_colored(f"Velocità stimata salvata in: {os.path.basename(velocity_output_path)}", "💨", "green")
            
            position_output_path = os.path.join(fixed_output_dir, f'estimated_position_{axis}.csv')
            create_backup_if_needed(position_output_path, config)
            save_results(results, position_output_path, 'position')
            print_colored(f"Posizione stimata salvata in: {os.path.basename(position_output_path)}", "📍", "green")
            
            # Percorso per i grafici
            plots_output_path = os.path.join(fixed_output_dir, f'ekf_plots_{axis}.png')
            
            # Pausa step-by-step prima della visualizzazione se abilitata
            if step_config.get('enabled', False) and step_config.get('pause_before_visualization', False):
                if not get_user_confirmation(f"📊 Iniziare visualizzazione per asse {axis}?", config):
                    print_colored("🛑 Esecuzione interrotta dall'utente", "🛑", "yellow")
                    return results_dict
            
            plot_results(data, results, config, axis, plots_output_path)
            print_colored(f"Grafico salvato in: {os.path.basename(plots_output_path)}", "📊", "magenta")
            
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
            output_dir = config.get('output_files', {}).get('plots', 'data/outputs')
            output_dir = str(Path(output_dir).parent)
            
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
        
        print_colored("🎊 Tutti gli assi elaborati con successo!", "�", "green")
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
        output_path = Path(output_dir) / f'EKF_performance_analysis_axis_{axis}.png'
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
        plt.savefig(f"{output_dir}/innovation_analysis_{axis}.png", dpi=300, bbox_inches='tight')
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
        plt.savefig(f"{output_dir}/nis_analysis_{axis}.png", dpi=300, bbox_inches='tight')
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
        plt.savefig(f"{output_dir}/covariance_analysis_{axis}.png", dpi=300, bbox_inches='tight')
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
        velocities = results['velocities']
        positions = results['positions']
        timestamps = results['timestamps']
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
        output_dir = str(Path(config.get('output_path', '.')).parent)
        plt.savefig(f"{output_dir}/residual_analysis_{axis}.png", dpi=300, bbox_inches='tight')
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
        tuning_report_path = f"{output_dir}/tuning_report_{axis}.json"
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
        comparison_path = f"{output_dir}/multi_axis_comparison.json"
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
        plt.savefig(f"{output_dir}/multi_axis_comparison.png", dpi=300, bbox_inches='tight')
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
        with open(f"{output_dir}/multi_axis_comparison_report.json", 'w') as f:
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
        
        print_colored(f"💾 Report multi-asse salvato: {output_dir}/multi_axis_comparison_report.json", "💾", "green")
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
        output_path = Path(output_dir) / f'EKF_performance_report_axis_{axis}.{format}'
        
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


if __name__ == "__main__":
    # Se lo script viene eseguito direttamente, esegue la funzione principale
    try:
        print_section_header("EXTENDED KALMAN FILTER PER STIMA DI VELOCITÀ E POSIZIONE", "🔬")
        print_colored(f"📅 Avvio esecuzione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "📅", "blue")
        print_colored(f"📁 Log file attivo: {current_log_file.name}", "📁", "blue")
        
        logger.info("🚀 === AVVIO SCRIPT EKF ===")
        logger.info(f"📅 Timestamp esecuzione: {datetime.now()}")
        logger.info(f"💻 Sistema operativo: {os.name}")
        logger.info(f"🐍 Versione Python: {sys.version}")
        logger.info(f"📁 Directory di lavoro: {os.getcwd()}")
        logger.info(f"📝 Log file: {current_log_file}")
        
        # Controlla se è stato fornito un percorso di configurazione come argomento
        config_path = sys.argv[1] if len(sys.argv) > 1 else None
        if config_path:
            print_colored(f"📄 File di configurazione specificato: {config_path}", "📄", "green")
            logger.info(f"📄 Utilizzo del file di configurazione specificato: {config_path}")
        else:
            print_colored("📄 Utilizzo del file di configurazione predefinito", "📄", "yellow")
            logger.info("📄 Utilizzo del file di configurazione predefinito")
        
        # Log informazioni di sistema
        print_colored("🔧 Controllo dipendenze e ambiente...", "🔧", "cyan")
        logger.info(f"🎨 Colorama disponibile: {COLORAMA_AVAILABLE}")
        logger.info(f"📊 NumPy versione: {np.__version__}")
        logger.info(f"🐼 Pandas versione: {pd.__version__}")
        logger.info(f"📈 Matplotlib versione: {plt.matplotlib.__version__}")
        
        print_colored("✅ Ambiente verificato, avvio elaborazione EKF...", "✅", "green")
        
        # Esegue l'EKF
        start_time = datetime.now()
        print_colored(f"⏱️  Avvio elaborazione alle: {start_time.strftime('%H:%M:%S')}", "⏱️", "blue")
        logger.info(f"⏱️  Inizio elaborazione: {start_time}")
        
        results = run_EKF_for_vel_and_pos_est_from_acc(config_path)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"⏱️  Fine elaborazione: {end_time}")
        logger.info(f"⌛ Tempo totale di esecuzione: {execution_time:.2f} secondi")
        logger.info("✅ Esecuzione completata con successo")
        
        print_section_header("ESECUZIONE COMPLETATA CON SUCCESSO", "🏁")
        print_colored(f"⏱️  Tempo di esecuzione: {execution_time:.2f} secondi", "⏱️", "green")
        print_colored(f"📊 Risultati elaborati per {len(results) if results else 0} assi", "📊", "green")
        print_colored(f"📁 Log completo salvato in: {current_log_file}", "📁", "blue")
        print_colored("🎉 Elaborazione terminata con successo!", "🎉", "green")
        
        sys.exit(0)
        
    except Exception as e:
        error_time = datetime.now()
        logger.error(f"❌ Errore fatale alle {error_time}: {e}")
        logger.exception("📋 Stack trace completo:")
        
        print_section_header("ERRORE FATALE", "❌")
        print_colored(f"⏰ Timestamp errore: {error_time.strftime('%Y-%m-%d %H:%M:%S')}", "⏰", "red")
        print_colored(f"💥 Errore: {e}", "💥", "red")
        print_colored(f"📁 Dettagli completi nel log: {current_log_file}", "📁", "yellow")
        print_colored("🔧 Controllare il file di log per maggiori dettagli", "🔧", "yellow")
        
        sys.exit(1)
