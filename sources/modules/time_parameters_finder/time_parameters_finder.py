from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Any, Tuple
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ==========================
# 0. Logger dedicato
# ==========================

class DebugLogger:
    """Logger dedicato per output di debug senza inquinare la console."""
    def __init__(self, log_path: str | None = None):
        if log_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"debug_squat_analysis_{timestamp}.log"
        self.log_path = log_path
        self.file = open(log_path, 'w', encoding='utf-8')
        self.console = True  # Set to False to suppress console output
        
    def log(self, msg: str, console: bool = False):
        """Scrivi su file e opzionalmente su console."""
        self.file.write(msg + "\n")
        self.file.flush()
        if console or self.console:
            print(msg)
    
    def close(self):
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

# Global logger instance
_logger: DebugLogger | None = None

def set_logger(logger: DebugLogger):
    global _logger
    _logger = logger

def log(msg: str, console: bool = False):
    """Utility per logging."""
    if _logger:
        _logger.log(msg, console)
    else:
        print(msg)


# ==========================
# 1. Definizioni base
# ==========================

class Phase(Enum):
    STAND = auto()
    ECC = auto()      # discesa (eccentrica)
    BOTTOM = auto()   # buca
    CONC = auto()     # salita (concentrica)
    UNKNOWN = auto()


@dataclass
class RepTime:
    T_ecc: float
    T_con: float
    T_buca: float
    T_top: float

    @property
    def T_work(self) -> float:
        return self.T_ecc + self.T_con

    @property
    def T_pause(self) -> float:
        return self.T_buca + self.T_top

    @property
    def T_TUT(self) -> float:
        return self.T_work + self.T_pause


@dataclass
class RepSegments:
    """Intervalli temporali delle fasi per una singola ripetizione."""
    t_ecc_start: float
    t_ecc_end: float
    t_bottom_start: float
    t_bottom_end: float
    t_conc_start: float
    t_conc_end: float
    t_top_start: float
    t_top_end: float


# ==========================
# 2. Utility numeriche
# ==========================

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    if window > len(x):
        window = len(x)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def numerical_derivative(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.gradient(x, t)
    return dx


def _safe_var(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.var(x, ddof=1))


def _safe_cv(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    m = float(np.mean(x))
    if np.isclose(m, 0.0):
        return 0.0
    return float(np.std(x, ddof=1) / m)


def _safe_trend(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 0.0
    idx = np.arange(1, n + 1, dtype=float)
    slope, _ = np.polyfit(idx, x, deg=1)
    return float(slope)


# ==========================
# 3. Loader CSV & scelta asse principale
# ==========================

def load_euler_from_csv(
    path: str,
    drop_duplicate_timestamps: bool = True,
    debug: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legge CSV: timestamp_ms, ang1, ang2, ang3
    Ritorna:
    - t: tempo in secondi (0 all'inizio)
    - angles: matrice shape (N, 3) con i tre assi
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["ts_ms", "ang1", "ang2", "ang3"],
    )
    
    # Remove rows with NaN values
    df = df.dropna()

    if drop_duplicate_timestamps:
        df = df.groupby("ts_ms", as_index=False).mean()

    df = df.sort_values("ts_ms").reset_index(drop=True)

    t0 = df["ts_ms"].iloc[0]
    t = (df["ts_ms"] - t0).to_numpy(dtype=float) / 1000.0  # ms -> s

    angles = df[["ang1", "ang2", "ang3"]].to_numpy(dtype=float)
    
    # DEBUG: Data loading diagnostics
    if debug:
        log(f"\n[DATA] N samples: {len(t)}, duration: {t[-1]-t[0]:.2f} s")
        dt = np.diff(t)
        log(f"[DATA] dt mean={dt.mean():.4f}s, std={dt.std():.4f}s, min={dt.min():.4f}s, max={dt.max():.4f}s")
        log(f"[DATA] ang ranges (deg):")
        for i, name in enumerate(["ang1", "ang2", "ang3"]):
            col = angles[:, i]
            log(f"   {name}: min={col.min():.1f}, max={col.max():.1f}, Δ={col.max()-col.min():.1f}")
        
        # Sanity check
        assert (dt > 0).all(), "⚠️  ERROR: Timestamp non strettamente crescente!"
    
    return t, angles


def select_principal_axis(angles: np.ndarray, debug: bool = True) -> int:
    """
    Seleziona automaticamente l’asse con rotazione principale
    usando la massima escursione (max-min).
    Ritorna 0, 1 o 2 (per ang1, ang2, ang3).
    """
    ranges = angles.max(axis=0) - angles.min(axis=0)
    principal_axis = int(np.argmax(np.abs(ranges)))
    return principal_axis


def axis_coherence_diagnostics(angles: np.ndarray, principal_axis: int, debug: bool = True) -> Dict[str, float]:
    """
    Calcola correlazioni tra asse principale e gli altri due, come diagnostica.
    """
    main = angles[:, principal_axis]
    corrs = {}
    if debug:
        log(f"\n[AXIS CORR] Correlations with principal axis:")
    for idx, name in zip(range(3), ["ang1", "ang2", "ang3"]):
        if idx == principal_axis:
            continue
        other = angles[:, idx]
        if np.std(other) < 1e-6 or np.std(main) < 1e-6:
            corr = np.nan
        else:
            corr = float(np.corrcoef(main, other)[0, 1])
        corrs[f"corr_principal_vs_{name}"] = corr
        if debug:
            log(f"   {name}: r={corr:.3f}")
    return corrs


# ==========================
# 4. Stima baseline STAND e livello BOTTOM
# ==========================

def estimate_stand_baseline(
    t: np.ndarray,
    theta: np.ndarray,
    window_sec: float = 1.0,
    debug: bool = True,
) -> Tuple[float, float]:
    duration = t[-1] - t[0]
    w = min(window_sec, max(duration / 4.0, 0.2))

    start_mask = t <= (t[0] + w)
    end_mask = t >= (t[-1] - w)

    theta_start = theta[start_mask]
    theta_end = theta[end_mask]

    mu_start, mu_end = float(np.mean(theta_start)), float(np.mean(theta_end))
    std_start, std_end = float(np.std(theta_start)), float(np.std(theta_end))

    mu_stand = 0.5 * (mu_start + mu_end)
    sigma_stand = max(std_start, std_end, 1e-3)  # evito zero

    if debug:
        log(f"\n[BASELINE] Stand estimation window: {w:.2f}s")
        log(f"[BASELINE] mu_start={mu_start:.2f}°, mu_end={mu_end:.2f}°")
        log(f"[BASELINE] mu_stand={mu_stand:.2f}°, σ_stand={sigma_stand:.2f}°")

    return mu_stand, sigma_stand


def estimate_bottom_level(
    theta: np.ndarray,
    omega: np.ndarray,
    mu_stand: float,
    vel_thresh: float,
    dist_factor: float = 2.0,
    debug: bool = True,
) -> float:
    """
    Stima livello medio della buca cercando punti:
    - velocità quasi nulla
    - lontani dalla baseline in piedi
    """
    stable_mask = np.abs(omega) < vel_thresh
    theta_stable = theta[stable_mask]

    if theta_stable.size == 0:
        idx_ext = np.argmax(np.abs(theta - mu_stand))
        mu_bottom = float(theta[idx_ext])
        if debug:
            log(f"[BASELINE] ⚠️  No stable points, using max deviation: mu_bottom={mu_bottom:.2f}°")
        return mu_bottom

    dist = np.abs(theta_stable - mu_stand)
    if dist.size == 0:
        idx_ext = np.argmax(np.abs(theta - mu_stand))
        mu_bottom = float(theta[idx_ext])
        if debug:
            log(f"[BASELINE] ⚠️  Empty dist, using max deviation: mu_bottom={mu_bottom:.2f}°")
        return mu_bottom

    thr = dist_factor * np.std(dist) if dist.size > 1 else dist[0]
    bottom_points = theta_stable[dist >= thr]

    if bottom_points.size == 0:
        idx = np.argmax(dist)
        mu_bottom = float(theta_stable[idx])
        if debug:
            log(f"[BASELINE] ⚠️  No points beyond threshold, using farthest: mu_bottom={mu_bottom:.2f}°")
        return mu_bottom

    mu_bottom = float(np.mean(bottom_points))
    
    if debug:
        delta = abs(mu_bottom - mu_stand)
        log(f"[BASELINE] mu_bottom={mu_bottom:.2f}°, Δ={delta:.2f}° from stand")
        log(f"[BASELINE] Bottom points used: {bottom_points.size} (stable: {theta_stable.size})")
        if delta < 20.0:
            log(f"[BASELINE] ⚠️  WARNING: Small Δ between stand/bottom ({delta:.1f}°) may cause classification issues!")
    
    return mu_bottom


# ==========================
# 5. Classificazione fasi per campione
# ==========================

def classify_phases(
    t: np.ndarray,
    theta: np.ndarray,
    mu_stand: float,
    mu_bottom: float,
    vel_thresh: float,
    stand_dist_factor: float = 2.0,
    smooth_window: int = 7,
) -> Tuple[List[Phase], np.ndarray]:
    """
    Ritorna:
    - lista delle fasi per campione
    - velocità angolare (omega)
    """
    theta_s = moving_average(theta, smooth_window)
    omega = numerical_derivative(t, theta_s)

    dist_stand = np.abs(theta_s - mu_stand)
    dist_bottom = np.abs(theta_s - mu_bottom)

    # euristica per soglia standing
    n_head = max(10, len(theta_s) // 20)
    sigma_stand = np.std(theta_s[:n_head]) if n_head > 0 else np.std(theta_s)
    stand_thr = stand_dist_factor * max(sigma_stand, 1e-3)

    phases = [Phase.UNKNOWN] * len(theta)

    for i in range(len(theta)):
        w = omega[i]
        ds = dist_stand[i]
        db = dist_bottom[i]
        is_slow = np.abs(w) < vel_thresh

        # For slow points, choose the closest baseline
        if is_slow:
            if ds < stand_thr and db < stand_thr:
                # Both close: choose the closer one
                if ds < db:
                    phases[i] = Phase.STAND
                else:
                    phases[i] = Phase.BOTTOM
            elif ds < stand_thr:
                phases[i] = Phase.STAND
            elif db < stand_thr:
                phases[i] = Phase.BOTTOM
            else:
                phases[i] = Phase.UNKNOWN
            continue

        # Fast movement: classify by velocity direction
        if np.abs(w) >= vel_thresh:
            if w > 0:
                phases[i] = Phase.ECC
            elif w < 0:
                phases[i] = Phase.CONC
            continue

        phases[i] = Phase.UNKNOWN

    return phases, omega


# ==========================
# 4b. Rough repetition counter (detector semplice)
# ==========================

def rough_rep_counter(
    t: np.ndarray,
    theta: np.ndarray,
    prominence: float = 5.0,
    debug: bool = True,
) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    """
    Detector semplice basato su minimi angolari (buca) per avere
    un numero di ripetizioni "golden reference" indipendente dall'algoritmo a fasi.
    """
    # Assumendo squat come sequenza di valli (buca)
    inverted = -theta
    peaks, props = find_peaks(inverted, prominence=prominence)
    
    if debug:
        log(f"\n[ROUGH] Rough detector: {len(peaks)} minimi (buca) con prominence={prominence}°")
        if len(peaks) > 0:
            log(f"[ROUGH] Peak times: {t[peaks].tolist()}")
            log(f"[ROUGH] Peak angles: {theta[peaks].tolist()}")
    
    return len(peaks), peaks, props


# ==========================
# 5. Consistency check: velocity-phase alignment
# ==========================

def check_velocity_phase_consistency(
    omega: np.ndarray,
    phases: List[Phase],
    vel_thresh: float,
    debug: bool = True,
) -> Dict[str, float]:
    """
    Verifica che la label ECC/CONC sia coerente con il segno della velocità.
    Ritorna percentuali di coerenza.
    """
    phases = np.array(phases)
    fast_mask = np.abs(omega) >= vel_thresh

    ecc_mask = fast_mask & (phases == Phase.ECC)
    conc_mask = fast_mask & (phases == Phase.CONC)

    ecc_pos = (omega[ecc_mask] > 0).mean() if ecc_mask.any() else np.nan
    conc_neg = (omega[conc_mask] < 0).mean() if conc_mask.any() else np.nan

    if debug:
        log(f"\n[CONSIST] Velocity-Phase consistency check:")
        log(f"[CONSIST] ECC: {ecc_pos*100:.1f}% dei campioni con ω>0 (dovrebbe essere >90%)")
        log(f"[CONSIST] CONC: {conc_neg*100:.1f}% dei campioni con ω<0 (dovrebbe essere >90%)")
        if ecc_pos < 0.9 or conc_neg < 0.9:
            log(f"[CONSIST] ⚠️  WARNING: Low consistency! Check vel_thresh or sign convention.")
    
    return {"ecc_consistency": ecc_pos, "conc_consistency": conc_neg}


# ==========================
# 5a. Scan raw phase pattern
# ==========================

def scan_raw_phase_pattern(
    t: np.ndarray,
    phases: List[Phase],
    debug: bool = True,
) -> Dict[str, Any]:
    """
    Detector di pattern ECC per capire se mancano BOTTOM/CONC.
    Utile per diagnosticare buca non riconosciuta.
    """
    phases_arr = np.array(phases)
    N = len(phases_arr)
    idx = 0
    count_ecc = 0
    ecc_with_bottom = 0
    ecc_with_conc = 0
    
    patterns = []
    
    while idx < N:
        try:
            pos = list(phases_arr[idx:]).index(Phase.ECC)
        except ValueError:
            break
        idx += pos
        
        # Da ECC prova a vedere se nei successivi X campioni appare BOTTOM e CONC
        lookahead = 500  # da tarare in funzione di fs
        window = phases_arr[idx:min(idx+lookahead, N)]
        has_bottom = Phase.BOTTOM in window
        has_conc = Phase.CONC in window
        
        count_ecc += 1
        if has_bottom:
            ecc_with_bottom += 1
        if has_conc:
            ecc_with_conc += 1
        
        patterns.append({
            "ecc_idx": idx,
            "ecc_time": float(t[idx]),
            "has_bottom": bool(has_bottom),
            "has_conc": bool(has_conc),
        })
        
        idx += 1
    
    if debug:
        log(f"\n[SCAN] Raw phase pattern scan:")
        log(f"[SCAN] ECC found {count_ecc} times")
        log(f"[SCAN]   with BOTTOM: {ecc_with_bottom}/{count_ecc} ({100*ecc_with_bottom/count_ecc:.1f}%)" if count_ecc > 0 else "[SCAN]   with BOTTOM: 0/0")
        log(f"[SCAN]   with CONC: {ecc_with_conc}/{count_ecc} ({100*ecc_with_conc/count_ecc:.1f}%)" if count_ecc > 0 else "[SCAN]   with CONC: 0/0")
        
        if count_ecc > 0 and (ecc_with_bottom < count_ecc or ecc_with_conc < count_ecc):
            log(f"[SCAN] ⚠️  Some ECC without BOTTOM/CONC → baseline/threshold issue!")
    
    return {
        "count_ecc": count_ecc,
        "ecc_with_bottom": ecc_with_bottom,
        "ecc_with_conc": ecc_with_conc,
        "patterns": patterns,
    }


# ==========================
# 5b. Filtro temporale delle fasi
# ==========================

def segment_stats(t: np.ndarray, phases: List[Phase], debug: bool = True) -> Dict[Phase, List[float]]:
    """
    Calcola statistiche sui segmenti contigui di ciascuna fase.
    Utile per diagnosticare frammentazione eccessiva.
    """
    phases_arr = np.array(phases)
    N = len(phases_arr)
    segs = {p: [] for p in Phase}
    i = 0
    while i < N:
        p = phases_arr[i]
        j = i + 1
        while j < N and phases_arr[j] == p:
            j += 1
        dur = t[j-1] - t[i] if j > i else 0.0
        segs[p].append(dur)
        i = j
    
    if debug:
        log(f"\n[SEG] Phase segment statistics:")
        for p in Phase:
            arr = np.array(segs[p])
            if arr.size == 0:
                log(f"   {p.name:8s}: 0 segments")
            else:
                log(f"   {p.name:8s}: n={arr.size:3d}, mean={arr.mean():.3f}s, "
                      f"min={arr.min():.3f}s, max={arr.max():.3f}s, std={arr.std():.3f}s")
    return segs


def filter_short_phases(
    t: np.ndarray,
    phases: List[Phase],
    min_duration: float = 0.1,
    debug: bool = False,
) -> Tuple[List[Phase], Dict[str, Any]]:
    """
    Rimuove segmenti di fase troppo corti sostituendoli con la fase adiacente più comune.
    Ritorna anche statistiche sul filtro per diagnostica.
    """
    phases_arr = np.array(phases, dtype=object)
    N = len(phases_arr)
    
    changed = 0
    seg_count = {p: 0 for p in Phase}
    seg_short = {p: 0 for p in Phase}
    
    i = 0
    while i < N:
        current_phase = phases_arr[i]
        
        # Trova la fine del segmento corrente
        j = i
        while j < N and phases_arr[j] == current_phase:
            j += 1
        
        seg_count[current_phase] += 1
        
        # Calcola durata del segmento
        if j - i > 0:
            seg_duration = t[min(j-1, N-1)] - t[i]
            
            # Se troppo corto, sostituisci con fase vicina
            if seg_duration < min_duration and (i > 0 or j < N):
                seg_short[current_phase] += 1
                
                # Scegli la fase prima o dopo (la più comune)
                before_phase = phases_arr[i-1] if i > 0 else None
                after_phase = phases_arr[j] if j < N else None
                
                if before_phase is not None and after_phase is not None:
                    # Usa la fase più comune tra prima e dopo
                    replacement = before_phase if before_phase == after_phase else before_phase
                elif before_phase is not None:
                    replacement = before_phase
                elif after_phase is not None:
                    replacement = after_phase
                else:
                    replacement = current_phase
                
                phases_arr[i:j] = replacement
                changed += (j - i)
        
        i = j
    
    stats = {
        "changed_samples": changed,
        "total_samples": N,
        "changed_fraction": changed / N if N > 0 else 0.0,
        "seg_count": seg_count,
        "seg_short": seg_short,
    }
    
    if debug:
        log(f"\n[FILTER] Filter statistics:")
        log(f"[FILTER] Changed samples: {changed}/{N} ({changed/N*100:.1f}%)")
        for p in Phase:
            log(f"[FILTER]   {p.name:8s}: seg={seg_count[p]}, short={seg_short[p]}")
    
    return phases_arr.tolist(), stats


# ==========================
# 6. Segmentazione ripetizioni
# ==========================

def segment_repetitions(
    t: np.ndarray,
    phases: List[Phase],
    min_phase_duration: float = 0.1,
) -> List[RepTime]:
    """
    Cerca pattern:
    STAND -> ECC -> BOTTOM -> CONC -> STAND
    e calcola i tempi delle fasi per ogni ripetizione.
    
    NOTA: Usa segment_repetitions_with_segments se hai bisogno anche
    degli intervalli temporali per plotting/tabelle.
    """
    phases = np.array(phases)
    N = len(phases)
    reps: List[RepTime] = []
    i = 0

    def seg_time(i_start: int, i_end: int) -> float:
        if i_start < 0 or i_end <= i_start:
            return 0.0
        return float(t[i_end] - t[i_start])

    while i < N:
        # ECC start
        while i < N and phases[i] != Phase.ECC:
            i += 1
        if i >= N:
            break
        idx_ecc_start = i

        while i < N and phases[i] == Phase.ECC:
            i += 1
        idx_ecc_end = i - 1
        if i >= N:
            break

        if phases[i] != Phase.BOTTOM:
            continue

        idx_bottom_start = i
        while i < N and phases[i] == Phase.BOTTOM:
            i += 1
        idx_bottom_end = i - 1
        if i >= N:
            break

        if phases[i] != Phase.CONC:
            continue

        idx_conc_start = i
        while i < N and phases[i] == Phase.CONC:
            i += 1
        idx_conc_end = i - 1

        idx_top_start = i
        while i < N and phases[i] == Phase.STAND:
            i += 1
        idx_top_end = i - 1

        T_ecc = seg_time(idx_ecc_start, idx_ecc_end)
        T_buca = seg_time(idx_bottom_start, idx_bottom_end)
        T_con = seg_time(idx_conc_start, idx_conc_end)
        T_top = seg_time(idx_top_start, idx_top_end)

        if T_ecc < min_phase_duration or T_con < min_phase_duration:
            continue

        reps.append(RepTime(T_ecc=T_ecc, T_con=T_con, T_buca=T_buca, T_top=T_top))

    return reps


def segment_repetitions_with_segments(
    t: np.ndarray,
    phases: List[Phase],
    min_phase_duration: float = 0.1,
    debug: bool = True,
) -> Tuple[List[RepTime], List[RepSegments]]:
    """
    Come segment_repetitions, ma in più restituisce gli intervalli temporali
    delle fasi per ogni ripetizione. Utile per plotting pulito e tabelle.
    """
    if debug:
        log("\n" + "="*80)
        log("🔍 [REP] segment_repetitions_with_segments STARTED")
        log("="*80)
    
    phases = np.array(phases)
    N = len(phases)
    
    if debug:
        log(f"[REP] Total samples: {N}")
        log(f"[REP] Time range: {t[0]:.3f}s to {t[-1]:.3f}s (duration: {t[-1]-t[0]:.3f}s)")
        log(f"[REP] Min phase duration threshold: {min_phase_duration}s")
        
        # Count phase distribution
        from collections import Counter
        phase_counts = Counter(phases)
        log(f"[REP] Phase distribution:")
        for phase_val in [Phase.STAND, Phase.ECC, Phase.BOTTOM, Phase.CONC, Phase.UNKNOWN]:
            count = phase_counts.get(phase_val, 0)
            pct = 100.0 * count / N if N > 0 else 0.0
            log(f"   {phase_val.name:8s}: {count:6d} samples ({pct:5.1f}%)")
    
    reps: List[RepTime] = []
    segments: List[RepSegments] = []
    debug_reps = []  # Track all candidates
    i = 0
    rep_number = 0

    def seg_time(i_start: int, i_end: int) -> float:
        if i_start < 0 or i_end <= i_start:
            return 0.0
        return float(t[i_end] - t[i_start])

    while i < N:
        # ECC start
        while i < N and phases[i] != Phase.ECC:
            i += 1
        if i >= N:
            break
        idx_ecc_start = i

        while i < N and phases[i] == Phase.ECC:
            i += 1
        idx_ecc_end = i - 1
        if i >= N:
            break

        if phases[i] != Phase.BOTTOM:
            continue

        idx_bottom_start = i
        while i < N and phases[i] == Phase.BOTTOM:
            i += 1
        idx_bottom_end = i - 1
        if i >= N:
            break

        if phases[i] != Phase.CONC:
            continue

        idx_conc_start = i
        while i < N and phases[i] == Phase.CONC:
            i += 1
        idx_conc_end = i - 1

        idx_top_start = i
        while i < N and phases[i] == Phase.STAND:
            i += 1
        idx_top_end = i - 1

        T_ecc = seg_time(idx_ecc_start, idx_ecc_end)
        T_buca = seg_time(idx_bottom_start, idx_bottom_end)
        T_con = seg_time(idx_conc_start, idx_conc_end)
        T_top = seg_time(idx_top_start, idx_top_end)

        # Constraints for minimum phase durations (more relaxed)
        MIN_BOTTOM = 0.05  # Minimum bottom phase duration (seconds)
        MIN_TOP = 0.03     # Minimum top phase duration (seconds)

        # DEBUG: Track all candidate patterns
        reason = "OK"
        if T_ecc < min_phase_duration:
            reason = f"SKIP: ecc too short ({T_ecc:.3f}s < {min_phase_duration:.3f}s)"
        elif T_con < min_phase_duration:
            reason = f"SKIP: conc too short ({T_con:.3f}s < {min_phase_duration:.3f}s)"
        elif T_buca < MIN_BOTTOM:
            reason = f"SKIP: bottom too short ({T_buca:.3f}s < {MIN_BOTTOM:.3f}s)"
        elif T_top < MIN_TOP:
            reason = f"SKIP: top too short ({T_top:.3f}s < {MIN_TOP:.3f}s)"
        
        debug_reps.append({
            "t_ecc": (t[idx_ecc_start], t[idx_ecc_end]),
            "t_bottom": (t[idx_bottom_start], t[idx_bottom_end]),
            "t_conc": (t[idx_conc_start], t[idx_conc_end]),
            "t_top": (t[idx_top_start], t[idx_top_end]) if idx_top_start < N else (0, 0),
            "T_ecc": T_ecc,
            "T_buca": T_buca,
            "T_con": T_con,
            "T_top": T_top,
            "reason": reason,
        })

        if reason != "OK":
            continue

        reps.append(RepTime(T_ecc=T_ecc, T_con=T_con, T_buca=T_buca, T_top=T_top))

        seg = RepSegments(
            t_ecc_start=float(t[idx_ecc_start]),
            t_ecc_end=float(t[idx_ecc_end]),
            t_bottom_start=float(t[idx_bottom_start]),
            t_bottom_end=float(t[idx_bottom_end]),
            t_conc_start=float(t[idx_conc_start]),
            t_conc_end=float(t[idx_conc_end]),
            t_top_start=float(t[idx_top_start]) if idx_top_start < N else float(t[idx_conc_end]),
            t_top_end=float(t[idx_top_end]) if idx_top_end < N else float(t[idx_conc_end]),
        )
        segments.append(seg)
        rep_number += 1

    if debug:
        log(f"\n[REP] Candidate patterns found: {len(debug_reps)}")
        log(f"[REP] ✅ Accepted repetitions: {len(reps)}")
        log(f"[REP] ❌ Rejected patterns: {len(debug_reps) - len(reps)}")
        
        if len(debug_reps) > 0:
            log(f"\n[REP] Detailed pattern analysis:")
            for k, r in enumerate(debug_reps, 1):
                status = "✅" if r['reason'] == "OK" else "❌"
                log(f"   {status} Pattern {k}: {r['reason']}")
                log(f"      T_ecc={r['T_ecc']:.3f}s, T_buca={r['T_buca']:.3f}s, "
                      f"T_con={r['T_con']:.3f}s, T_top={r['T_top']:.3f}s")
                log(f"      Time: ECC[{r['t_ecc'][0]:.2f},{r['t_ecc'][1]:.2f}], "
                      f"BOT[{r['t_bottom'][0]:.2f},{r['t_bottom'][1]:.2f}], "
                      f"CONC[{r['t_conc'][0]:.2f},{r['t_conc'][1]:.2f}]")
        
        log("="*80 + "\n")

    return reps, segments


def compute_stand_init_final(
    t: np.ndarray,
    phases: List[Phase],
) -> Tuple[float, float]:
    phases = np.array(phases)
    N = len(phases)

    idx = 0
    while idx < N and phases[idx] == Phase.STAND:
        idx += 1
    T_init = float(t[idx - 1] - t[0]) if idx > 0 else 0.0

    idx = N - 1
    while idx >= 0 and phases[idx] == Phase.STAND:
        idx -= 1
    T_final = float(t[-1] - t[idx + 1]) if idx < N - 1 else 0.0

    return T_init, T_final


# ==========================
# 7. Motore dei 46 parametri temporali
# ==========================

def compute_time_metrics(
    rep_times: List[RepTime],
    T_stand_init: float = 0.0,
    T_stand_final: float = 0.0,
) -> Dict[str, Any]:
    N_rep = len(rep_times)
    if N_rep == 0:
        return {
            "N_rep": 0,
            "T_stand_init": T_stand_init,
            "T_stand_final": T_stand_final,
        }

    T_ecc = np.array([r.T_ecc for r in rep_times], dtype=float)
    T_con = np.array([r.T_con for r in rep_times], dtype=float)
    T_buca = np.array([r.T_buca for r in rep_times], dtype=float)
    T_top = np.array([r.T_top for r in rep_times], dtype=float)
    T_work = np.array([r.T_work for r in rep_times], dtype=float)
    T_pause = np.array([r.T_pause for r in rep_times], dtype=float)
    T_TUT = np.array([r.T_TUT for r in rep_times], dtype=float)

    T_ecc_tot = float(T_ecc.sum())
    T_con_tot = float(T_con.sum())
    T_work_tot = float(T_work.sum())
    T_buca_tot = float(T_buca.sum())
    T_top_tot = float(T_top.sum())
    T_pause_tot = float(T_pause.sum())
    T_TUT_tot = float(T_TUT.sum())

    mean_T_ecc = float(T_ecc.mean())
    mean_T_con = float(T_con.mean())
    mean_T_buca = float(T_buca.mean())
    mean_T_top = float(T_top.mean())
    mean_T_work = float(T_work.mean())
    mean_T_pause = float(T_pause.mean())
    mean_T_TUT = float(T_TUT.mean())

    var_T_ecc = _safe_var(T_ecc)
    var_T_con = _safe_var(T_con)
    var_T_buca = _safe_var(T_buca)
    var_T_top = _safe_var(T_top)
    var_T_work = _safe_var(T_work)
    var_T_pause = _safe_var(T_pause)
    var_T_TUT = _safe_var(T_TUT)

    cv_ecc = _safe_cv(T_ecc)
    cv_con = _safe_cv(T_con)
    cv_buca = _safe_cv(T_buca)
    cv_top = _safe_cv(T_top)
    cv_work = _safe_cv(T_work)
    cv_pause = _safe_cv(T_pause)
    cv_TUT = _safe_cv(T_TUT)

    trend_T_ecc = _safe_trend(T_ecc)
    trend_T_con = _safe_trend(T_con)
    trend_T_pause = _safe_trend(T_pause)
    trend_T_TUT = _safe_trend(T_TUT)

    if T_pause_tot > 0:
        WP_ratio = T_work_tot / T_pause_tot
    else:
        WP_ratio = np.nan

    if T_TUT_tot > 0:
        useful_time_frac = T_work_tot / T_TUT_tot
    else:
        useful_time_frac = np.nan

    denom_density = T_work_tot + T_pause_tot
    if denom_density > 0:
        set_density = T_work_tot / denom_density
    else:
        set_density = np.nan

    metrics: Dict[str, Any] = {
        "N_rep": N_rep,
        "T_stand_init": float(T_stand_init),
        "T_stand_final": float(T_stand_final),

        "per_rep": {
            "T_ecc": T_ecc.tolist(),
            "T_con": T_con.tolist(),
            "T_buca": T_buca.tolist(),
            "T_top": T_top.tolist(),
            "T_work": T_work.tolist(),
            "T_pause": T_pause.tolist(),
            "T_TUT": T_TUT.tolist(),
        },

        "totals": {
            "T_ecc_tot": T_ecc_tot,
            "T_con_tot": T_con_tot,
            "T_work_tot": T_work_tot,
            "T_buca_tot": T_buca_tot,
            "T_top_tot": T_top_tot,
            "T_pause_tot": T_pause_tot,
            "T_TUT_tot": T_TUT_tot,
        },

        "means": {
            "mean_T_ecc": mean_T_ecc,
            "mean_T_con": mean_T_con,
            "mean_T_buca": mean_T_buca,
            "mean_T_top": mean_T_top,
            "mean_T_work": mean_T_work,
            "mean_T_pause": mean_T_pause,
            "mean_T_TUT": mean_T_TUT,
        },

        "variances": {
            "var_T_ecc": var_T_ecc,
            "var_T_con": var_T_con,
            "var_T_buca": var_T_buca,
            "var_T_top": var_T_top,
            "var_T_work": var_T_work,
            "var_T_pause": var_T_pause,
            "var_T_TUT": var_T_TUT,
        },

        "cv": {
            "cv_ecc": cv_ecc,
            "cv_con": cv_con,
            "cv_buca": cv_buca,
            "cv_top": cv_top,
            "cv_work": cv_work,
            "cv_pause": cv_pause,
            "cv_TUT": cv_TUT,
        },

        "trends": {
            "trend_T_ecc": trend_T_ecc,
            "trend_T_con": trend_T_con,
            "trend_T_pause": trend_T_pause,
            "trend_T_TUT": trend_T_TUT,
        },

        "ratios": {
            "WP_ratio": float(WP_ratio),
            "useful_time_frac": float(useful_time_frac),
            "set_density": float(set_density),
        },
    }

    return metrics


# ==========================
# 8. Plot delle fasi (angle vs time)
# ==========================

def plot_squat_phases(
    t: np.ndarray,
    theta: np.ndarray,
    phases: List[Phase],
    title: str = "Squat phases (sample-level debug)",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot angolo vs tempo con zone colorate sample-by-sample.
    NOTA: Questo produce la "tenda veneziana". Usa plot_squat_phases_by_rep
    per visualizzazioni pulite basate su ripetizioni validate.
    
    Utile solo per debugging della classificazione fine.
    """
    phases = np.array(phases)

    # Define distinct colors for each phase
    phase_colors = {
        Phase.STAND: ('#808080', 'Stand', 0.3),           # Gray
        Phase.ECC: ('#FF4444', 'Eccentric ↓', 0.35),     # Red
        Phase.BOTTOM: ('#FFA500', 'Bottom ●', 0.4),       # Orange
        Phase.CONC: ('#44FF44', 'Concentric ↑', 0.35),   # Green
        Phase.UNKNOWN: ('#CCCCCC', 'Unknown', 0.15),      # Light gray
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot angle trace with thicker line and darker color
    ax.plot(t, theta, color='#000080', linewidth=1.5, label="Angle", zorder=10)

    def shade_phase(phase: Phase):
        if phase not in phase_colors:
            return
        
        color, label, alpha = phase_colors[phase]
        mask = phases == phase
        if not np.any(mask):
            return
        
        # Find contiguous segments
        idx = np.where(mask)[0]
        starts = [idx[0]]
        ends = []
        for k in range(1, len(idx)):
            if idx[k] != idx[k-1] + 1:
                ends.append(idx[k-1])
                starts.append(idx[k])
        ends.append(idx[-1])

        # Plot first segment with label, rest without
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(t[s], t[e], color=color, alpha=alpha, 
                      label=label if i == 0 else None, zorder=1)

    # Shade phases in specific order for better visibility
    for phase in [Phase.STAND, Phase.ECC, Phase.BOTTOM, Phase.CONC, Phase.UNKNOWN]:
        shade_phase(phase)

    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    # Reorder: Angle first, then phases
    angle_idx = labels.index('Angle') if 'Angle' in labels else -1
    if angle_idx >= 0:
        handles = [handles[angle_idx]] + [h for i, h in enumerate(handles) if i != angle_idx]
        labels = [labels[angle_idx]] + [l for i, l in enumerate(labels) if i != angle_idx]
    
    ax.legend(handles, labels, loc='upper right', framealpha=0.9, fontsize=10)

    ax.set_xlabel("Time [s]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Angle [deg]", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add minor gridlines
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.1, linestyle=':')
    
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_squat_phases_by_rep(
    t: np.ndarray,
    theta: np.ndarray,
    rep_segments: List[RepSegments],
    title: str = "Squat phases by repetition",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot angle vs time, colorando SOLO i blocchi ECC/BOTTOM/CONC/TOP
    per ciascuna ripetizione validata. Niente micro-segmenti, niente UNKNOWN.
    
    Questo è il plot "pulito" per report e analisi dei tempi.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, theta, color='#000080', linewidth=1.5, label="Angle", zorder=10)

    colors = {
        "ecc": ("#FF4444", "Eccentric ↓", 0.35),
        "bottom": ("#FFA500", "Bottom ●", 0.40),
        "conc": ("#44FF44", "Concentric ↑", 0.35),
        "top": ("#808080", "Stand (top)", 0.25),
    }

    # Per evitare doppioni in legenda
    used_labels = set()

    for seg in rep_segments:
        blocks = [
            ("ecc", seg.t_ecc_start, seg.t_ecc_end),
            ("bottom", seg.t_bottom_start, seg.t_bottom_end),
            ("conc", seg.t_conc_start, seg.t_conc_end),
            ("top", seg.t_top_start, seg.t_top_end),
        ]
        for key, t_start, t_end in blocks:
            if t_end <= t_start:
                continue
            color, label, alpha = colors[key]
            lbl = label if label not in used_labels else None
            ax.axvspan(t_start, t_end, color=color, alpha=alpha, label=lbl, zorder=1)
            if lbl is not None:
                used_labels.add(label)

    ax.set_xlabel("Time [s]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Angle [deg]", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add minor gridlines for better readability
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.1, linestyle=':')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ==========================
# 9. Ottimizzazione adattiva dei parametri
# ==========================

def validate_pipeline_consistency(
    t: np.ndarray,
    phases: List[Phase],
    rep_times: List[RepTime],
    rep_segments: List[RepSegments],
    debug: bool = True,
) -> None:
    """
    Validazione strutturale della pipeline: invarianti e sanity check.
    """
    if debug:
        log(f"\n[VALID] Pipeline consistency validation:")
    
    # 1) lunghezze coerenti
    assert len(t) == len(phases), f"[VALID] ❌ ERROR: t e phases hanno lunghezze diverse ({len(t)} vs {len(phases)})"
    if debug:
        log(f"[VALID] ✅ Lengths match: t={len(t)}, phases={len(phases)}")
    
    # 2) ogni RepTime ha segmenti coerenti
    for k, (rt, seg) in enumerate(zip(rep_times, rep_segments), 1):
        # ricomponi tempi da segmenti e confronta
        T_ecc_seg = seg.t_ecc_end - seg.t_ecc_start
        T_con_seg = seg.t_conc_end - seg.t_conc_start
        T_buca_seg = seg.t_bottom_end - seg.t_bottom_start
        T_top_seg = seg.t_top_end - seg.t_top_start

        def close(a, b, tol=1e-3):
            return abs(a-b) <= tol

        assert close(rt.T_ecc, T_ecc_seg), f"[VALID] ❌ [REP {k}] T_ecc mismatch: {rt.T_ecc:.3f} vs {T_ecc_seg:.3f}"
        assert close(rt.T_con, T_con_seg), f"[VALID] ❌ [REP {k}] T_con mismatch: {rt.T_con:.3f} vs {T_con_seg:.3f}"
        assert close(rt.T_buca, T_buca_seg), f"[VALID] ❌ [REP {k}] T_buca mismatch: {rt.T_buca:.3f} vs {T_buca_seg:.3f}"
        assert close(rt.T_top, T_top_seg), f"[VALID] ❌ [REP {k}] T_top mismatch: {rt.T_top:.3f} vs {T_top_seg:.3f}"

        # 3) ordini temporali
        assert seg.t_ecc_start < seg.t_ecc_end <= seg.t_bottom_start <= seg.t_bottom_end \
               <= seg.t_conc_start <= seg.t_conc_end <= seg.t_top_start <= seg.t_top_end, \
               f"[VALID] ❌ [REP {k}] ordine temporale non valido"
    
    if debug:
        log(f"[VALID] ✅ All {len(rep_times)} repetitions validated successfully")


def dump_rep_time_table(rep_times: List[RepTime], debug: bool = True) -> None:
    """
    Tabella dettagliata dei tempi delle fasi per ogni ripetizione.
    """
    if not rep_times:
        if debug:
            log("[REPTAB] No reps")
        return
    
    if debug:
        log(f"\n[REPTAB] Repetition timing table:")
        log("[REPTAB] Rep |  T_ecc  T_buca   T_con   T_top   T_TUT")
        log("[REPTAB] ----|" + "-" * 42)
        for k, r in enumerate(rep_times, 1):
            log(f"[REPTAB] {k:3d} | {r.T_ecc:6.3f} {r.T_buca:7.3f} {r.T_con:7.3f} {r.T_top:7.3f} {r.T_TUT:7.3f}")


def local_param_sweep(
    t: np.ndarray,
    theta: np.ndarray,
    omega: np.ndarray,
    mu_stand: float,
    best_vel: float,
    best_dur: float,
    expected_reps: int | None = None,
    debug: bool = True,
) -> None:
    """
    Sweep sistematico 'intorno' al best per capire sensibilità parametri.
    """
    vel_values = [best_vel * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]]
    dur_values = [max(0.05, best_dur + d) for d in [-0.04, -0.02, 0.0, 0.02, 0.04]]

    if debug:
        log(f"\n[SWEEP] Local parameter sweep around best params:")
        log(f"[SWEEP] Best: vel={best_vel:.1f} deg/s, dur={best_dur:.2f}s")
        if expected_reps:
            log(f"[SWEEP] Target reps: {expected_reps}")
        log(f"[SWEEP] vel_thresh | min_dur | n_reps")
        log(f"[SWEEP] " + "-" * 35)
    
    for v in vel_values:
        for d in dur_values:
            mu_bottom = estimate_bottom_level(theta, omega, mu_stand, v, debug=False)
            phases, _ = classify_phases(t, theta, mu_stand, mu_bottom, v)
            phases, _ = filter_short_phases(t, phases, min_duration=d*0.5, debug=False)
            reps, _ = segment_repetitions_with_segments(t, phases, min_phase_duration=d, debug=False)
            marker = " ✅" if (expected_reps and len(reps) == expected_reps) else ""
            if debug:
                log(f"[SWEEP] {v:10.1f} | {d:7.2f} | {len(reps):6d}{marker}")


def generate_synthetic_squat(
    n_reps: int = 10,
    fs: float = 100.0,
    T_ecc: float = 0.5,
    T_buca: float = 0.3,
    T_con: float = 0.4,
    T_top: float = 0.6,
    noise_std: float = 1.0,
    stand_angle: float = -90,
    bottom_angle: float = -130,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un segnale sintetico di squat con N ripetizioni "perfette".
    Utile per test unitari e validazione algoritmo.
    """
    t_list = [0.0]
    angles = []

    def add_phase(T, start_angle, end_angle, n_steps):
        tt = np.linspace(0, T, n_steps, endpoint=False)
        if np.isclose(start_angle, end_angle):
            vals = start_angle + np.zeros_like(tt)
        else:
            vals = np.linspace(start_angle, end_angle, n_steps, endpoint=False)
        for dt, val in zip(tt, vals):
            t_list.append(t_list[-1] + 1/fs)
            angles.append(val + np.random.randn()*noise_std)

    for _ in range(n_reps):
        add_phase(T_ecc, stand_angle, bottom_angle, int(T_ecc*fs))
        add_phase(T_buca, bottom_angle, bottom_angle, int(T_buca*fs))
        add_phase(T_con, bottom_angle, stand_angle, int(T_con*fs))
        add_phase(T_top, stand_angle, stand_angle, int(T_top*fs))

    t = np.array(t_list[1:])
    angles_arr = np.array(angles)
    return t, angles_arr


def optimize_parameters(
    t: np.ndarray,
    theta: np.ndarray,
    smooth_window: int = 51,
    expected_reps: int | None = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Trova automaticamente i migliori parametri per massimizzare
    il numero di ripetizioni rilevate con buona qualità.
    
    Se expected_reps è specificato, cerca di trovare esattamente quel numero
    di ripetizioni, adattando i parametri in modo iterativo.
    
    Returns:
    --------
    best_vel_thresh, best_min_duration, diagnostics
    """
    theta_s = moving_average(theta, smooth_window)
    omega = numerical_derivative(t, theta_s)
    mu_stand, sigma_stand = estimate_stand_baseline(t, theta_s)
    
    # Test range of velocity thresholds based on data statistics
    omega_abs_mean = np.abs(omega).mean()
    omega_std = omega.std()
    
    # Expanded search range for adaptive finding
    vel_thresh_candidates = [
        omega_abs_mean * 0.1,
        omega_abs_mean * 0.2,
        omega_abs_mean * 0.3,
        omega_abs_mean * 0.5,
        omega_abs_mean * 0.8,
        omega_abs_mean * 1.0,
        omega_abs_mean * 1.5,
        omega_abs_mean * 2.0,
        omega_std * 0.03,
        omega_std * 0.05,
        omega_std * 0.08,
        omega_std * 0.1,
        omega_std * 0.15,
        omega_std * 0.2,
    ]
    vel_thresh_candidates = sorted(set([max(3.0, min(150.0, v)) for v in vel_thresh_candidates]))
    
    min_duration_candidates = [0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    best_score = -1
    best_params = None
    results = []
    
    for vel_thresh in vel_thresh_candidates:
        mu_bottom = estimate_bottom_level(theta_s, omega, mu_stand, vel_thresh)
        
        for min_duration in min_duration_candidates:
            phases, _ = classify_phases(t, theta_s, mu_stand, mu_bottom, vel_thresh, smooth_window)
            phases, _ = filter_short_phases(t, phases, min_duration=min_duration * 0.5, debug=False)
            
            reps, _ = segment_repetitions_with_segments(t, phases, min_phase_duration=min_duration)
            n_reps = len(reps)
            
            if n_reps == 0:
                continue
            
            # Score based on: number of reps, phase balance, and consistency
            from collections import Counter
            phase_counts = Counter(phases)
            
            # Check phase balance (want all 4 phases present)
            has_all_phases = all(phase_counts[p] > 0 for p in [Phase.STAND, Phase.ECC, Phase.BOTTOM, Phase.CONC])
            phase_balance = min(phase_counts.values()) / max(phase_counts.values()) if max(phase_counts.values()) > 0 else 0
            
            # Calculate rep consistency (low CV is better)
            if n_reps > 1:
                T_TUT = np.array([r.T_TUT for r in reps])
                cv_TUT = _safe_cv(T_TUT)
                consistency_score = 1.0 / (1.0 + cv_TUT)  # Lower CV = higher score
            else:
                consistency_score = 0.5
            
            # If expected_reps is specified, heavily penalize mismatch
            if expected_reps is not None:
                rep_match_score = 1.0 / (1.0 + abs(n_reps - expected_reps))
                # Exact match gets huge bonus
                if n_reps == expected_reps:
                    rep_match_score = 10.0
            else:
                rep_match_score = 1.0
            
            # Combined score
            score = n_reps * (1.0 + phase_balance) * consistency_score * (1.5 if has_all_phases else 1.0) * rep_match_score
            
            results.append({
                'vel_thresh': vel_thresh,
                'min_duration': min_duration,
                'n_reps': n_reps,
                'score': score,
                'has_all_phases': has_all_phases,
                'phase_balance': phase_balance,
                'consistency': consistency_score,
                'rep_match': rep_match_score if expected_reps is not None else None,
            })
            
            if score > best_score:
                best_score = score
                best_params = (vel_thresh, min_duration)
    
    if best_params is None:
        # Fallback to default
        best_params = (15.0, 0.2)
    
    # DEBUG: Best set for each n_reps
    best_by_n = {}
    for r in results:
        n = r['n_reps']
        if n not in best_by_n or r['score'] > best_by_n[n]['score']:
            best_by_n[n] = r
    
    if expected_reps is not None and expected_reps not in best_by_n:
        print(f"\n⚠️  [OPT] WARNING: No parameter combination found {expected_reps} repetitions!")
        print(f"[OPT] This indicates a segmentation issue, not just scoring.")
    
    diagnostics = {
        'tested_combinations': len(results),
        'best_score': best_score,
        'all_results': sorted(results, key=lambda x: x['score'], reverse=True)[:10],
        'expected_reps': expected_reps,
        'best_by_n_reps': best_by_n,
    }
    
    return best_params[0], best_params[1], diagnostics


# ==========================
# 10. Pipeline completa: file -> 46 parametri + plot
# ==========================

def analyze_squat_file_auto_axis(
    path: str,
    smooth_window: int = 51,
    vel_thresh: float | None = None,
    min_phase_duration: float | None = None,
    adaptive: bool = True,
    expected_reps: int | None = None,
    make_plot: bool = True,
    plot_path: str | None = None,
    debug: bool = True,
    export_debug_json: bool = True,
    log_path: str | None = None,
) -> Dict[str, Any]:
    """
    Pipeline completa con ottimizzazione adattiva e logging dedicato:
    - legge file CSV grezzo
    - sceglie asse con rotazione principale
    - ottimizza automaticamente i parametri (se adaptive=True)
    - stima baseline stand e buca
    - classifica fasi
    - segmenta ripetizioni
    - calcola i 46 parametri temporali
    - validazione completa con diagnostiche
    - (opzionale) genera plot con le zone temporali delle fasi
    
    Parameters:
    -----------
    smooth_window : int
        Window size for moving average smoothing (default: 51 for ~750Hz data)
    vel_thresh : float | None
        Velocity threshold in deg/s. If None and adaptive=True, will be optimized
    min_phase_duration : float | None
        Minimum duration in seconds. If None and adaptive=True, will be optimized
    adaptive : bool
        If True, automatically finds best parameters (default: True)
    expected_reps : int | None
        Expected number of repetitions. If provided, algorithm will adapt to find exactly this number
    debug : bool
        Enable detailed debug logging (default: True)
    export_debug_json : bool
        Export debug JSON file (default: True)
    log_path : str | None
        Path to log file. If None, auto-generated with timestamp
    """
    # Setup logger
    if log_path is None:
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(path))[0]
        log_path = f"debug_{base}_{timestamp}.log"
    
    logger = DebugLogger(log_path)
    logger.console = False  # Only file logging by default
    set_logger(logger)
    
    log("=" * 80, console=True)
    log(f"🔍 SQUAT ANALYSIS DEBUG LOG", console=True)
    log(f"File: {path}", console=True)
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", console=True)
    if expected_reps:
        log(f"🎯 Target repetitions: {expected_reps}", console=True)
    log("=" * 80, console=True)
    
    # Load data
    t, angles = load_euler_from_csv(path, debug=debug)
    principal_axis = select_principal_axis(angles, debug=debug)
    theta = angles[:, principal_axis]

    axis_diag = axis_coherence_diagnostics(angles, principal_axis, debug=debug)
    
    # Rough detector (golden reference)
    rough_n, rough_peaks, rough_props = rough_rep_counter(t, theta, prominence=5.0, debug=debug)

    # Adaptive parameter optimization
    if adaptive and (vel_thresh is None or min_phase_duration is None):
        log(f"\n[OPT] Starting parameter optimization...", console=True)
        if expected_reps:
            log(f"[OPT] Target: {expected_reps} repetitions", console=True)
        
        opt_vel_thresh, opt_min_duration, opt_diagnostics = optimize_parameters(
            t, theta, smooth_window, expected_reps=expected_reps
        )
        
        # FORZA parametri best per n=expected_reps se esiste
        if expected_reps is not None and opt_diagnostics is not None:
            best_by_n = opt_diagnostics.get("best_by_n_reps", {})
            if expected_reps in best_by_n:
                br = best_by_n[expected_reps]
                vel_thresh = br["vel_thresh"]
                min_phase_duration = br["min_duration"]
                log(f"[OPT] ⚡️ Forcing parameters for n={expected_reps}: vel_thresh={vel_thresh:.2f}, min_duration={min_phase_duration:.3f}", console=True)
            else:
                if vel_thresh is None:
                    vel_thresh = opt_vel_thresh
                if min_phase_duration is None:
                    min_phase_duration = opt_min_duration
                log(f"[OPT] ⚠️  No param set found for n={expected_reps}, using best overall", console=True)
        else:
            if vel_thresh is None:
                vel_thresh = opt_vel_thresh
            if min_phase_duration is None:
                min_phase_duration = opt_min_duration
        
        log(f"[OPT] Selected: vel_thresh={vel_thresh:.2f} deg/s, min_duration={min_phase_duration:.3f}s", console=True)
        if 'best_by_n_reps' in opt_diagnostics:
            log(f"\n[OPT] Best parameter set for each n_reps:")
            for n in sorted(opt_diagnostics['best_by_n_reps'].keys()):
                br = opt_diagnostics['best_by_n_reps'][n]
                log(f"   n={n}: vel={br['vel_thresh']:.1f} deg/s, dur={br['min_duration']:.2f}s, "
                      f"score={br['score']:.2f}")
    else:
        opt_diagnostics = None
        if vel_thresh is None:
            vel_thresh = 15.0
        if min_phase_duration is None:
            min_phase_duration = 0.2

    theta_s = moving_average(theta, smooth_window)
    omega = numerical_derivative(t, theta_s)

    mu_stand, sigma_stand = estimate_stand_baseline(t, theta_s, debug=debug)
    mu_bottom = estimate_bottom_level(theta_s, omega, mu_stand, vel_thresh, debug=debug)

    phases, omega = classify_phases(
        t=t,
        theta=theta_s,
        mu_stand=mu_stand,
        mu_bottom=mu_bottom,
        vel_thresh=vel_thresh,
        smooth_window=smooth_window,
    )
    
    # DEBUG: Phase distribution before filtering
    from collections import Counter
    phase_counts = Counter(phases)
    tot = len(phases)
    log(f"\n[PHASE] Phase counts (before filtering):")
    for p in Phase:
        count = phase_counts.get(p, 0)
        pct = 100.0 * count / tot if tot > 0 else 0.0
        log(f"   {p.name:8s}: {count:6d} ({pct:5.1f}%)")
    
    # Velocity-phase consistency check
    vel_consistency = check_velocity_phase_consistency(omega, phases, vel_thresh, debug=debug)
    
    # Scan raw phase patterns
    pattern_scan = scan_raw_phase_pattern(t, phases, debug=debug)
    
    # Apply temporal filtering to remove too-short phase segments
    # Usiamo direttamente min_phase_duration per essere più selettivi
    phases, filter_stats = filter_short_phases(t, phases, min_duration=min_phase_duration, debug=debug)
    
    # DEBUG: Segment statistics
    segment_stats(t, phases, debug=True)

    T_stand_init, T_stand_final = compute_stand_init_final(t, phases)

    rep_times, rep_segments = segment_repetitions_with_segments(
        t=t,
        phases=phases,
        min_phase_duration=min_phase_duration,
        debug=debug,
    )
    
    # Validation
    if len(rep_times) > 0:
        try:
            validate_pipeline_consistency(t, phases, rep_times, rep_segments, debug=debug)
        except AssertionError as e:
            log(f"\n[VALID] ❌ Validation failed: {e}", console=True)
    
    # Rep time table
    dump_rep_time_table(rep_times, debug=debug)
    
    # Local sweep if optimized
    if adaptive and expected_reps:
        local_param_sweep(t, theta_s, omega, mu_stand, vel_thresh, min_phase_duration, 
                         expected_reps=expected_reps, debug=debug)

    metrics = compute_time_metrics(
        rep_times=rep_times,
        T_stand_init=T_stand_init,
        T_stand_final=T_stand_final,
    )

    # Quality flags
    delta_stand_bottom = abs(mu_bottom - mu_stand)
    ecc_vs_conc_symmetry = float(np.mean([r.T_ecc for r in rep_times]) / np.mean([r.T_con for r in rep_times])) if len(rep_times) > 0 else np.nan
    phase_balance = float(min(phase_counts.values()) / max(phase_counts.values())) if max(phase_counts.values()) > 0 else 0.0
    
    quality_flags = {
        "delta_stand_bottom_deg": float(delta_stand_bottom),
        "ecc_vs_conc_symmetry": float(ecc_vs_conc_symmetry),
        "phase_balance": float(phase_balance),
        "filter_modified_frac": filter_stats["changed_fraction"],
        "velocity_phase_consistency_ecc": float(vel_consistency["ecc_consistency"]),
        "velocity_phase_consistency_conc": float(vel_consistency["conc_consistency"]),
        "rough_detector_n_reps": rough_n,
    }
    
    # Log quality summary
    log(f"\n{'='*80}")
    log(f"📊 QUALITY FLAGS SUMMARY")
    log(f"{'='*80}")
    log(f"Δ stand-bottom: {delta_stand_bottom:.1f}° {'✅' if delta_stand_bottom >= 20 else '⚠️  (< 20°)'}")
    log(f"ECC/CONC symmetry: {ecc_vs_conc_symmetry:.2f} {'✅' if 0.5 <= ecc_vs_conc_symmetry <= 2.0 else '⚠️'}")
    log(f"Phase balance: {phase_balance:.2f} {'✅' if phase_balance >= 0.3 else '⚠️  (low)'}")
    log(f"Filter modified: {filter_stats['changed_fraction']*100:.1f}% {'✅' if filter_stats['changed_fraction'] < 0.3 else '⚠️  (> 30%)'}")
    log(f"Vel-phase consistency ECC: {vel_consistency['ecc_consistency']*100:.1f}% {'✅' if vel_consistency['ecc_consistency'] > 0.9 else '⚠️  (< 90%)'}")
    log(f"Vel-phase consistency CONC: {vel_consistency['conc_consistency']*100:.1f}% {'✅' if vel_consistency['conc_consistency'] > 0.9 else '⚠️  (< 90%)'}")
    log(f"Rough detector: {rough_n} reps")
    log(f"Phase-based detector: {len(rep_times)} reps")
    if expected_reps:
        match = "✅ MATCH" if len(rep_times) == expected_reps else f"❌ MISMATCH (expected {expected_reps})"
        log(f"Expected: {expected_reps} reps → {match}")
    log(f"{'='*80}\n")

    metrics["principal_axis"] = principal_axis  # 0:ang1, 1:ang2, 2:ang3
    metrics["axis_diagnostics"] = axis_diag
    metrics["baseline"] = {
        "mu_stand": float(mu_stand),
        "sigma_stand": float(sigma_stand),
        "mu_bottom": float(mu_bottom),
    }
    metrics["parameters"] = {
        "vel_thresh": float(vel_thresh),
        "min_phase_duration": float(min_phase_duration),
        "smooth_window": int(smooth_window),
    }
    metrics["rep_segments"] = [seg.__dict__ for seg in rep_segments]
    metrics["quality_flags"] = quality_flags
    metrics["pattern_scan"] = pattern_scan
    metrics["filter_stats"] = filter_stats
    metrics["log_path"] = logger.log_path
    if opt_diagnostics is not None:
        metrics["optimization"] = opt_diagnostics

    # Export debug JSON
    if export_debug_json:
        import json
        import os
        debug_out = {
            "file": os.path.basename(path),
            "principal_axis": metrics["principal_axis"],
            "parameters": metrics["parameters"],
            "baseline": metrics["baseline"],
            "N_rep": metrics["N_rep"],
            "rep_segments": metrics["rep_segments"],
            "axis_diagnostics": axis_diag,
            "quality_flags": quality_flags,
        }
        if opt_diagnostics:
            debug_out["optimization_summary"] = {
                "tested_combinations": opt_diagnostics.get('tested_combinations', 0),
                "best_score": opt_diagnostics.get('best_score', 0),
                "expected_reps": opt_diagnostics.get('expected_reps'),
                "n_reps_found": list(opt_diagnostics.get('best_by_n_reps', {}).keys()),
            }
        
        debug_json_path = path.replace('.txt', '_debug.json')
        with open(debug_json_path, 'w') as f:
            json.dump(debug_out, f, indent=2)
        log(f"\n[DEBUG] Exported debug JSON to: {debug_json_path}", console=True)

    if make_plot:
        axis_name = ["ang1", "ang2", "ang3"][principal_axis]
        title = f"Squat phases on {axis_name}"
        plot_squat_phases_by_rep(
            t=t,
            theta=theta_s,
            rep_segments=rep_segments,
            title=title,
            save_path=plot_path,
            show=(plot_path is None),
        )
    
    # Close logger
    logger.close()
    
    return metrics


# ==========================
# 10. Esempio d’uso
# ==========================

if __name__ == "__main__":
    # file CSV con: timestamp_ms, ang1, ang2, ang3
    path = "/Volumes/nvme/Github/bmyLab4Biomechs/sources/modules/time_parameters_finder/FILE_EULER2025-10-28-10-23-31.txt"

    # USER: Set expected_reps if you know how many repetitions to expect
    # The algorithm will adapt to find exactly that number
    EXPECTED_REPS = 5  # Target: 5 ripetizioni pulite
    
    print("🔍 Analyzing squat data with ADAPTIVE parameter optimization + FULL DEBUG PROTOCOL...\n")
    if EXPECTED_REPS is not None:
        print(f"🎯 Target: {EXPECTED_REPS} repetitions (algorithm will adapt to find this number)\n")
    
    metrics = analyze_squat_file_auto_axis(
        path=path,
        smooth_window=101,  # Smoothing più aggressivo per ridurre micro-oscillazioni
        adaptive=True,  # Enable adaptive optimization
        expected_reps=EXPECTED_REPS,  # Will try to find exactly this many reps
        make_plot=True,
        plot_path="squat_fasi.png",
        debug=True,
        export_debug_json=True,
    )

    print("="*70)
    print("📊 SQUAT ANALYSIS RESULTS (console summary)")
    print("="*70)
    print(f"\n⚠️  Full debug output saved to: {metrics.get('log_path', 'debug log file')}")
    
    axis_names = ["ang1", "ang2", "ang3"]
    print(f"\n🎯 Principal axis: {axis_names[metrics['principal_axis']]} (axis {metrics['principal_axis']})")
    print(f"📈 Number of repetitions detected: {metrics.get('N_rep', 0)}")
    
    if EXPECTED_REPS is not None:
        n_found = metrics.get('N_rep', 0)
        if n_found == EXPECTED_REPS:
            print(f"✅ SUCCESS: Found exactly {EXPECTED_REPS} repetitions as expected!")
        else:
            print(f"⚠️  Found {n_found} reps, expected {EXPECTED_REPS} (difference: {abs(n_found - EXPECTED_REPS)})")
    
    # Quality flags summary
    if "quality_flags" in metrics:
        qf = metrics["quality_flags"]
        print(f"\n🏷️  Quality Flags:")
        print(f"   • Rough detector (simple): {qf['rough_detector_n_reps']} reps")
        print(f"   • Phase-based detector: {metrics.get('N_rep', 0)} reps")
        print(f"   • Δ stand-bottom: {qf['delta_stand_bottom_deg']:.1f}° {'✅' if qf['delta_stand_bottom_deg'] >= 20 else '⚠️  (< 20°)'}")
        print(f"   • Filter modified: {qf['filter_modified_frac']*100:.1f}% {'✅' if qf['filter_modified_frac'] < 0.3 else '⚠️  (> 30%)'}")
        print(f"   • Vel consistency: ECC {qf['velocity_phase_consistency_ecc']*100:.0f}%, CONC {qf['velocity_phase_consistency_conc']*100:.0f}%")
    
    if metrics.get("N_rep", 0) > 0:
        print(f"\n⏱️  Timing Summary (per-rep averages):")
        print(f"   • Eccentric: {metrics['means']['mean_T_ecc']:.2f} s")
        print(f"   • Bottom: {metrics['means']['mean_T_buca']:.2f} s")
        print(f"   • Concentric: {metrics['means']['mean_T_con']:.2f} s")
        print(f"   • TUT per rep: {metrics['means']['mean_T_TUT']:.2f} s")
        print(f"   • CV TUT: {metrics['cv']['cv_TUT']:.1%}")
    
    print(f"\n✅ Plot saved to: squat_fasi.png")
    print(f"📄 Check the full debug log for detailed diagnostics!")
    print(f"\n✅ Plot saved to: squat_fasi.png")
    print(f"📄 Check the full debug log for detailed diagnostics!")
    print("\n" + "="*70)