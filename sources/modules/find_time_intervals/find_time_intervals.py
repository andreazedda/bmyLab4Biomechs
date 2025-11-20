import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, find_peaks, periodogram, spectrogram, coherence, butter, filtfilt, decimate
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import adfuller, acf
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

# Helper functions for cycle and phase detection
def detect_cycles(signal, prominence=1.0, distance=100):
    """Detect cycles using peaks (e.g., squat repetitions)"""
    peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-signal, prominence=prominence, distance=distance)
    cycles = sorted(list(peaks) + list(troughs))
    return cycles, peaks, troughs

def detect_phases(signal, peaks, troughs):
    """Improved phase detection using linear regression on segments.
    Distinguishes between stable_high and stable_low regions based on median signal value."""
    phases = []
    idxs = sorted(list(peaks) + list(troughs))
    
    # Calculate median of entire signal to distinguish high vs low stable regions
    signal_median = np.median(signal)
    
    for i in range(len(idxs)-1):
        start, end = idxs[i], idxs[i+1]
        seg = signal[start:end]
        if len(seg) < 5:
            # For very short segments, classify as stable based on mean value
            seg_mean = np.mean(seg)
            ph = 'stable_high' if seg_mean > signal_median else 'stable_low'
            phases.append((start, end, ph))
            continue
        
        x = np.arange(len(seg))
        slope, _, _, _, _ = linregress(x, seg)
        
        if slope > 0.01:  # threshold to avoid noise
            ph = 'ascending'
        elif slope < -0.01:
            ph = 'descending'
        else:
            # Stable region - classify as high or low based on segment mean
            seg_mean = np.mean(seg)
            ph = 'stable_high' if seg_mean > signal_median else 'stable_low'
        
        phases.append((start, end, ph))
    return phases

def cross_axis_correlation(df, col1, col2):
    """Compute correlation between two columns"""
    return np.corrcoef(df[col1], df[col2])[0,1]

def cross_corr_lags(x, y, max_lag, fs):
    """Return lags (in samples and seconds) and cross-correlation normalized."""
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    corr_full = np.correlate(x, y, mode='full') / len(x)
    lags = np.arange(-len(x)+1, len(x))
    mid = len(corr_full) // 2
    idx = slice(mid - max_lag, mid + max_lag + 1)
    return lags[idx], corr_full[idx]

def compute_cycle_metrics(theta, ts, cycles):
    """Compute cycle-level statistics: duration, amplitude, trends."""
    cycle_durations = []
    cycle_amplitudes = []
    
    for i in range(len(cycles) - 1):
        i0, i1 = cycles[i], cycles[i+1]
        seg_t = ts[i1] - ts[i0]
        seg_amp = theta[i0:i1].max() - theta[i0:i1].min()
        cycle_durations.append(seg_t)
        cycle_amplitudes.append(seg_amp)
    
    cycle_durations = np.array(cycle_durations)
    cycle_amplitudes = np.array(cycle_amplitudes)
    
    result = {
        'durations': cycle_durations,
        'amplitudes': cycle_amplitudes,
        'mean_duration': cycle_durations.mean() if len(cycle_durations) > 0 else 0,
        'cv_duration': cycle_durations.std() / max(cycle_durations.mean(), 1e-9) if len(cycle_durations) > 0 else 0,
        'mean_amplitude': cycle_amplitudes.mean() if len(cycle_amplitudes) > 0 else 0,
        'cv_amplitude': cycle_amplitudes.std() / max(cycle_amplitudes.mean(), 1e-9) if len(cycle_amplitudes) > 0 else 0,
    }
    
    # Compute trends (fatigue indicators)
    if len(cycle_durations) > 1:
        idx = np.arange(len(cycle_durations))
        slope_d, *_ = linregress(idx, cycle_durations)
        slope_a, *_ = linregress(idx, cycle_amplitudes)
        result['trend_duration'] = slope_d
        result['trend_amplitude'] = slope_a
    else:
        result['trend_duration'] = 0
        result['trend_amplitude'] = 0
    
    return result

def compute_phase_statistics(phases, ts):
    """Compute statistics on phase types and durations."""
    phase_types = [p for _, _, p in phases]
    phase_durations = {
        'ascending': [], 
        'descending': [], 
        'stable_high': [],
        'stable_low': [],
        'stable': []  # Keep for backward compatibility
    }
    
    for s, e, p in phases:
        phase_durations[p].append(ts[e] - ts[s])
    
    phase_stats = {}
    for p, arr in phase_durations.items():
        if arr:
            arr = np.array(arr)
            phase_stats[p] = {
                'count': len(arr),
                'mean_duration': arr.mean(),
                'cv_duration': arr.std() / max(arr.mean(), 1e-9),
                'total_time': arr.sum()
            }
        else:
            phase_stats[p] = {'count': 0, 'mean_duration': 0, 'cv_duration': 0, 'total_time': 0}
    
    return phase_stats

def moving_average_filter(signal, window_size):
    """Apply moving average filter to remove noise.
    
    Args:
        signal: 1D numpy array
        window_size: integer, size of the moving average window (odd number recommended)
    
    Returns:
        filtered signal with same length as input
    """
    # Use convolution for moving average
    kernel = np.ones(window_size) / window_size
    # 'same' mode keeps the same length, padding at edges
    filtered = np.convolve(signal, kernel, mode='same')
    return filtered

def detect_exercise_repetitions(signal, ts, min_prominence=None, min_distance_sec=2.0):
    """Detect exercise repetitions (e.g., squats) using adaptive peak detection.
    
    This function identifies complete repetitions by finding valleys (bottom positions)
    with sufficient prominence and temporal separation.
    
    Args:
        signal: 1D numpy array of angle values
        ts: 1D numpy array of timestamps in seconds
        min_prominence: Minimum prominence for peak detection (auto if None)
        min_distance_sec: Minimum time between repetitions in seconds
    
    Returns:
        dict with repetition metrics and indices
    """
    # Auto-determine prominence if not provided (use 20% of signal range)
    if min_prominence is None:
        signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)
        min_prominence = signal_range * 0.2
    
    # Calculate sampling rate and minimum distance in samples
    fs = len(ts) / (ts[-1] - ts[0]) if len(ts) > 1 else 1.0
    min_distance_samples = int(min_distance_sec * fs)
    
    # Find valleys (troughs) - these represent bottom positions of reps
    troughs, trough_properties = find_peaks(-signal, 
                                             prominence=min_prominence,
                                             distance=min_distance_samples)
    
    num_reps = len(troughs)
    
    if num_reps == 0:
        return {
            'num_repetitions': 0,
            'trough_indices': np.array([]),
            'repetition_times': np.array([]),
            'inter_rep_intervals': np.array([]),
            'repetition_durations': np.array([]),
            'rep_ranges': np.array([]),
            'rep_depths': np.array([]),
        }
    
    # Compute inter-repetition intervals
    rep_times = ts[troughs]
    inter_rep_intervals = np.diff(rep_times) if num_reps > 1 else np.array([])
    
    # For each repetition, find the range of motion
    rep_ranges = []
    rep_depths = []
    rep_durations = []
    
    for i in range(num_reps):
        # Define window around trough (±50% of expected interval)
        if i == 0:
            # First rep: from start to midpoint to next trough
            if num_reps > 1:
                window_start = 0
                window_end = int((troughs[i] + troughs[i+1]) / 2)
            else:
                window_start = max(0, troughs[i] - int(min_distance_samples/2))
                window_end = min(len(signal)-1, troughs[i] + int(min_distance_samples/2))
        elif i == num_reps - 1:
            # Last rep: from midpoint of previous to end
            window_start = int((troughs[i-1] + troughs[i]) / 2)
            window_end = len(signal) - 1
        else:
            # Middle reps: from midpoint to previous to midpoint to next
            window_start = int((troughs[i-1] + troughs[i]) / 2)
            window_end = int((troughs[i] + troughs[i+1]) / 2)
        
        rep_segment = signal[window_start:window_end]
        rep_time_segment = ts[window_start:window_end]
        
        # Range of motion = max - min in this window
        rep_range = np.max(rep_segment) - np.min(rep_segment)
        rep_ranges.append(rep_range)
        
        # Depth = value at trough
        rep_depths.append(signal[troughs[i]])
        
        # Duration of this repetition
        rep_duration = rep_time_segment[-1] - rep_time_segment[0] if len(rep_time_segment) > 1 else 0
        rep_durations.append(rep_duration)
    
    # Identify suspicious repetitions (ROM < 50% of median ROM)
    rep_ranges_array = np.array(rep_ranges)
    median_rom = np.median(rep_ranges_array) if len(rep_ranges_array) > 0 else 0
    suspicious_flags = []
    
    for i, rom in enumerate(rep_ranges_array):
        if median_rom > 0 and rom < 0.5 * median_rom:
            suspicious_flags.append(True)
        else:
            suspicious_flags.append(False)
    
    num_suspicious = sum(suspicious_flags)
    
    return {
        'num_repetitions': num_reps,
        'trough_indices': troughs,
        'repetition_times': rep_times,
        'inter_rep_intervals': inter_rep_intervals,
        'repetition_durations': np.array(rep_durations),
        'rep_ranges': np.array(rep_ranges),
        'rep_depths': np.array(rep_depths),
        'prominence_threshold': min_prominence,
        'suspicious_reps': np.array(suspicious_flags),
        'num_suspicious': num_suspicious,
        'median_rom': median_rom,
    }

def validate_repetition_phases(signal, ts, trough_idx, window_start, window_end, phases, angle_tolerance_deg=5.0):
    """Validate if a detected repetition has proper phase sequence and cyclic behavior.
    
    A valid repetition MUST have:
    1. Phase sequence: descending→ascending OR ascending→descending (with optional stable phases)
    2. Cyclic behavior: signal returns to approximately the same angle (±tolerance)
    3. Standing position recovery: start and end angles are similar
    
    Args:
        signal: Full signal array
        ts: Full timestamp array
        trough_idx: Index of the detected trough/valley
        window_start: Start index of repetition window
        window_end: End index of repetition window
        phases: List of phase tuples (start_idx, end_idx, phase_type)
        angle_tolerance_deg: Maximum allowed difference between start/end angles
    
    Returns:
        dict with validation results: {
            'is_valid': bool,
            'has_proper_phases': bool,
            'is_cyclic': bool,
            'angle_recovery': float (difference in degrees),
            'phase_sequence': list of phase types in order
        }
    """
    # Find phases within this repetition window
    rep_phases = []
    for start_idx, end_idx, phase_type in phases:
        # Check if phase overlaps with repetition window
        if (start_idx >= window_start and start_idx < window_end) or \
           (end_idx > window_start and end_idx <= window_end) or \
           (start_idx <= window_start and end_idx >= window_end):
            rep_phases.append(phase_type)
    
    # Remove consecutive duplicates (e.g., stable-stable-stable → stable)
    phase_sequence = []
    for phase in rep_phases:
        if not phase_sequence or phase != phase_sequence[-1]:
            phase_sequence.append(phase)
    
    # Check for proper phase sequence
    has_descending = 'descending' in phase_sequence
    has_ascending = 'ascending' in phase_sequence
    
    # Valid sequences examples:
    # - descending → ascending
    # - ascending → descending
    # - descending → stable → ascending
    # - ascending → stable → descending
    has_proper_phases = has_descending and has_ascending
    
    # Check cyclic behavior: find peaks (max angles) before and after trough
    # Strategy: Find actual local peaks near the start and end of the window
    # This is more accurate than just taking max values
    
    # Search for peaks in the entire window
    from scipy.signal import find_peaks as fp
    peaks_indices, _ = fp(signal[window_start:window_end], prominence=1.0)
    peaks_indices = peaks_indices + window_start  # Convert to absolute indices
    
    # Find the peak closest to window_start (standing position before descent)
    peaks_before_trough = peaks_indices[peaks_indices < trough_idx]
    if len(peaks_before_trough) > 0:
        start_peak_idx = peaks_before_trough[-1]  # Last peak before trough
        start_angle = signal[start_peak_idx]
    else:
        # Fallback: use max in pre-trough region
        pre_trough_window = signal[window_start:trough_idx+1]
        if len(pre_trough_window) > 0:
            start_angle = np.max(pre_trough_window)
        else:
            start_angle = signal[window_start]
    
    # Find the peak closest to window_end (standing position after ascent)
    peaks_after_trough = peaks_indices[peaks_indices > trough_idx]
    if len(peaks_after_trough) > 0:
        end_peak_idx = peaks_after_trough[0]  # First peak after trough
        end_angle = signal[end_peak_idx]
    else:
        # Fallback: use max in post-trough region
        post_trough_window = signal[trough_idx:window_end]
        if len(post_trough_window) > 0:
            end_angle = np.max(post_trough_window)
        else:
            end_angle = signal[window_end - 1] if window_end > window_start else signal[window_start]
    
    angle_recovery = abs(end_angle - start_angle)
    
    is_cyclic = angle_recovery <= angle_tolerance_deg
    
    # Overall validation
    is_valid = has_proper_phases and is_cyclic
    
    return {
        'is_valid': bool(is_valid),
        'has_proper_phases': bool(has_proper_phases),
        'is_cyclic': bool(is_cyclic),
        'angle_recovery': float(angle_recovery),
        'phase_sequence': phase_sequence,
        'has_descending': bool(has_descending),
        'has_ascending': bool(has_ascending),
        # Debug info
        'start_angle': float(start_angle),
        'end_angle': float(end_angle),
        'window_start_idx': int(window_start),
        'window_end_idx': int(window_end),
        'trough_idx': int(trough_idx)
    }

def analyze_repetition_quality(rep_data):
    """Analyze quality metrics for exercise repetitions.
    
    Args:
        rep_data: dict from detect_exercise_repetitions()
    
    Returns:
        dict with quality metrics including suspicious repetition warnings
    """
    if rep_data['num_repetitions'] == 0:
        return {
            'consistency_score': 0.0,
            'range_cv': 0.0,
            'timing_cv': 0.0,
            'fatigue_indicator': 0.0,
            'tempo_consistency': 0.0,
            'suspicious_count': 0,
            'suspicious_percentage': 0.0,
        }
    
    ranges = rep_data['rep_ranges']
    durations = rep_data['repetition_durations']
    intervals = rep_data['inter_rep_intervals']
    
    # Coefficient of variation for range (lower = more consistent)
    range_cv = np.std(ranges) / (np.mean(ranges) + 1e-9) if len(ranges) > 0 else 0.0
    
    # Coefficient of variation for timing
    timing_cv = np.std(intervals) / (np.mean(intervals) + 1e-9) if len(intervals) > 0 else 0.0
    
    # Tempo consistency (duration CV)
    tempo_cv = np.std(durations) / (np.mean(durations) + 1e-9) if len(durations) > 0 else 0.0
    
    # Fatigue indicator: trend in range over repetitions (negative = fatigue)
    if len(ranges) > 2:
        rep_indices = np.arange(len(ranges))
        slope, *_ = linregress(rep_indices, ranges)
        fatigue_indicator = slope  # Negative means decreasing range (fatigue)
    else:
        fatigue_indicator = 0.0
    
    # Overall consistency score (inverse of average CV, 0-100 scale)
    avg_cv = (range_cv + timing_cv + tempo_cv) / 3.0
    consistency_score = max(0, 100 * (1 - avg_cv))
    
    # Add suspicious repetition statistics
    num_suspicious = rep_data.get('num_suspicious', 0)
    suspicious_percentage = (num_suspicious / rep_data['num_repetitions'] * 100) if rep_data['num_repetitions'] > 0 else 0.0
    
    return {
        'consistency_score': consistency_score,
        'range_cv': range_cv,
        'timing_cv': timing_cv,
        'tempo_cv': tempo_cv,
        'fatigue_indicator': fatigue_indicator,
        'suspicious_count': num_suspicious,
        'suspicious_percentage': suspicious_percentage,
    }


def compute_detailed_temporal_parameters(rep_data, phases, ts, theta):
    """Compute comprehensive temporal parameters for biomechanical analysis.
    
    Calculates 50 temporal parameters across 7 categories:
    A. Per-repetition micro-phase timings (8 params × N reps)
    B. Global timing parameters (4 params)
    C. Session total times (6 params)
    D. Temporal variances (16 params: 4 phases × 4 metrics)
    E. Regularity and continuity (6 params)
    F. Temporal outliers (4 params)
    G. Longitudinal temporal dynamics (6 params)
    
    Args:
        rep_data: dict from detect_exercise_repetitions()
        phases: list of (start_idx, end_idx, phase_type) tuples
        ts: timestamp array
        theta: angle signal array
        
    Returns:
        dict with detailed temporal parameters
    """
    if rep_data['num_repetitions'] == 0:
        return {
            'per_repetition': [],
            'global_timing': {},
            'session_totals': {},
            'temporal_variances': {},
            'regularity': {},
            'outliers': {},
            'longitudinal_trends': {}
        }
    
    num_reps = rep_data['num_repetitions']
    trough_indices = rep_data['trough_indices']
    rep_durations = np.array(rep_data['repetition_durations'])
    
    # Map phase types to biomechanical terms
    # ascending = concentric (rising), descending = eccentric (lowering)
    # stable_low = bottom hold, stable_high = top hold
    phase_mapping = {
        'ascending': 'concentric',
        'descending': 'eccentric',
        'stable_low': 'bottom_hold',
        'stable_high': 'top_hold'
    }
    
    # ========================================================================
    # A. PER-REPETITION MICRO-PHASE TIMINGS
    # ========================================================================
    per_rep_timings = []
    
    # Organize phases by repetition window
    fs = len(ts) / (ts[-1] - ts[0]) if len(ts) > 1 else 1.0
    min_distance_samples = int(2.0 * fs)
    
    for i, trough_idx in enumerate(trough_indices):
        # Define repetition window
        if i == 0:
            if num_reps > 1:
                window_start = 0
                window_end = int((trough_indices[i] + trough_indices[i+1]) / 2)
            else:
                window_start = max(0, trough_idx - int(min_distance_samples/2))
                window_end = min(len(theta)-1, trough_idx + int(min_distance_samples/2))
        elif i == num_reps - 1:
            window_start = int((trough_indices[i-1] + trough_idx) / 2)
            window_end = len(theta) - 1
        else:
            window_start = int((trough_indices[i-1] + trough_idx) / 2)
            window_end = int((trough_idx + trough_indices[i+1]) / 2)
        
        # Find phases in this window
        rep_phases = {
            'eccentric': 0.0,
            'concentric': 0.0,
            'bottom_hold': 0.0,
            'top_hold': 0.0
        }
        
        for start_idx, end_idx, phase_type in phases:
            # Check if phase overlaps with repetition window
            if end_idx < window_start or start_idx > window_end:
                continue
            
            # Calculate overlap
            overlap_start = max(start_idx, window_start)
            overlap_end = min(end_idx, window_end)
            
            if overlap_end > overlap_start:
                phase_duration = ts[overlap_end] - ts[overlap_start]
                biomech_phase = phase_mapping.get(phase_type, phase_type)
                if biomech_phase in rep_phases:
                    rep_phases[biomech_phase] += phase_duration
        
        # Total repetition time
        rep_total = rep_durations[i]
        
        # Normalized percentages
        rep_phases_pct = {
            'eccentric_pct': (rep_phases['eccentric'] / rep_total * 100) if rep_total > 0 else 0,
            'concentric_pct': (rep_phases['concentric'] / rep_total * 100) if rep_total > 0 else 0,
            'bottom_hold_pct': (rep_phases['bottom_hold'] / rep_total * 100) if rep_total > 0 else 0,
            'top_hold_pct': (rep_phases['top_hold'] / rep_total * 100) if rep_total > 0 else 0
        }
        
        per_rep_timings.append({
            'rep_num': i + 1,
            'eccentric_time': rep_phases['eccentric'],
            'concentric_time': rep_phases['concentric'],
            'bottom_hold_time': rep_phases['bottom_hold'],
            'top_hold_time': rep_phases['top_hold'],
            **rep_phases_pct,
            'total_time': rep_total
        })
    
    # ========================================================================
    # B. GLOBAL TIMING PARAMETERS
    # ========================================================================
    
    # Cycle time: time from start of rep to start of next rep
    cycle_times = []
    for i in range(num_reps - 1):
        cycle_time = ts[trough_indices[i+1]] - ts[trough_indices[i]]
        cycle_times.append(cycle_time)
    
    cycle_times = np.array(cycle_times) if len(cycle_times) > 0 else np.array([0])
    
    mean_cycle_time = np.mean(cycle_times) if len(cycle_times) > 0 else 0
    execution_frequency = (1.0 / mean_cycle_time) if mean_cycle_time > 0 else 0
    
    # Density time ratio: work time / total rep time
    # Work time = eccentric + concentric (excluding holds)
    work_times = np.array([r['eccentric_time'] + r['concentric_time'] for r in per_rep_timings])
    density_ratios = work_times / rep_durations if len(rep_durations) > 0 else np.array([0])
    mean_density_ratio = np.mean(density_ratios) if len(density_ratios) > 0 else 0
    
    global_timing = {
        'mean_rep_duration': float(np.mean(rep_durations)),
        'mean_cycle_time': float(mean_cycle_time),
        'execution_frequency_hz': float(execution_frequency),
        'mean_density_ratio': float(mean_density_ratio)
    }
    
    # ========================================================================
    # C. SESSION TOTAL TIMES
    # ========================================================================
    
    total_eccentric = sum(r['eccentric_time'] for r in per_rep_timings)
    total_concentric = sum(r['concentric_time'] for r in per_rep_timings)
    total_bottom_hold = sum(r['bottom_hold_time'] for r in per_rep_timings)
    total_top_hold = sum(r['top_hold_time'] for r in per_rep_timings)
    
    session_totals = {
        'total_eccentric_time': float(total_eccentric),
        'total_concentric_time': float(total_concentric),
        'total_bottom_hold_time': float(total_bottom_hold),
        'total_top_hold_time': float(total_top_hold),
        'total_work_time': float(total_eccentric + total_concentric),
        'total_pause_time': float(total_bottom_hold + total_top_hold)
    }
    
    # ========================================================================
    # D. TEMPORAL VARIANCES (4 phases × 4 metrics = 16 params)
    # ========================================================================
    
    eccentric_times = np.array([r['eccentric_time'] for r in per_rep_timings])
    concentric_times = np.array([r['concentric_time'] for r in per_rep_timings])
    bottom_hold_times = np.array([r['bottom_hold_time'] for r in per_rep_timings])
    top_hold_times = np.array([r['top_hold_time'] for r in per_rep_timings])
    
    def compute_variance_metrics(arr):
        """Compute variance, std, CV, range for an array."""
        arr = arr[arr > 0]  # Exclude zeros
        if len(arr) == 0:
            return {'var': 0, 'std': 0, 'cv': 0, 'range': 0}
        return {
            'var': float(np.var(arr)),
            'std': float(np.std(arr)),
            'cv': float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0,
            'range': float(np.max(arr) - np.min(arr))
        }
    
    temporal_variances = {
        'eccentric': compute_variance_metrics(eccentric_times),
        'concentric': compute_variance_metrics(concentric_times),
        'bottom_hold': compute_variance_metrics(bottom_hold_times),
        'top_hold': compute_variance_metrics(top_hold_times)
    }
    
    # ========================================================================
    # E. REGULARITY AND CONTINUITY
    # ========================================================================
    
    regularity = {
        'rep_duration_var': float(np.var(rep_durations)),
        'rep_duration_cv': float(np.std(rep_durations) / np.mean(rep_durations)) if np.mean(rep_durations) > 0 else 0,
        'rep_duration_range': float(np.max(rep_durations) - np.min(rep_durations)),
        'cycle_time_var': float(np.var(cycle_times)) if len(cycle_times) > 0 else 0,
        'cycle_time_cv': float(np.std(cycle_times) / np.mean(cycle_times)) if len(cycle_times) > 0 and np.mean(cycle_times) > 0 else 0,
        'cycle_time_range': float(np.max(cycle_times) - np.min(cycle_times)) if len(cycle_times) > 0 else 0
    }
    
    # ========================================================================
    # F. TEMPORAL OUTLIERS
    # ========================================================================
    
    median_eccentric = np.median(eccentric_times[eccentric_times > 0]) if np.sum(eccentric_times > 0) > 0 else 0
    median_concentric = np.median(concentric_times[concentric_times > 0]) if np.sum(concentric_times > 0) > 0 else 0
    
    # Thresholds for outliers
    slow_eccentric_threshold = median_eccentric * 2.0
    fast_concentric_threshold = median_concentric * 0.5
    excessive_bottom_hold_threshold = 2.0  # seconds
    excessive_top_hold_threshold = 2.0  # seconds
    
    outliers = {
        'num_slow_eccentric': int(np.sum(eccentric_times > slow_eccentric_threshold)),
        'num_fast_concentric': int(np.sum((concentric_times > 0) & (concentric_times < fast_concentric_threshold))),
        'num_excessive_bottom_hold': int(np.sum(bottom_hold_times > excessive_bottom_hold_threshold)),
        'num_excessive_top_hold': int(np.sum(top_hold_times > excessive_top_hold_threshold))
    }
    
    # ========================================================================
    # G. LONGITUDINAL TEMPORAL DYNAMICS (TRENDS)
    # ========================================================================
    
    def compute_trend(arr):
        """Compute linear trend (slope) for an array."""
        arr = np.array(arr)
        if len(arr) < 2:
            return 0.0
        indices = np.arange(len(arr))
        slope, *_ = linregress(indices, arr)
        return float(slope)
    
    longitudinal_trends = {
        'eccentric_trend': compute_trend(eccentric_times),
        'concentric_trend': compute_trend(concentric_times),
        'bottom_hold_trend': compute_trend(bottom_hold_times),
        'top_hold_trend': compute_trend(top_hold_times),
        'rep_duration_trend': compute_trend(rep_durations),
        'cycle_time_trend': compute_trend(cycle_times) if len(cycle_times) > 0 else 0
    }
    
    return {
        'per_repetition': per_rep_timings,
        'global_timing': global_timing,
        'session_totals': session_totals,
        'temporal_variances': temporal_variances,
        'regularity': regularity,
        'outliers': outliers,
        'longitudinal_trends': longitudinal_trends
    }


# ============================================================================
# ADVANCED FUNCTIONS - Production-ready biomechanical analysis
# ============================================================================

def butterworth_lowpass_filter(signal, cutoff_hz, fs, order=4):
    """Apply Butterworth low-pass filter for biomechanical signal processing.
    
    Superior to moving average for preserving signal shape while removing noise.
    Zero-phase filtering (filtfilt) avoids phase distortion.
    
    Args:
        signal: Input signal array
        cutoff_hz: Cutoff frequency in Hz (typically 5-10 Hz for biomechanics)
        fs: Sampling frequency in Hz
        order: Filter order (default 4 for good roll-off)
        
    Returns:
        Filtered signal with same length as input
    """
    nyquist = fs / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    
    if normalized_cutoff >= 1.0:
        print(f"⚠️  Warning: Cutoff frequency {cutoff_hz}Hz >= Nyquist frequency {nyquist}Hz. Returning original signal.")
        return signal
    
    # Design Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Apply zero-phase filtering (forward-backward to avoid phase distortion)
    filtered = filtfilt(b, a, signal)
    
    return filtered


def downsample_signal(signal, original_fs, target_fs):
    """Downsample signal with anti-aliasing filter.
    
    Args:
        signal: Input signal array
        original_fs: Original sampling frequency in Hz
        target_fs: Target sampling frequency in Hz
        
    Returns:
        Downsampled signal
    """
    if original_fs <= target_fs:
        return signal
    
    q = int(np.round(original_fs / target_fs))  # Decimation factor
    
    if q <= 1:
        return signal
    
    # Use scipy.signal.decimate which includes anti-aliasing filter
    downsampled = decimate(signal, q, ftype='iir', zero_phase=True)
    
    return downsampled


def detect_exercise_repetitions_advanced(signal, ts, phases=None, min_prominence=None, min_distance_sec=2.0, 
                                          min_rom_deg=None, remove_outliers=True, remove_first_last=True,
                                          validate_phases=True, angle_tolerance_deg=10.0):
    """Advanced exercise repetition detection with ROM threshold and outlier handling.
    
    Improved version for production biomechanical analysis:
    - ROM threshold filtering (exclude half-reps)
    - IQR outlier detection
    - Option to exclude first/last reps (adjustment phase)
    - Phase sequence validation (descending→ascending or vice versa)
    - Cyclic behavior check (return to starting position)
    - Detailed filtering statistics
    
    Args:
        signal: Input angle signal (degrees)
        ts: Timestamps (seconds)
        phases: List of phase tuples (start_idx, end_idx, phase_type) for validation
        min_prominence: Minimum prominence for peak detection (auto if None)
        min_distance_sec: Minimum time between repetitions (seconds)
        min_rom_deg: Minimum Range of Motion to consider valid rep (degrees)
        remove_outliers: Whether to remove statistical outliers using IQR method
        remove_first_last: Whether to exclude first and last reps (adjustment phase)
        validate_phases: Whether to validate phase sequence and cyclic behavior
        angle_tolerance_deg: Maximum angle difference for cyclic validation (degrees)
        
    Returns:
        Dictionary with repetition metrics (filtered for quality)
    """
    # Auto-determine prominence if not provided
    if min_prominence is None:
        signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)
        min_prominence = signal_range * 0.2
    
    # Calculate sampling rate and minimum distance in samples
    fs = len(ts) / (ts[-1] - ts[0]) if len(ts) > 1 else 1.0
    min_distance_samples = int(min_distance_sec * fs)
    
    # Find valleys (troughs) - potential repetitions
    troughs, trough_properties = find_peaks(-signal, 
                                             prominence=min_prominence,
                                             distance=min_distance_samples)
    
    if len(troughs) < 2:
        return {
            'num_repetitions': 0,
            'repetition_times': np.array([]),
            'rep_ranges': np.array([]),
            'rep_depths': np.array([]),
            'repetition_durations': np.array([]),
            'inter_rep_intervals': np.array([]),
            'prominence_threshold': min_prominence,
            'filtering_applied': {
                'min_rom_threshold': min_rom_deg,
                'outliers_removed': 0,
                'first_last_removed': remove_first_last,
                'total_detected_before_filtering': len(troughs)
            }
        }
    
    # Extract all potential repetitions
    all_rep_times = ts[troughs]
    all_rep_depths = signal[troughs]
    all_rep_ranges = []
    all_rep_durations = []
    
    for i in range(len(troughs)):
        # Define window around trough
        if i == 0:
            if len(troughs) > 1:
                window_start = 0
                window_end = int((troughs[i] + troughs[i+1]) / 2)
            else:
                window_start = max(0, troughs[i] - int(min_distance_samples/2))
                window_end = min(len(signal)-1, troughs[i] + int(min_distance_samples/2))
        elif i == len(troughs) - 1:
            window_start = int((troughs[i-1] + troughs[i]) / 2)
            window_end = len(signal) - 1
        else:
            window_start = int((troughs[i-1] + troughs[i]) / 2)
            window_end = int((troughs[i] + troughs[i+1]) / 2)
        
        rep_segment = signal[window_start:window_end]
        rep_time_segment = ts[window_start:window_end]
        
        # Range of motion and duration
        rep_range = np.max(rep_segment) - np.min(rep_segment)
        rep_duration = rep_time_segment[-1] - rep_time_segment[0] if len(rep_time_segment) > 1 else 0
        
        all_rep_ranges.append(rep_range)
        all_rep_durations.append(rep_duration)
    
    # Create mask for valid repetitions
    valid_mask = np.ones(len(all_rep_times), dtype=bool)
    filtering_stats = {
        'min_rom_threshold': min_rom_deg, 
        'outliers_removed': 0, 
        'first_last_removed': remove_first_last,
        'total_detected_before_filtering': len(all_rep_times)
    }
    
    # Filter 1: ROM threshold
    if min_rom_deg is not None:
        rom_mask = np.array(all_rep_ranges) >= min_rom_deg
        reps_below_threshold = (~rom_mask).sum()
        valid_mask &= rom_mask
        filtering_stats['reps_below_rom_threshold'] = int(reps_below_threshold)
    
    # Filter 2: Remove first and last reps (adjustment/overshoot phase)
    if remove_first_last and len(all_rep_times) > 2:
        valid_mask[0] = False
        valid_mask[-1] = False
    
    # Filter 3: Suspicious repetitions (ROM < 50% of median)
    if valid_mask.sum() > 2:
        valid_ranges = np.array(all_rep_ranges)[valid_mask]
        median_rom = np.median(valid_ranges)
        
        suspicious_mask = np.array(all_rep_ranges) >= (0.5 * median_rom)
        suspicious_removed = valid_mask.sum() - (valid_mask & suspicious_mask).sum()
        valid_mask &= suspicious_mask
        filtering_stats['suspicious_removed'] = int(suspicious_removed)
        filtering_stats['median_rom'] = float(median_rom)
        
        if suspicious_removed > 0:
            print(f"  ⚠️  Removed {suspicious_removed} suspicious repetitions (ROM < 50% median)")
    else:
        filtering_stats['suspicious_removed'] = 0
    
    # Filter 4: Phase sequence and cyclic behavior validation
    if validate_phases and phases is not None and valid_mask.sum() > 0:
        phase_validation_failed = []
        
        valid_indices = np.where(valid_mask)[0]
        for idx in valid_indices:
            i = idx  # Index in original troughs array
            trough_idx = troughs[i]
            
            # Define window for this repetition
            if i == 0:
                if len(troughs) > 1:
                    window_start = 0
                    window_end = int((troughs[i] + troughs[i+1]) / 2)
                else:
                    window_start = max(0, trough_idx - int(min_distance_samples/2))
                    window_end = min(len(signal)-1, trough_idx + int(min_distance_samples/2))
            elif i == len(troughs) - 1:
                window_start = int((troughs[i-1] + trough_idx) / 2)
                window_end = len(signal) - 1
            else:
                window_start = int((troughs[i-1] + trough_idx) / 2)
                window_end = int((trough_idx + troughs[i+1]) / 2)
            
            # Validate phase sequence and cyclic behavior
            validation = validate_repetition_phases(
                signal, ts, trough_idx, window_start, window_end, 
                phases, angle_tolerance_deg=angle_tolerance_deg
            )
            
            if not validation['is_valid']:
                valid_mask[i] = False
                phase_validation_failed.append({
                    'rep_num': i + 1,
                    'has_proper_phases': validation['has_proper_phases'],
                    'is_cyclic': validation['is_cyclic'],
                    'angle_recovery': validation['angle_recovery'],
                    'phase_sequence': validation['phase_sequence']
                })
        
        filtering_stats['phase_validation_failed'] = len(phase_validation_failed)
        
        if len(phase_validation_failed) > 0:
            print(f"  🔄 Removed {len(phase_validation_failed)} repetitions lacking proper phase sequence or cyclic behavior")
            for failed in phase_validation_failed[:2]:  # Show first 2
                reasons = []
                if not failed['has_proper_phases']:
                    reasons.append(f"phases={failed['phase_sequence']}")
                if not failed['is_cyclic']:
                    reasons.append(f"Δangle={failed['angle_recovery']:.1f}°")
                print(f"     • Rep #{failed['rep_num']}: {', '.join(reasons)}")
    else:
        filtering_stats['phase_validation_failed'] = 0
    
    # Filter 5: IQR outlier detection on ROM
    if remove_outliers and valid_mask.sum() > 4:
        valid_ranges = np.array(all_rep_ranges)[valid_mask]
        Q1 = np.percentile(valid_ranges, 25)
        Q3 = np.percentile(valid_ranges, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Apply outlier mask only to currently valid reps
        temp_outlier_mask = (np.array(all_rep_ranges) >= lower_bound) & (np.array(all_rep_ranges) <= upper_bound)
        outliers_removed = valid_mask.sum() - (valid_mask & temp_outlier_mask).sum()
        valid_mask &= temp_outlier_mask
        filtering_stats['outliers_removed'] = int(outliers_removed)
        filtering_stats['iqr_bounds'] = {'lower': float(lower_bound), 'upper': float(upper_bound)}
    
    # Apply filter
    rep_times = np.array(all_rep_times)[valid_mask]
    rep_ranges = np.array(all_rep_ranges)[valid_mask]
    rep_depths = np.array(all_rep_depths)[valid_mask]
    rep_durations = np.array(all_rep_durations)[valid_mask]
    
    # Compute inter-rep intervals
    intervals = np.diff(rep_times) if len(rep_times) > 1 else np.array([])
    
    return {
        'num_repetitions': len(rep_times),
        'repetition_times': rep_times,
        'rep_ranges': rep_ranges,
        'rep_depths': rep_depths,
        'repetition_durations': rep_durations,
        'inter_rep_intervals': intervals,
        'prominence_threshold': min_prominence,
        'filtering_applied': filtering_stats
    }


def analyze_repetition_quality_advanced(rep_data, target_rom_deg=None):
    """Advanced quality analysis with separated interpretable indices.
    
    Production-ready version for athlete feedback with:
    - ROM Consistency Index (0-100)
    - Tempo Consistency Index (0-100)
    - Depth Index (0-100, relative to target)
    - IGM Score (weighted combination)
    - Robust fatigue analysis with R² check
    
    Args:
        rep_data: Dictionary from detect_exercise_repetitions_advanced
        target_rom_deg: Target ROM for the exercise (optional, e.g., 90° for squat)
        
    Returns:
        Dictionary with structured quality metrics
    """
    if rep_data['num_repetitions'] < 2:
        return {
            'rom_consistency_index': 0,
            'tempo_consistency_index': 0,
            'depth_index': 0,
            'igm_score': 0,
            'fatigue_analysis': {
                'trend': 'insufficient_data',
                'slope': 0,
                'r_squared': 0,
                'significant': False
            },
            'qualitative_feedback': 'Insufficient data (< 2 valid reps)',
            'raw_cvs': {'range_cv': 0, 'tempo_cv': 0, 'timing_cv': 0}
        }
    
    ranges = np.array(rep_data['rep_ranges'])
    durations = np.array(rep_data['repetition_durations'])
    intervals = np.array(rep_data['inter_rep_intervals']) if len(rep_data['inter_rep_intervals']) > 0 else np.array([])
    
    # ROM Consistency Index: inverse CV scaled 0-100 (lower CV = higher consistency)
    range_cv = np.std(ranges) / np.mean(ranges) if np.mean(ranges) > 0 else 0
    rom_consistency = max(0, min(100, 100 * (1 - range_cv)))
    
    # Tempo Consistency Index: based on duration CV
    tempo_cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
    tempo_consistency = max(0, min(100, 100 * (1 - tempo_cv)))
    
    # Depth Index: how close to target ROM (if provided)
    if target_rom_deg is not None and target_rom_deg > 0:
        mean_rom = np.mean(ranges)
        depth_achievement = min(1.0, mean_rom / target_rom_deg)
        depth_index = depth_achievement * 100
    else:
        # Without target, use relative score based on consistency
        depth_index = rom_consistency  # Use ROM consistency as proxy
    
    # Robust Fatigue Analysis with R² check
    if len(ranges) >= 3:
        rep_numbers = np.arange(len(ranges))
        slope, intercept, r_value, p_value, std_err = linregress(rep_numbers, ranges)
        r_squared = r_value ** 2
        
        # Only trust trend if R² > 0.5 (good fit)
        if r_squared > 0.5:
            if slope < -0.5:
                trend = 'fatigue_detected'
            elif slope > 0.5:
                trend = 'progressive_exploration'
            else:
                trend = 'stable'
            significant = p_value < 0.05
        else:
            trend = 'inconsistent'
            significant = False
        
        fatigue_analysis = {
            'trend': trend,
            'slope': float(slope),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'significant': significant
        }
    else:
        fatigue_analysis = {
            'trend': 'insufficient_data',
            'slope': 0,
            'r_squared': 0,
            'significant': False
        }
    
    # IGM Score: weighted combination of indices
    # ROM consistency = 40%, Tempo consistency = 30%, Depth = 30%
    igm_score = (rom_consistency * 0.4 + tempo_consistency * 0.3 + depth_index * 0.3)
    
    # Qualitative feedback for athlete
    if igm_score >= 80:
        feedback = "Excellent: High consistency and depth control"
    elif igm_score >= 60:
        feedback = "Good: Consistent performance with minor variations"
    elif igm_score >= 40:
        feedback = "Fair: Noticeable variability, focus on consistency"
    else:
        feedback = "Poor: High variability, work on movement control"
    
    return {
        'rom_consistency_index': rom_consistency,
        'tempo_consistency_index': tempo_consistency,
        'depth_index': depth_index,
        'igm_score': igm_score,
        'fatigue_analysis': fatigue_analysis,
        'qualitative_feedback': feedback,
        'raw_cvs': {
            'range_cv': range_cv,
            'tempo_cv': tempo_cv,
            'timing_cv': np.std(intervals) / np.mean(intervals) if len(intervals) > 0 and np.mean(intervals) > 0 else 0
        }
    }


# ============================================================================
# END ADVANCED FUNCTIONS
# ============================================================================

# Main script
print("="*80)
print("BIOMECHANICS SIGNAL ANALYSIS - EULER ANGLES")
print("="*80)

file_path = "/Volumes/nvme/Github/bmyLab4Biomechs/sources/modules/find_time_intervals/FILE_EULER2015-1-1-13-06-49.txt"
file_name = os.path.basename(file_path)  # Extract filename for analysis

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")
print(f"\n📂 File found: {file_name}")
print(f"   Path: {file_path}")

# Convert to dataframe
print("\n" + "-"*80)
print("DATA LOADING")
print("-"*80)
df = pd.read_csv(file_path, header=None, names=["ts_ms", "ang1", "ang2", "ang3"])
df = df.drop_duplicates(subset="ts_ms").sort_values("ts_ms").reset_index(drop=True)
print(f"✓ Loaded {len(df)} unique timestamps")

# Convert timestamps to seconds
t0 = df["ts_ms"].iloc[0]
t = (df["ts_ms"] - t0).to_numpy(dtype=float) / 1000.0

# Sampling quality diagnostics
dt = np.diff(t)
print("\n" + "-"*80)
print("⏱️  SAMPLING QUALITY DIAGNOSTICS")
print("-"*80)
print(f"  📊 Δt statistics:")
print(f"     • Mean:     {dt.mean()*1000:7.3f} ms")
print(f"     • Std Dev:  {dt.std()*1000:7.3f} ms")
print(f"     • Min:      {dt.min()*1000:7.3f} ms")
print(f"     • Max:      {dt.max()*1000:7.3f} ms")

# Detect sampling gaps
gap_thr = 3 * np.median(dt)
num_gaps = np.sum(dt > gap_thr)
if num_gaps > 0:
    print(f"\n  ⚠️  Sampling gaps (dt > {gap_thr*1000:.1f} ms): {num_gaps}")
    gap_indices = np.where(dt > gap_thr)[0]
    print(f"     • Gap locations (first 5): {gap_indices[:5].tolist()}")
    print(f"     • Max gap duration: {dt[gap_indices].max()*1000:.1f} ms")
else:
    print(f"\n  ✅ No significant sampling gaps detected")

# Resample all the signals to uniform time intervals
print("\n" + "-"*80)
print("🔄 SIGNAL RESAMPLING")
print("-"*80)
uniform_t = np.linspace(t[0], t[-1], len(t))
print(f"✅ Resampling to uniform time intervals: {len(uniform_t)} samples")
print(f"  ⏰ Time range: {uniform_t[0]:.2f} → {uniform_t[-1]:.2f} seconds")
print(f"  ⌛ Duration: {uniform_t[-1] - uniform_t[0]:.2f} seconds\n")

# Resample all three angle axes
resampled_data = {"ts_s": uniform_t}
for angle_col in ["ang1", "ang2", "ang3"]:
    theta = df[angle_col].to_numpy()
    print(f"  {angle_col.upper()}: min={theta.min():7.2f}°, max={theta.max():7.2f}°, std={theta.std():5.2f}°")
    
    interp_func = interp1d(t, theta, kind='linear', fill_value="extrapolate")
    resampled_data[angle_col] = interp_func(uniform_t)

# Apply moving average filter to denoise signals
print("\n" + "-"*80)
print("🔧 NOISE FILTERING (Moving Average)")
print("-"*80)

# Determine window size based on sampling rate
if len(uniform_t) > 1:
    fs_approx = len(uniform_t) / (uniform_t[-1] - uniform_t[0])
    signal_duration = uniform_t[-1] - uniform_t[0]
else:
    fs_approx = 100  # fallback
    signal_duration = 1.0

# Window size: 1/10 of signal duration (minimum requirement)
# This ensures we don't filter out the fundamental movement frequency
window_duration = signal_duration / 100.0  # seconds
window_size = max(5, int(window_duration * fs_approx))  # Convert to samples
if window_size % 2 == 0:  # Make it odd for symmetry
    window_size += 1

print(f"  📏 Signal duration: {signal_duration:.2f} s")
print(f"  📏 Filter window duration: {window_duration:.3f} s (1/10 of signal)")
print(f"  📐 Filter window size: {window_size} samples (~{window_size/fs_approx*1000:.1f} ms)")
print(f"  🎯 Purpose: Remove high-frequency noise while preserving biomechanical frequencies\n")

# Apply filter to all axes and add filtered versions
for angle_col in ["ang1", "ang2", "ang3"]:
    original = resampled_data[angle_col]
    filtered = moving_average_filter(original, window_size)
    resampled_data[f"{angle_col}_filtered"] = filtered
    
    # Compute noise reduction metrics
    noise_original = np.std(np.diff(original))
    noise_filtered = np.std(np.diff(filtered))
    noise_reduction = (1 - noise_filtered/noise_original) * 100
    
    print(f"  {angle_col.upper()}:")
    print(f"     • Original noise (diff std): {noise_original:.4f}°")
    print(f"     • Filtered noise (diff std): {noise_filtered:.4f}°")
    print(f"     • Noise reduction: {noise_reduction:5.1f}%")

print(f"\n✅ Filtered signals added to dataset")

# PCA Analysis on 3D Euler angles
print("\n" + "-"*80)
print("🧬 PRINCIPAL COMPONENT ANALYSIS (3D Euler Angles)")
print("-"*80)
X = df[["ang1", "ang2", "ang3"]].to_numpy()
X_centered = X - X.mean(axis=0, keepdims=True)
pca = PCA(n_components=3)
pca.fit(X_centered)
explained = pca.explained_variance_ratio_

print(f"  📊 Explained variance:")
print(f"     • PC1: {explained[0]*100:5.1f}% {'🎯' if explained[0] > 0.7 else '📈'}")
print(f"     • PC2: {explained[1]*100:5.1f}%")
print(f"     • PC3: {explained[2]*100:5.1f}%")
print(f"\n  🔢 PC1 loadings (ang1, ang2, ang3): [{pca.components_[0][0]:+.3f}, {pca.components_[0][1]:+.3f}, {pca.components_[0][2]:+.3f}]")
print(f"  🔢 PC2 loadings (ang1, ang2, ang3): [{pca.components_[1][0]:+.3f}, {pca.components_[1][1]:+.3f}, {pca.components_[1][2]:+.3f}]")

# Add principal component as resampled signal (original)
pc1_raw = X_centered @ pca.components_[0]
interp_pc1 = interp1d(t, pc1_raw, kind='linear', fill_value="extrapolate")
resampled_data["ang_principal"] = interp_pc1(uniform_t)

# Add filtered principal component
resampled_data["ang_principal_filtered"] = moving_average_filter(resampled_data["ang_principal"], window_size)

print(f"\n✅ Principal component (PC1) added - both original and filtered versions")

# Save the resampled data to a new CSV file
resampled_df = pd.DataFrame(resampled_data)
resampled_path = file_path.replace(".txt", "_resampled.csv")
resampled_df.to_csv(resampled_path, index=False)
print(f"\n✓ Resampled data saved: {os.path.basename(resampled_path)}")

# Save the plot of the resampled signals (original)
plt.figure(figsize=(14, 7))
plt.plot(resampled_df["ts_s"], resampled_df["ang1"], label="Angle 1", linewidth=1.0, alpha=0.8)
plt.plot(resampled_df["ts_s"], resampled_df["ang2"], label="Angle 2", linewidth=1.0, alpha=0.8)
plt.plot(resampled_df["ts_s"], resampled_df["ang3"], label="Angle 3", linewidth=1.0, alpha=0.8)
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Angle [degrees]", fontsize=12)
plt.title("📊 Original Resampled Euler Angles - Time Series", fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
path_of_this_script = os.path.abspath(__file__)
plt.savefig(os.path.join(os.path.dirname(path_of_this_script), "resampled_signals_original.png"), dpi=150, bbox_inches='tight')
print(f"✓ Original time series plot saved: resampled_signals_original.png")
plt.close()

# Save the plot of the filtered signals
plt.figure(figsize=(14, 7))
plt.plot(resampled_df["ts_s"], resampled_df["ang1_filtered"], label="Angle 1 (Filtered)", linewidth=1.5, alpha=0.9)
plt.plot(resampled_df["ts_s"], resampled_df["ang2_filtered"], label="Angle 2 (Filtered)", linewidth=1.5, alpha=0.9)
plt.plot(resampled_df["ts_s"], resampled_df["ang3_filtered"], label="Angle 3 (Filtered)", linewidth=1.5, alpha=0.9)
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Angle [degrees]", fontsize=12)
plt.title("✨ Filtered Euler Angles - Time Series (Noise Reduced)", fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(path_of_this_script), "resampled_signals_filtered.png"), dpi=150, bbox_inches='tight')
print(f"✓ Filtered time series plot saved: resampled_signals_filtered.png")
plt.close()

# Save comparison plot (original vs filtered for one axis)
plt.figure(figsize=(14, 7))
plt.plot(resampled_df["ts_s"], resampled_df["ang1"], label="ANG1 Original", linewidth=0.8, alpha=0.5, color='blue')
plt.plot(resampled_df["ts_s"], resampled_df["ang1_filtered"], label="ANG1 Filtered", linewidth=1.5, alpha=0.9, color='red')
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Angle [degrees]", fontsize=12)
plt.title("🔍 Original vs Filtered Signal Comparison - ANG1", fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(path_of_this_script), "signal_comparison.png"), dpi=150, bbox_inches='tight')
print(f"✓ Comparison plot saved: signal_comparison.png")
plt.close()

# Create comprehensive overview plot with phases and repetitions for each axis
print(f"\n📊 Generating comprehensive time-series overview plots with phase regions...")

for axis_col in ["ang1", "ang2", "ang3", "ang_principal"]:
    # Will be populated after phase/rep detection, placeholder for now
    # This will be generated later in the script after phase_stats and repetition_stats are computed
    pass

# Initialize containers for advanced analysis
cycle_stats = {}
phase_stats = {}
repetition_stats = {}  # Store detailed repetition analysis
cross_corrs = {}

print("\n" + "="*80)
print("SIGNAL CHARACTERIZATION & STATISTICAL ANALYSIS")
print("="*80)
print("  📋 Analyzing both ORIGINAL and FILTERED signals")
print("  🔬 Original: Raw resampled data with noise")
print("  ✨ Filtered: Noise-reduced via moving average filter")
print("="*80)

# Elaborating the stats for every axis signal (both original and filtered)
for angle_col in ["ang1", "ang2", "ang3", "ang_principal", 
                   "ang1_filtered", "ang2_filtered", "ang3_filtered", "ang_principal_filtered"]:
    theta = resampled_df[angle_col].to_numpy()
    theta_detrended = detrend(theta)
    ts = resampled_df["ts_s"].to_numpy()

    # Signal quality diagnostics
    dynamic_range = np.percentile(theta, 95) - np.percentile(theta, 5)
    noise_est = np.std(np.diff(theta))  # rough noise estimate from diff
    snr_like = dynamic_range / (noise_est + 1e-9)
    
    # Clipping detection
    clip_low = (theta <= theta.min() + 1e-3).sum()
    clip_high = (theta >= theta.max() - 1e-3).sum()


    # Basic statistics
    min_val = theta.min()
    max_val = theta.max()
    range_val = max_val - min_val
    mean_val = theta.mean()
    median_val = np.median(theta)
    std_val = theta.std()
    skew_val = pd.Series(theta).skew()
    kurt_val = pd.Series(theta).kurtosis()
    min_idx = np.argmin(theta)
    max_idx = np.argmax(theta)

    # Autocorrelation
    autocorr = acf(theta, nlags=100, fft=True)
    peak_lag = np.argmax(autocorr[1:]) + 1
    periodicity = peak_lag if peak_lag > 0 else None

    # Frequency analysis (FFT) with multiple peak detection
    if len(ts) > 1:
        fs = len(ts) / (ts[-1] - ts[0])
    else:
        fs = 1.0
    freqs, power = periodogram(theta_detrended, fs=fs)
    
    # Find multiple frequency peaks for better analysis
    freq_peaks_idx, _ = find_peaks(power[1:], prominence=np.max(power)*0.1)  # 10% of max power
    freq_peaks_idx = freq_peaks_idx + 1  # Adjust for skipping DC component
    
    # Get top 5 frequency peaks sorted by power
    if len(freq_peaks_idx) > 0:
        freq_peak_powers = power[freq_peaks_idx]
        sorted_indices = np.argsort(freq_peak_powers)[::-1]
        top_freq_indices = freq_peaks_idx[sorted_indices[:min(5, len(freq_peaks_idx))]]
        top_freqs = freqs[top_freq_indices]
        top_powers = power[top_freq_indices]
    else:
        top_freqs = np.array([])
        top_powers = np.array([])
    
    dominant_freq = freqs[np.argmax(power[1:])+1] if len(power) > 1 else 0
    period = 1/dominant_freq if dominant_freq > 0 else None

    # Stationarity test
    adf_result = adfuller(theta)
    stationary = adf_result[1] < 0.05  # p-value < 0.05

    # Trend (linear regression)
    slope, intercept, r_value, p_value, std_err = linregress(ts, theta)

    # Peak detection
    peaks, _ = find_peaks(theta)
    troughs, _ = find_peaks(-theta)
    num_peaks = len(peaks)
    num_troughs = len(troughs)

    # Cycle segmentation
    cycles, cycle_peaks, cycle_troughs = detect_cycles(theta, prominence=1.0, distance=int(len(theta)/40))
    
    # Cycle-level metrics
    cycle_metrics = compute_cycle_metrics(theta, ts, cycles)
    
    cycle_stats[angle_col] = {
        "num_cycles": len(cycles),
        "cycle_indices": [int(c) for c in cycles],
        "num_peaks": len(cycle_peaks),
        "num_troughs": len(cycle_troughs),
        "cycle_durations": cycle_metrics['durations'].tolist(),
        "cycle_amplitudes": cycle_metrics['amplitudes'].tolist(),
        "mean_cycle_duration": float(cycle_metrics['mean_duration']),
        "cv_cycle_duration": float(cycle_metrics['cv_duration']),
        "mean_cycle_amplitude": float(cycle_metrics['mean_amplitude']),
        "cv_cycle_amplitude": float(cycle_metrics['cv_amplitude']),
        "trend_duration_per_cycle": float(cycle_metrics['trend_duration']),
        "trend_amplitude_per_cycle": float(cycle_metrics['trend_amplitude'])
    }

    # Phase detection
    phases = detect_phases(theta, cycle_peaks, cycle_troughs)
    phase_statistics = compute_phase_statistics(phases, ts)
    phase_stats[angle_col] = {
        'segments': [{"start": int(s), "end": int(e), "type": p} for s, e, p in phases],
        'statistics': phase_statistics
    }
    
    # === EXERCISE REPETITION ANALYSIS (compute before saving) ===
    rep_data = detect_exercise_repetitions(theta, ts, min_prominence=None, min_distance_sec=2.0)
    
    # === VALIDATE REPETITIONS WITH PHASE ANALYSIS ===
    if rep_data['num_repetitions'] > 0 and 'trough_indices' in rep_data:
        fs = len(ts) / (ts[-1] - ts[0]) if len(ts) > 1 else 1.0
        min_distance_samples = int(2.0 * fs)
        
        validation_results = []
        invalid_reps = []
        
        for i, trough_idx in enumerate(rep_data['trough_indices']):
            # Define window for this repetition
            if i == 0:
                if rep_data['num_repetitions'] > 1:
                    window_start = 0
                    window_end = int((rep_data['trough_indices'][i] + rep_data['trough_indices'][i+1]) / 2)
                else:
                    window_start = max(0, trough_idx - int(min_distance_samples/2))
                    window_end = min(len(theta)-1, trough_idx + int(min_distance_samples/2))
            elif i == rep_data['num_repetitions'] - 1:
                window_start = int((rep_data['trough_indices'][i-1] + trough_idx) / 2)
                window_end = len(theta) - 1
            else:
                window_start = int((rep_data['trough_indices'][i-1] + trough_idx) / 2)
                window_end = int((trough_idx + rep_data['trough_indices'][i+1]) / 2)
            
            # Validate this repetition
            validation = validate_repetition_phases(
                theta, ts, trough_idx, window_start, window_end, 
                phases, angle_tolerance_deg=20.0  # Allow 20° tolerance for angle recovery (realistic for biomechanical movements)
            )
            
            validation_results.append(validation)
            
            if not validation['is_valid']:
                invalid_reps.append({
                    'rep_num': i + 1,
                    'reason': [],
                    'phase_sequence': validation['phase_sequence'],
                    'angle_recovery': validation['angle_recovery']
                })
                
                if not validation['has_proper_phases']:
                    invalid_reps[-1]['reason'].append(f"Missing phase sequence (has_asc={validation['has_ascending']}, has_desc={validation['has_descending']})")
                if not validation['is_cyclic']:
                    invalid_reps[-1]['reason'].append(f"Not cyclic (Δangle={validation['angle_recovery']:.1f}°)")
        
        # Add validation info to rep_data
        rep_data['validation_results'] = validation_results
        rep_data['num_invalid'] = len(invalid_reps)
        rep_data['invalid_reps'] = invalid_reps
        
        # Update suspicious flags to include phase validation
        original_suspicious = rep_data['suspicious_reps'].copy()
        for i, validation in enumerate(validation_results):
            if not validation['is_valid']:
                rep_data['suspicious_reps'][i] = True
        
        rep_data['num_suspicious'] = int(np.sum(rep_data['suspicious_reps']))
        
        # Log validation issues
        if len(invalid_reps) > 0:
            print(f"     ⚠️  Phase validation: {len(invalid_reps)} repetitions lack proper phase sequence or cyclic behavior")
            for inv in invalid_reps[:3]:  # Show first 3
                reasons = ', '.join(inv['reason'])
                print(f"        • Rep #{inv['rep_num']}: {reasons}")
    
    rep_quality = analyze_repetition_quality(rep_data)
    
    # === COMPUTE DETAILED TEMPORAL PARAMETERS ===
    temporal_params = compute_detailed_temporal_parameters(rep_data, phases, ts, theta)
    
    # Save repetition analysis data (including phase validation results and detailed temporal parameters)
    repetition_stats[angle_col] = {
        'num_repetitions': int(rep_data['num_repetitions']),
        'trough_indices': rep_data['trough_indices'].tolist() if len(rep_data['trough_indices']) > 0 else [],
        'repetition_times': rep_data['repetition_times'].tolist() if len(rep_data['repetition_times']) > 0 else [],
        'inter_rep_intervals': rep_data['inter_rep_intervals'].tolist() if len(rep_data['inter_rep_intervals']) > 0 else [],
        'rep_durations': rep_data['repetition_durations'].tolist() if len(rep_data['repetition_durations']) > 0 else [],
        'rep_ranges': rep_data['rep_ranges'].tolist() if len(rep_data['rep_ranges']) > 0 else [],
        'rep_depths': rep_data['rep_depths'].tolist() if len(rep_data['rep_depths']) > 0 else [],
        'prominence_threshold': float(rep_data['prominence_threshold']),
        'quality': {
            'consistency_score': float(rep_quality['consistency_score']),
            'range_cv': float(rep_quality['range_cv']),
            'timing_cv': float(rep_quality['timing_cv']),
            'tempo_cv': float(rep_quality['tempo_cv']),
            'fatigue_indicator': float(rep_quality['fatigue_indicator']),
        },
        # Phase validation data
        'num_suspicious': int(rep_data.get('num_suspicious', 0)),
        'suspicious_reps': rep_data.get('suspicious_reps', np.zeros(rep_data['num_repetitions'], dtype=bool)).tolist(),
        'median_rom': float(rep_data.get('median_rom', 0)),
        'num_invalid': int(rep_data.get('num_invalid', 0)),
        'invalid_reps': rep_data.get('invalid_reps', []),
        'validation_results': rep_data.get('validation_results', []),
        'has_validation': 'validation_results' in rep_data,
        # Detailed temporal parameters (50 parameters across 7 categories)
        'temporal_parameters': temporal_params
    }

    # Cross-axis correlation
    if angle_col == "ang1":
        cross_corrs["ang1_ang2"] = cross_axis_correlation(resampled_df, "ang1", "ang2")
        cross_corrs["ang1_ang3"] = cross_axis_correlation(resampled_df, "ang1", "ang3")
    if angle_col == "ang2":
        cross_corrs["ang2_ang3"] = cross_axis_correlation(resampled_df, "ang2", "ang3")

    # Sliding window statistics
    window_sec = 2.0
    step_sec = 0.5
    if len(ts) > 1:
        fs = len(ts) / (ts[-1] - ts[0])
    else:
        fs = 1.0
    window = int(window_sec * fs)
    step = int(step_sec * fs)
    
    win_means = []
    win_stds = []
    win_t = []
    
    for start in range(0, len(theta) - window, step):
        end = start + window
        seg = theta[start:end]
        win_means.append(seg.mean())
        win_stds.append(seg.std())
        win_t.append(ts[start + window//2])
    
    win_means = np.array(win_means)
    win_stds = np.array(win_stds)

    # Print statistics with improved formatting
    print(f"\n{'─'*80}")
    print(f"📊 AXIS: {angle_col.upper()}")
    print(f"{'─'*80}")
    
    print(f"\n  📈 Descriptive Statistics:")
    print(f"     • Min:        {min_val:8.2f}° (sample {min_idx:6d})")
    print(f"     • Max:        {max_val:8.2f}° (sample {max_idx:6d})")
    print(f"     • Range:      {range_val:8.2f}°")
    print(f"     • Mean:       {mean_val:8.2f}°")
    print(f"     • Median:     {median_val:8.2f}°")
    print(f"     • Std Dev:    {std_val:8.2f}°")
    print(f"     • Skewness:   {skew_val:8.2f}")
    print(f"     • Kurtosis:   {kurt_val:8.2f}")
    
    print(f"\n  🔬 Signal Quality:")
    print(f"     • Dynamic range (5-95%): {dynamic_range:8.2f}°")
    print(f"     • Noise estimate (diff): {noise_est:8.4f}°")
    print(f"     • SNR-like index:        {snr_like:8.2f}")
    print(f"     • Clipping (low/high):   {clip_low:4d} / {clip_high:4d} samples")
    
    print(f"\n  🔍 Signal Properties:")
    stationary_str = "✓ Stationary" if stationary else "✗ Non-stationary"
    print(f"     • {stationary_str} (ADF p-value < 0.05)")
    print(f"     • Linear trend: {slope:+.4f} deg/s")
    print(f"     • Peaks detected:   {num_peaks:4d}")
    print(f"     • Troughs detected: {num_troughs:4d}")
    
    print(f"\n  🌊 Frequency & Periodicity:")
    print(f"     • Dominant frequency: {dominant_freq:.4f} Hz (period: {1/dominant_freq:.2f}s)" if dominant_freq > 0 else "     • Dominant frequency: N/A")
    
    if len(top_freqs) > 0:
        print(f"     • Top frequency peaks (Hz):")
        for i, (freq, pwr) in enumerate(zip(top_freqs, top_powers)):
            period_str = f"period={1/freq:.2f}s" if freq > 0 else "period=N/A"
            print(f"        {i+1}. {freq:.4f} Hz ({period_str}, power={pwr:.2e})")
    
    if period:
        print(f"     • Estimated period:   {period:.2f} s")
    else:
        print(f"     • Estimated period:   N/A")
    print(f"     • Autocorr periodicity: {periodicity} samples (lag of 1st peak)")
    
    print(f"\n  🔄 Cycle Analysis:")
    print(f"     • Total cycles:      {len(cycles):4d}")
    print(f"     • Cycle peaks:       {len(cycle_peaks):4d}")
    print(f"     • Cycle troughs:     {len(cycle_troughs):4d}")
    if len(cycle_metrics['durations']) > 0:
        print(f"\n  🧬 Cycle-Level Metrics:")
        print(f"     • Mean cycle duration:   {cycle_metrics['mean_duration']:.3f} s (CV={cycle_metrics['cv_duration']:.2%})")
        print(f"     • Mean cycle amplitude:  {cycle_metrics['mean_amplitude']:.2f}° (CV={cycle_metrics['cv_amplitude']:.2%})")
        print(f"     • Trend duration/cycle:  {cycle_metrics['trend_duration']:+.4f} s/cycle")
        print(f"     • Trend amplitude/cycle: {cycle_metrics['trend_amplitude']:+.4f} °/cycle")
    
    print(f"\n  📍 Phase Statistics:")
    phase_order = ['ascending', 'descending', 'stable_high', 'stable_low', 'stable']
    phase_labels = {
        'ascending': 'ascending  ',
        'descending': 'descending ',
        'stable_high': 'stable_high',
        'stable_low': 'stable_low ',
        'stable': 'stable     '
    }
    
    for phase_type in phase_order:
        pstats = phase_statistics.get(phase_type, {'count': 0})
        if pstats['count'] > 0:
            label = phase_labels.get(phase_type, phase_type)
            print(f"     • {label}: n={pstats['count']:3d}, mean={pstats['mean_duration']:.3f}s, "
                  f"CV={pstats['cv_duration']:.2%}, total={pstats['total_time']:.1f}s")
    
    print(f"\n  📉 Sliding-Window Stats ({window_sec}s window, {step_sec}s step):")
    if len(win_means) > 0:
        print(f"     • Std of window means: {win_means.std():.2f}°")
        print(f"     • Std of window stds:  {win_stds.std():.2f}°")
        print(f"     • Mean window std:     {win_stds.mean():.2f}°")
    
    # === EXERCISE REPETITION ANALYSIS (already computed above, just print) ===
    print(f"\n  🏋️ EXERCISE REPETITION ANALYSIS:")
    print(f"     ╔{'═'*75}╗")
    print(f"     ║ {'REPETITION COUNT & IDENTIFICATION':^73} ║")
    print(f"     ╠{'═'*75}╣")
    print(f"     ║ Total Repetitions Detected: {rep_data['num_repetitions']:>3d} {' '*38} ║")
    print(f"     ║ Detection Threshold (prominence): {rep_data['prominence_threshold']:>6.2f}° {' '*31} ║")
    
    num_suspicious = rep_data.get('num_suspicious', 0)
    if num_suspicious > 0:
        suspicious_pct = (num_suspicious / rep_data['num_repetitions'] * 100) if rep_data['num_repetitions'] > 0 else 0
        print(f"     ║ ⚠️  Suspicious Reps (ROM<50% median): {num_suspicious:>3d} ({suspicious_pct:>4.1f}%) {' '*26} ║")
    
    print(f"     ╚{'═'*75}╝")
    
    if rep_data['num_repetitions'] > 0:
        # Repetition timing details
        print(f"\n     📊 PER-REPETITION METRICS:")
        print(f"     ┌{'─'*75}┐")
        print(f"     │ {'Rep':<4} │ {'Time (s)':<10} │ {'Duration (s)':<12} │ {'Range (°)':<11} │ {'Depth (°)':<11} │")
        print(f"     ├{'─'*75}┤")
        
        suspicious_flags = rep_data.get('suspicious_reps', np.zeros(rep_data['num_repetitions'], dtype=bool))
        
        for i in range(rep_data['num_repetitions']):
            rep_num = i + 1
            rep_time = rep_data['repetition_times'][i]
            rep_dur = rep_data['repetition_durations'][i]
            rep_range = rep_data['rep_ranges'][i]
            rep_depth = rep_data['rep_depths'][i]
            is_suspicious = suspicious_flags[i]
            
            marker = "⚠️" if is_suspicious else " "
            print(f"     │{marker}{rep_num:<3} │ {rep_time:>10.2f} │ {rep_dur:>12.2f} │ {rep_range:>11.2f} │ {rep_depth:>11.2f} │")
        
        print(f"     └{'─'*75}┘")
        
        # Inter-repetition intervals
        if len(rep_data['inter_rep_intervals']) > 0:
            print(f"\n     ⏱️  INTER-REPETITION INTERVALS:")
            intervals = rep_data['inter_rep_intervals']
            print(f"        • Mean interval:    {np.mean(intervals):>6.2f} s")
            print(f"        • Std deviation:    {np.std(intervals):>6.2f} s")
            print(f"        • Min interval:     {np.min(intervals):>6.2f} s")
            print(f"        • Max interval:     {np.max(intervals):>6.2f} s")
            print(f"        • Coefficient of variation: {np.std(intervals)/np.mean(intervals)*100:>5.1f}%")
        
        # Range of motion statistics
        ranges = rep_data['rep_ranges']
        print(f"\n     📏 RANGE OF MOTION ANALYSIS:")
        print(f"        • Mean range:       {np.mean(ranges):>6.2f}°")
        print(f"        • Std deviation:    {np.std(ranges):>6.2f}°")
        print(f"        • Min range:        {np.min(ranges):>6.2f}°")
        print(f"        • Max range:        {np.max(ranges):>6.2f}°")
        print(f"        • Consistency (CV): {rep_quality['range_cv']*100:>5.1f}%")
        
        # Duration analysis
        durations = rep_data['repetition_durations']
        print(f"\n     ⏲️  REPETITION DURATION ANALYSIS:")
        print(f"        • Mean duration:    {np.mean(durations):>6.2f} s")
        print(f"        • Std deviation:    {np.std(durations):>6.2f} s")
        print(f"        • Tempo CV:         {rep_quality['tempo_cv']*100:>5.1f}%")
        
        # Quality metrics
        print(f"\n     ⭐ PERFORMANCE QUALITY METRICS:")
        print(f"        • Overall Consistency Score: {rep_quality['consistency_score']:>5.1f}/100")
        print(f"        • Range Consistency (lower=better):  {rep_quality['range_cv']*100:>5.1f}%")
        print(f"        • Timing Consistency (lower=better): {rep_quality['timing_cv']*100:>5.1f}%")
        print(f"        • Tempo Consistency (lower=better):  {rep_quality['tempo_cv']*100:>5.1f}%")
        
        # Fatigue indicator
        if abs(rep_quality['fatigue_indicator']) > 0.01:
            fatigue_direction = "INCREASING" if rep_quality['fatigue_indicator'] > 0 else "DECREASING"
            print(f"        • Fatigue Indicator: {fatigue_direction} ({rep_quality['fatigue_indicator']:+.3f}°/rep)")
            if rep_quality['fatigue_indicator'] < -0.5:
                print(f"          ⚠️  WARNING: Significant range decrease detected (possible fatigue)")
        else:
            print(f"        • Fatigue Indicator: STABLE ({rep_quality['fatigue_indicator']:+.3f}°/rep)")
        
        # Frequency from repetitions
        if rep_data['num_repetitions'] > 1:
            total_time = rep_data['repetition_times'][-1] - rep_data['repetition_times'][0]
            avg_rep_frequency = (rep_data['num_repetitions'] - 1) / total_time
            print(f"\n     🎯 DERIVED METRICS:")
            print(f"        • Average repetition frequency: {avg_rep_frequency:.4f} Hz ({1/avg_rep_frequency:.2f} s/rep)")
            print(f"        • Total exercise duration: {total_time:.2f} s")
            print(f"        • Work density: {rep_data['num_repetitions']/total_time*60:.1f} reps/minute")
        
        # === DETAILED TEMPORAL PARAMETERS ===
        if 'temporal_parameters' in repetition_stats.get(angle_col, {}):
            tp = repetition_stats[angle_col]['temporal_parameters']
            
            print(f"\n     ⏱️  DETAILED TEMPORAL PARAMETERS:")
            print(f"     ┌{'─'*75}┐")
            
            # B. Global Timing Parameters
            gt = tp['global_timing']
            print(f"     │ {'GLOBAL TIMING':<73} │")
            print(f"     ├{'─'*75}┤")
            print(f"     │   Mean Rep Duration:        {gt['mean_rep_duration']:>6.2f} s {' '*35} │")
            print(f"     │   Mean Cycle Time:          {gt['mean_cycle_time']:>6.2f} s {' '*35} │")
            print(f"     │   Execution Frequency:      {gt['execution_frequency_hz']:>6.3f} Hz {' '*34} │")
            print(f"     │   Mean Density Ratio:       {gt['mean_density_ratio']:>6.2%} {' '*36} │")
            
            # C. Session Total Times
            st = tp['session_totals']
            print(f"     ├{'─'*75}┤")
            print(f"     │ {'SESSION TOTALS':<73} │")
            print(f"     ├{'─'*75}┤")
            print(f"     │   Total Eccentric Time:     {st['total_eccentric_time']:>6.2f} s {' '*35} │")
            print(f"     │   Total Concentric Time:    {st['total_concentric_time']:>6.2f} s {' '*35} │")
            print(f"     │   Total Bottom Hold Time:   {st['total_bottom_hold_time']:>6.2f} s {' '*35} │")
            print(f"     │   Total Top Hold Time:      {st['total_top_hold_time']:>6.2f} s {' '*35} │")
            print(f"     │   Total Work Time:          {st['total_work_time']:>6.2f} s {' '*35} │")
            print(f"     │   Total Pause Time:         {st['total_pause_time']:>6.2f} s {' '*35} │")
            
            # D. Temporal Variances (show summary)
            tv = tp['temporal_variances']
            print(f"     ├{'─'*75}┤")
            print(f"     │ {'TEMPORAL VARIANCES (CV)':<73} │")
            print(f"     ├{'─'*75}┤")
            print(f"     │   Eccentric CV:             {tv['eccentric']['cv']:>6.2%} {' '*36} │")
            print(f"     │   Concentric CV:            {tv['concentric']['cv']:>6.2%} {' '*36} │")
            print(f"     │   Bottom Hold CV:           {tv['bottom_hold']['cv']:>6.2%} {' '*36} │")
            print(f"     │   Top Hold CV:              {tv['top_hold']['cv']:>6.2%} {' '*36} │")
            
            # E. Regularity
            reg = tp['regularity']
            print(f"     ├{'─'*75}┤")
            print(f"     │ {'REGULARITY & CONTINUITY':<73} │")
            print(f"     ├{'─'*75}┤")
            print(f"     │   Rep Duration CV:          {reg['rep_duration_cv']:>6.2%} {' '*36} │")
            print(f"     │   Cycle Time CV:            {reg['cycle_time_cv']:>6.2%} {' '*36} │")
            
            # F. Outliers
            out = tp['outliers']
            print(f"     ├{'─'*75}┤")
            print(f"     │ {'TEMPORAL OUTLIERS':<73} │")
            print(f"     ├{'─'*75}┤")
            print(f"     │   Slow Eccentric Reps:      {out['num_slow_eccentric']:>3d} {' '*42} │")
            print(f"     │   Fast Concentric Reps:     {out['num_fast_concentric']:>3d} {' '*42} │")
            print(f"     │   Excessive Bottom Holds:   {out['num_excessive_bottom_hold']:>3d} {' '*42} │")
            print(f"     │   Excessive Top Holds:      {out['num_excessive_top_hold']:>3d} {' '*42} │")
            
            # G. Longitudinal Trends
            lt = tp['longitudinal_trends']
            print(f"     ├{'─'*75}┤")
            print(f"     │ {'LONGITUDINAL TRENDS (s/rep)':<73} │")
            print(f"     ├{'─'*75}┤")
            print(f"     │   Eccentric Trend:          {lt['eccentric_trend']:>+7.4f} s/rep {' '*30} │")
            print(f"     │   Concentric Trend:         {lt['concentric_trend']:>+7.4f} s/rep {' '*30} │")
            print(f"     │   Bottom Hold Trend:        {lt['bottom_hold_trend']:>+7.4f} s/rep {' '*30} │")
            print(f"     │   Top Hold Trend:           {lt['top_hold_trend']:>+7.4f} s/rep {' '*30} │")
            print(f"     │   Rep Duration Trend:       {lt['rep_duration_trend']:>+7.4f} s/rep {' '*30} │")
            print(f"     │   Cycle Time Trend:         {lt['cycle_time_trend']:>+7.4f} s/rep {' '*30} │")
            
            print(f"     └{'─'*75}┘")
    
    # === ADVANCED REPETITION ANALYSIS (Production-ready with Butterworth filtering) ===
    # Apply to all signals (both original and moving-average filtered)
    # For original signals: applies ADDITIONAL Butterworth filter
    # For filtered signals: applies Butterworth on the already filtered signal
    print(f"\n\n  🚀 ADVANCED REPETITION ANALYSIS (Production-Ready):")
    print(f"     ╔{'═'*75}╗")
    print(f"     ║ {'BUTTERWORTH FILTER + ROM THRESHOLD & OUTLIER REMOVAL':^73} ║")
    print(f"     ╚{'═'*75}╝")
    
    if "_filtered" in angle_col:
        print(f"     ℹ️  Note: Applying Butterworth filter on moving-average filtered signal")
    else:
        print(f"     ℹ️  Note: Applying Butterworth filter on original signal")
    
    # Apply Butterworth low-pass filter
    try:
        if len(ts) > 1:
            fs = len(ts) / (ts[-1] - ts[0])
        else:
            fs = 1.0
        
        cutoff_hz = min(10.0, fs / 4)  # Ensure cutoff < Nyquist
        theta_butterworth = butterworth_lowpass_filter(theta, cutoff_hz=cutoff_hz, fs=fs, order=4)
        
        print(f"     ✓ Applied Butterworth low-pass filter (cutoff={cutoff_hz:.1f}Hz, order=4)")
        
        # Determine ROM threshold automatically (30° or 50% of max observed ROM)
        signal_rom = np.percentile(theta_butterworth, 95) - np.percentile(theta_butterworth, 5)
        min_rom_threshold = min(30.0, signal_rom * 0.5)
        
        # Advanced detection with all filters enabled
        rep_data_adv = detect_exercise_repetitions_advanced(
            theta_butterworth, ts,
            phases=phases,  # Pass phase information for validation
            min_prominence=None,
            min_distance_sec=2.0,
            min_rom_deg=min_rom_threshold,
            remove_outliers=True,
            remove_first_last=True,
            validate_phases=True,  # Enable phase sequence validation
            angle_tolerance_deg=10.0  # Allow 10° for cyclic behavior
        )
        
        # Quality analysis with target ROM (if known - use 90° for squats as example)
        target_rom = 90.0 if "squat" in file_name.lower() else None
        rep_quality_adv = analyze_repetition_quality_advanced(rep_data_adv, target_rom_deg=target_rom)
        
        # Print filtering statistics
        filter_stats = rep_data_adv['filtering_applied']
        print(f"\n     📋 FILTERING STATISTICS:")
        print(f"        • Total reps detected (before filtering): {filter_stats['total_detected_before_filtering']}")
        print(f"        • ROM threshold applied: {filter_stats['min_rom_threshold']:.1f}°")
        if 'reps_below_rom_threshold' in filter_stats:
            print(f"        • Reps below ROM threshold: {filter_stats['reps_below_rom_threshold']}")
        if 'suspicious_removed' in filter_stats:
            print(f"        • Suspicious reps removed (ROM<50%): {filter_stats['suspicious_removed']}")
        if 'phase_validation_failed' in filter_stats and filter_stats['phase_validation_failed'] > 0:
            print(f"        • Phase validation failed: {filter_stats['phase_validation_failed']} (no proper asc/desc sequence or not cyclic)")
        print(f"        • First/last reps removed: {filter_stats['first_last_removed']}")
        print(f"        • Statistical outliers removed: {filter_stats['outliers_removed']}")
        if 'iqr_bounds' in filter_stats:
            print(f"        • IQR bounds: [{filter_stats['iqr_bounds']['lower']:.1f}°, {filter_stats['iqr_bounds']['upper']:.1f}°]")
        print(f"        • Valid reps after filtering: {rep_data_adv['num_repetitions']}")
        
        # Always save advanced results to dictionary (even if 0 reps)
        repetition_stats[f"{angle_col}_advanced"] = {
            'num_valid_repetitions': int(rep_data_adv['num_repetitions']),
            'repetition_times': rep_data_adv['repetition_times'].tolist() if len(rep_data_adv['repetition_times']) > 0 else [],
            'rep_durations': rep_data_adv['repetition_durations'].tolist() if len(rep_data_adv['repetition_durations']) > 0 else [],
            'rep_ranges': rep_data_adv['rep_ranges'].tolist() if len(rep_data_adv['rep_ranges']) > 0 else [],
            'rep_depths': rep_data_adv['rep_depths'].tolist() if len(rep_data_adv['rep_depths']) > 0 else [],
            'inter_rep_intervals': rep_data_adv['inter_rep_intervals'].tolist() if len(rep_data_adv['inter_rep_intervals']) > 0 else [],
            'filtering_applied': filter_stats,
            'quality_indices': {},  # Will be filled if reps > 0
            'fatigue_analysis': {},  # Will be filled if reps > 0
            'raw_cvs': {}  # Will be filled if reps > 0
        }
        
        if rep_data_adv['num_repetitions'] > 0:
            # Print advanced quality indices
            print(f"\n     🎯 ADVANCED QUALITY INDICES (0-100 scale):")
            print(f"        • ROM Consistency Index:   {rep_quality_adv['rom_consistency_index']:>6.1f}/100")
            print(f"        • Tempo Consistency Index: {rep_quality_adv['tempo_consistency_index']:>6.1f}/100")
            print(f"        • Depth Index:             {rep_quality_adv['depth_index']:>6.1f}/100")
            print(f"        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"        • IGM Score (weighted):    {rep_quality_adv['igm_score']:>6.1f}/100")
            print(f"        • Qualitative: {rep_quality_adv['qualitative_feedback']}")
            
            # Robust fatigue analysis
            print(f"\n     💪 ROBUST FATIGUE ANALYSIS:")
            fatigue = rep_quality_adv['fatigue_analysis']
            print(f"        • Trend: {fatigue['trend'].upper().replace('_', ' ')}")
            print(f"        • Slope: {fatigue['slope']:+.3f}°/rep")
            print(f"        • R² (fit quality): {fatigue['r_squared']:.3f}")
            print(f"        • Statistically significant: {'YES' if fatigue['significant'] else 'NO'}")
            
            if fatigue['trend'] == 'fatigue_detected':
                print(f"        ⚠️  FATIGUE DETECTED: Range decreasing over reps")
            elif fatigue['trend'] == 'progressive_exploration':
                print(f"        ℹ️  PROGRESSIVE EXPLORATION: Athlete exploring ROM limits")
            elif fatigue['trend'] == 'stable':
                print(f"        ✓ STABLE: Consistent ROM throughout exercise")
            elif fatigue['trend'] == 'inconsistent':
                print(f"        ⚠️  INCONSISTENT: No clear trend (R² < 0.5)")
            
            # Raw CVs for reference
            print(f"\n     📊 RAW COEFFICIENT OF VARIATION:")
            print(f"        • Range CV:  {rep_quality_adv['raw_cvs']['range_cv']*100:>5.1f}%")
            print(f"        • Tempo CV:  {rep_quality_adv['raw_cvs']['tempo_cv']*100:>5.1f}%")
            print(f"        • Timing CV: {rep_quality_adv['raw_cvs']['timing_cv']*100:>5.1f}%")
            
            # Valid repetitions list
            print(f"\n     ✅ VALID REPETITIONS (after filtering):")
            print(f"     ┌{'─'*75}┐")
            print(f"     │ {'Rep':<4} │ {'Time (s)':<10} │ {'Duration (s)':<12} │ {'Range (°)':<11} │ {'Depth (°)':<11} │")
            print(f"     ├{'─'*75}┤")
            
            for i in range(rep_data_adv['num_repetitions']):
                rep_num = i + 1
                rep_time = rep_data_adv['repetition_times'][i]
                rep_dur = rep_data_adv['repetition_durations'][i]
                rep_range = rep_data_adv['rep_ranges'][i]
                rep_depth = rep_data_adv['rep_depths'][i]
                print(f"     │ {rep_num:<4} │ {rep_time:>10.2f} │ {rep_dur:>12.2f} │ {rep_range:>11.2f} │ {rep_depth:>11.2f} │")
            
            print(f"     └{'─'*75}┘")
            
            # Update advanced results with quality metrics (only when reps > 0)
            repetition_stats[f"{angle_col}_advanced"]['quality_indices'] = {
                'rom_consistency_index': float(rep_quality_adv['rom_consistency_index']),
                'tempo_consistency_index': float(rep_quality_adv['tempo_consistency_index']),
                'depth_index': float(rep_quality_adv['depth_index']),
                'igm_score': float(rep_quality_adv['igm_score']),
                'qualitative_feedback': rep_quality_adv['qualitative_feedback']
            }
            repetition_stats[f"{angle_col}_advanced"]['fatigue_analysis'] = {
                'trend': fatigue['trend'],
                'slope': float(fatigue['slope']),
                'r_squared': float(fatigue['r_squared']),
                'significant': bool(fatigue['significant'])
            }
            repetition_stats[f"{angle_col}_advanced"]['raw_cvs'] = {
                'range_cv': float(rep_quality_adv['raw_cvs']['range_cv']),
                'tempo_cv': float(rep_quality_adv['raw_cvs']['tempo_cv']),
                'timing_cv': float(rep_quality_adv['raw_cvs']['timing_cv'])
            }
        else:
            print(f"\n     ⚠️  No valid repetitions after filtering")
            print(f"     Suggestion: Lower ROM threshold or check signal quality")
    
    except Exception as e:
        print(f"     ⚠️  Advanced analysis failed: {str(e)}")
        print(f"     Falling back to basic analysis only")

    # Plot the autocorrelation
    plt.figure(figsize=(10, 5))
    plt.stem(range(len(autocorr)), autocorr, basefmt=' ')
    plt.title(f"🔗 Autocorrelation Function - {angle_col.upper()}", fontsize=14, fontweight='bold')
    plt.xlabel("Lag (samples)", fontsize=11)
    plt.ylabel("Autocorrelation Coefficient", fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(path_of_this_script), f"autocorr_{angle_col}.png"), dpi=150, bbox_inches='tight')
    print(f"     ✅ Autocorrelation plot saved: autocorr_{angle_col}.png")
    plt.close()
    
    # Plot frequency spectrum (power spectral density) - FULL SPECTRUM
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Left plot: Full spectrum (0 to Nyquist frequency)
    ax1.semilogy(freqs, power, linewidth=1.5, color='#2E86AB', alpha=0.8)
    ax1.set_title(f"📊 Full Spectrum (0-{fs/2:.0f} Hz) - {angle_col.upper()}", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Frequency [Hz]", fontsize=11)
    ax1.set_ylabel("Power [dB]", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, fs/2])
    if dominant_freq > 0:
        ax1.axvline(x=dominant_freq, color='red', linestyle='--', linewidth=2, label=f'Dominant: {dominant_freq:.3f} Hz')
        ax1.legend(loc='upper right', fontsize=9)
    
    # Right plot: Zoom on biomechanics range (0-10 Hz for better detail)
    ax2.semilogy(freqs, power, linewidth=2, color='#A23B72')
    ax2.set_title(f"🔍 Biomechanics Range (0-10 Hz) - {angle_col.upper()}", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Frequency [Hz]", fontsize=11)
    ax2.set_ylabel("Power [dB]", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, min(10, fs/2)])
    if dominant_freq > 0 and dominant_freq < 10:
        ax2.axvline(x=dominant_freq, color='red', linestyle='--', linewidth=2, label=f'Dominant: {dominant_freq:.3f} Hz')
        ax2.legend(loc='upper right', fontsize=9)
    
    # Add annotations for frequency ranges
    ax1.axvspan(0, 5, alpha=0.1, color='green', label='Biomechanics (0-5 Hz)')
    ax1.axvspan(5, 50, alpha=0.05, color='yellow', label='Vibrations (5-50 Hz)')
    ax1.axvspan(50, fs/2, alpha=0.05, color='red', label=f'Noise (50-{fs/2:.0f} Hz)')
    ax1.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(path_of_this_script), f"spectrum_{angle_col}.png"), dpi=150, bbox_inches='tight')
    print(f"     ✅ Frequency spectrum plot saved: spectrum_{angle_col}.png")
    plt.close()
    
    # Generate spectrogram (time-frequency analysis)
    plt.figure(figsize=(12, 5))
    f, t_spec, Sxx = spectrogram(theta_detrended, fs=fs, nperseg=min(256, len(theta_detrended)//4), 
                                   noverlap=min(128, len(theta_detrended)//8))
    plt.pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency [Hz]', fontsize=11)
    plt.xlabel('Time [s]', fontsize=11)
    plt.title(f"🌈 Spectrogram (Time-Frequency) - {angle_col.upper()}", fontsize=14, fontweight='bold')
    plt.colorbar(label='Power [dB]')
    plt.ylim([0, min(5, fs/2)])  # Focus on low frequencies (0-5 Hz) for biomechanics
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(path_of_this_script), f"spectrogram_{angle_col}.png"), dpi=150, bbox_inches='tight')
    print(f"     ✅ Spectrogram saved: spectrogram_{angle_col}.png")
    plt.close()
    
    # === PHASE DETECTION VISUALIZATION ===
    # Generate for ALL signals (original + filtered)
    if angle_col in phase_stats and phase_stats[angle_col]['segments']:
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Plot signal
        ax.plot(ts, theta, linewidth=1.5, color='#2E86AB', alpha=0.7, label='Signal')
        
        # Color-code phases
        phase_colors = {
            'ascending': '#28a745',      # Green
            'descending': '#dc3545',     # Red
            'stable_high': '#ff9800',    # Orange (standing position)
            'stable_low': '#2196f3',     # Blue (bottom position)
            'stable': '#ffc107'          # Yellow (backward compatibility)
        }
        
        phases = phase_stats[angle_col]['segments']
        
        # Count phase types for legend
        phase_counts = {}
        for segment in phases:
            phase_type = segment['type']
            phase_counts[phase_type] = phase_counts.get(phase_type, 0) + 1
        
        # Shade regions
        for segment in phases:
            start_idx = segment['start']
            end_idx = segment['end']
            phase_type = segment['type']
            
            # Shade region
            ax.axvspan(ts[start_idx], ts[end_idx], 
                      alpha=0.2, 
                      color=phase_colors.get(phase_type, 'gray'),
                      label=None)
        
        # Add legend with actual phase types found
        from matplotlib.patches import Patch
        legend_elements = []
        phase_labels = {
            'ascending': '🔼 Ascending',
            'descending': '🔽 Descending', 
            'stable_high': '🟧 Stable High (standing)',
            'stable_low': '🟦 Stable Low (bottom)',
            'stable': '➡️ Stable'
        }
        
        for phase_type in ['ascending', 'descending', 'stable_high', 'stable_low', 'stable']:
            if phase_type in phase_counts:
                legend_elements.append(
                    Patch(facecolor=phase_colors[phase_type], alpha=0.3, 
                          label=f"{phase_labels[phase_type]} (n={phase_counts[phase_type]})")
                )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Angle [°]', fontsize=12)
        ax.set_title(f'📍 Phase Detection - {angle_col.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(path_of_this_script), f"phases_{angle_col}.png"), dpi=150, bbox_inches='tight')
        print(f"     ✅ Phase detection plot saved: phases_{angle_col}.png")
        plt.close()
    
    # === REPETITION DETECTION VISUALIZATION ===
    if angle_col in repetition_stats and repetition_stats[angle_col]['num_repetitions'] > 0:
        rs = repetition_stats[angle_col]
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Plot signal
        ax.plot(ts, theta, linewidth=1.5, color='#2E86AB', alpha=0.7, label='Signal')
        
        # Mark repetition valleys (bottom positions)
        if 'trough_indices' in rs and len(rs['trough_indices']) > 0:
            trough_indices = rs['trough_indices']
            trough_times = ts[trough_indices]
            trough_values = theta[trough_indices]
            suspicious_flags = np.array(rs.get('suspicious_reps', np.zeros(len(trough_indices), dtype=bool)))
            
            # Separate normal and suspicious reps
            normal_mask = ~suspicious_flags
            suspicious_mask = suspicious_flags
            
            # Plot normal repetitions
            if normal_mask.any():
                ax.scatter(trough_times[normal_mask], trough_values[normal_mask], 
                          s=150, c='red', marker='v', 
                          edgecolors='darkred', linewidths=2,
                          label=f"Valid Reps (n={normal_mask.sum()})", 
                          zorder=5)
            
            # Plot suspicious repetitions
            if suspicious_mask.any():
                ax.scatter(trough_times[suspicious_mask], trough_values[suspicious_mask], 
                          s=150, c='orange', marker='v', 
                          edgecolors='darkorange', linewidths=2,
                          label=f"⚠️ Suspicious (ROM<50%, n={suspicious_mask.sum()})", 
                          zorder=5)
            
            # Annotate each repetition
            for i, (t, val, is_suspicious) in enumerate(zip(trough_times, trough_values, suspicious_flags)):
                label = f'⚠#{i+1}' if is_suspicious else f'#{i+1}'
                color = 'darkorange' if is_suspicious else 'darkred'
                bgcolor = 'lightyellow' if is_suspicious else 'yellow'
                
                ax.annotate(label, 
                           xy=(t, val), 
                           xytext=(0, -15), 
                           textcoords='offset points',
                           ha='center', 
                           fontsize=9, 
                           fontweight='bold',
                           color=color,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=bgcolor, 
                                   alpha=0.7, 
                                   edgecolor=color))
        
        # Shade ROM for each repetition
        if 'rep_ranges' in rs and len(rs['rep_ranges']) > 0:
            for i in range(rs['num_repetitions']):
                rep_time = rs['repetition_times'][i]
                rep_range = rs['rep_ranges'][i]
                rep_depth = rs['rep_depths'][i]
                
                # Find peak in window around this rep
                window_start = max(0, np.where(ts >= rep_time - 2)[0][0] if any(ts >= rep_time - 2) else 0)
                window_end = min(len(ts)-1, np.where(ts <= rep_time + 2)[0][-1] if any(ts <= rep_time + 2) else len(ts)-1)
                
                if window_start < window_end:
                    window_signal = theta[window_start:window_end]
                    peak_val = np.max(window_signal)
                    
                    # Draw ROM range
                    ax.plot([rep_time, rep_time], [rep_depth, peak_val],
                           color='green', linewidth=3, alpha=0.5)
                    
                    # Annotate ROM
                    ax.text(rep_time, (rep_depth + peak_val) / 2,
                           f'{rep_range:.1f}°',
                           fontsize=8, color='darkgreen',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   alpha=0.8, 
                                   edgecolor='green'),
                           ha='left')
        
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Angle [°]', fontsize=12)
        ax.set_title(f'🏋️ Repetition Detection - {angle_col.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(path_of_this_script), f"repetitions_{angle_col}.png"), dpi=150, bbox_inches='tight')
        print(f"     ✅ Repetition detection plot saved: repetitions_{angle_col}.png")
        plt.close()
    
    # === ADVANCED REPETITION VISUALIZATION (if available) ===
    if "_filtered" not in angle_col and f"{angle_col}_advanced" in repetition_stats:
        rs_adv = repetition_stats[f"{angle_col}_advanced"]
        
        if rs_adv['num_valid_repetitions'] > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
            
            # Top panel: Base detection
            if angle_col in repetition_stats and repetition_stats[angle_col]['num_repetitions'] > 0:
                rs_base = repetition_stats[angle_col]
                ax1.plot(ts, theta, linewidth=1.5, color='#6c757d', alpha=0.5, label='Signal')
                
                if 'trough_indices' in rs_base and len(rs_base['trough_indices']) > 0:
                    trough_times = ts[rs_base['trough_indices']]
                    trough_values = theta[rs_base['trough_indices']]
                    ax1.scatter(trough_times, trough_values, 
                              s=100, c='orange', marker='o', 
                              alpha=0.6, edgecolors='darkorange',
                              label=f"Base Detection (n={len(trough_times)})")
            
            ax1.set_ylabel('Angle [°]', fontsize=12)
            ax1.set_title(f'🏋️ BASE vs ADVANCED Repetition Detection - {angle_col.upper()}', 
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Bottom panel: Advanced detection (filtered valid reps)
            ax2.plot(ts, theta, linewidth=1.5, color='#2E86AB', alpha=0.7, label='Signal')
            
            # Mark valid repetitions
            valid_times = rs_adv['repetition_times']
            valid_depths = rs_adv['rep_depths']
            
            ax2.scatter(valid_times, valid_depths, 
                      s=200, c='lime', marker='*', 
                      edgecolors='darkgreen', linewidths=2,
                      label=f"✅ Valid Reps (n={rs_adv['num_valid_repetitions']})", 
                      zorder=5)
            
            # Annotate valid reps with quality score
            filter_stats = rs_adv['filtering_applied']
            for i, (t, val) in enumerate(zip(valid_times, valid_depths)):
                ax2.annotate(f'✓{i+1}', 
                           xy=(t, val), 
                           xytext=(0, -15), 
                           textcoords='offset points',
                           ha='center', 
                           fontsize=10, 
                           fontweight='bold',
                           color='darkgreen',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='lightgreen', 
                                   alpha=0.8, 
                                   edgecolor='darkgreen'))
            
            # Show ROM threshold line
            if 'min_rom_threshold' in filter_stats:
                rom_threshold = filter_stats['min_rom_threshold']
                mean_signal = np.mean(theta)
                ax2.axhline(y=mean_signal - rom_threshold/2, 
                          color='red', linestyle='--', linewidth=2, 
                          alpha=0.5, label=f'ROM Threshold: {rom_threshold:.1f}°')
            
            # Show IQR bounds if available
            if 'iqr_bounds' in filter_stats:
                lower = filter_stats['iqr_bounds']['lower']
                upper = filter_stats['iqr_bounds']['upper']
                ax2.axhspan(mean_signal - upper/2, mean_signal - lower/2, 
                          alpha=0.1, color='green', 
                          label=f'IQR Acceptance Range')
            
            ax2.set_xlabel('Time [s]', fontsize=12)
            ax2.set_ylabel('Angle [°]', fontsize=12)
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # Add text box with filtering summary
            filter_text = (
                f"Filtering Applied:\n"
                f"• Total detected: {filter_stats['total_detected_before_filtering']}\n"
                f"• ROM threshold: {filter_stats.get('reps_below_rom_threshold', 0)} removed\n"
                f"• First/last: {'2 removed' if filter_stats['first_last_removed'] else 'kept'}\n"
                f"• Outliers: {filter_stats['outliers_removed']} removed\n"
                f"• Valid: {rs_adv['num_valid_repetitions']} ({(rs_adv['num_valid_repetitions']/filter_stats['total_detected_before_filtering']*100):.0f}%)"
            )
            ax2.text(0.02, 0.98, filter_text, 
                    transform=ax2.transAxes, 
                    fontsize=9, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(path_of_this_script), f"repetitions_comparison_{angle_col}.png"), 
                       dpi=150, bbox_inches='tight')
            print(f"     ✅ Base vs Advanced comparison plot saved: repetitions_comparison_{angle_col}.png")
            plt.close()

    # === TEMPORAL EVOLUTION PLOTS (Per-Repetition Timing) ===
    if 'temporal_parameters' in repetition_stats.get(angle_col, {}):
        tp = repetition_stats[angle_col]['temporal_parameters']
        per_rep = tp['per_repetition']
        
        if len(per_rep) > 0:
            print(f"     📊 Generating temporal evolution plots for {angle_col.upper()}...")
            
            # Extract time series data
            rep_numbers = np.arange(1, len(per_rep) + 1)
            eccentric_times = np.array([r['eccentric_time'] for r in per_rep])
            concentric_times = np.array([r['concentric_time'] for r in per_rep])
            bottom_hold_times = np.array([r['bottom_hold_time'] for r in per_rep])
            top_hold_times = np.array([r['top_hold_time'] for r in per_rep])
            rep_durations = np.array([r['total_time'] for r in per_rep])
            
            # Create figure with 5 subplots
            fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
            
            # 1. Total Repetition Duration
            ax = axes[0]
            ax.plot(rep_numbers, rep_durations, marker='o', linewidth=2, 
                   markersize=8, color='#2E86AB', label='Rep Duration')
            ax.axhline(y=np.mean(rep_durations), color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Mean: {np.mean(rep_durations):.2f}s')
            ax.fill_between(rep_numbers, rep_durations, alpha=0.3, color='#2E86AB')
            ax.set_ylabel('Duration [s]', fontsize=11, fontweight='bold')
            ax.set_title(f'⏱️ Temporal Evolution - {angle_col.upper()}\n1. Total Repetition Duration', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add trend line
            z = np.polyfit(rep_numbers, rep_durations, 1)
            p = np.poly1d(z)
            ax.plot(rep_numbers, p(rep_numbers), "--", color='orange', linewidth=1.5, 
                   alpha=0.8, label=f'Trend: {z[0]:+.3f}s/rep')
            
            # 2. Eccentric Phase (Descending)
            ax = axes[1]
            ax.plot(rep_numbers, eccentric_times, marker='s', linewidth=2, 
                   markersize=7, color='#dc3545', label='Eccentric Time')
            ax.axhline(y=np.mean(eccentric_times), color='darkred', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Mean: {np.mean(eccentric_times):.2f}s')
            ax.fill_between(rep_numbers, eccentric_times, alpha=0.3, color='#dc3545')
            ax.set_ylabel('Duration [s]', fontsize=11, fontweight='bold')
            ax.set_title('2. Eccentric Phase Duration (Descending)', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add trend
            z = np.polyfit(rep_numbers, eccentric_times, 1)
            p = np.poly1d(z)
            ax.plot(rep_numbers, p(rep_numbers), "--", color='orange', linewidth=1.5, 
                   alpha=0.8, label=f'Trend: {z[0]:+.3f}s/rep')
            
            # 3. Concentric Phase (Ascending)
            ax = axes[2]
            ax.plot(rep_numbers, concentric_times, marker='^', linewidth=2, 
                   markersize=7, color='#28a745', label='Concentric Time')
            ax.axhline(y=np.mean(concentric_times), color='darkgreen', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Mean: {np.mean(concentric_times):.2f}s')
            ax.fill_between(rep_numbers, concentric_times, alpha=0.3, color='#28a745')
            ax.set_ylabel('Duration [s]', fontsize=11, fontweight='bold')
            ax.set_title('3. Concentric Phase Duration (Ascending)', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add trend
            z = np.polyfit(rep_numbers, concentric_times, 1)
            p = np.poly1d(z)
            ax.plot(rep_numbers, p(rep_numbers), "--", color='orange', linewidth=1.5, 
                   alpha=0.8, label=f'Trend: {z[0]:+.3f}s/rep')
            
            # 4. Bottom Hold Phase
            ax = axes[3]
            ax.plot(rep_numbers, bottom_hold_times, marker='v', linewidth=2, 
                   markersize=7, color='#2196f3', label='Bottom Hold Time')
            ax.axhline(y=np.mean(bottom_hold_times), color='darkblue', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Mean: {np.mean(bottom_hold_times):.2f}s')
            ax.fill_between(rep_numbers, bottom_hold_times, alpha=0.3, color='#2196f3')
            ax.set_ylabel('Duration [s]', fontsize=11, fontweight='bold')
            ax.set_title('4. Bottom Hold Duration', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add trend
            z = np.polyfit(rep_numbers, bottom_hold_times, 1)
            p = np.poly1d(z)
            ax.plot(rep_numbers, p(rep_numbers), "--", color='orange', linewidth=1.5, 
                   alpha=0.8, label=f'Trend: {z[0]:+.3f}s/rep')
            
            # 5. Top Hold Phase
            ax = axes[4]
            ax.plot(rep_numbers, top_hold_times, marker='D', linewidth=2, 
                   markersize=6, color='#ff9800', label='Top Hold Time')
            ax.axhline(y=np.mean(top_hold_times), color='darkorange', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Mean: {np.mean(top_hold_times):.2f}s')
            ax.fill_between(rep_numbers, top_hold_times, alpha=0.3, color='#ff9800')
            ax.set_ylabel('Duration [s]', fontsize=11, fontweight='bold')
            ax.set_xlabel('Repetition Number', fontsize=11, fontweight='bold')
            ax.set_title('5. Top Hold Duration', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add trend
            z = np.polyfit(rep_numbers, top_hold_times, 1)
            p = np.poly1d(z)
            ax.plot(rep_numbers, p(rep_numbers), "--", color='orange', linewidth=1.5, 
                   alpha=0.8, label=f'Trend: {z[0]:+.3f}s/rep')
            
            plt.tight_layout()
            temporal_plot_filename = f"temporal_evolution_{angle_col}.png"
            plt.savefig(os.path.join(os.path.dirname(path_of_this_script), temporal_plot_filename), 
                       dpi=150, bbox_inches='tight')
            print(f"     ✅ Temporal evolution plot saved: {temporal_plot_filename}")
            plt.close()

# === GENERATE INTERACTIVE PLOTLY PLOTS FOR RAW SIGNALS ===
print(f"\n{'='*80}")
print("📊 GENERATING INTERACTIVE PLOTLY PLOTS")
print(f"{'='*80}")
print("  Creating interactive plots data for embedding in HTML...")

# Dictionary to store Plotly figure objects for each signal
plotly_figures = {}

for angle_col in ["ang1", "ang2", "ang3", "ang_principal",
                  "ang1_filtered", "ang2_filtered", "ang3_filtered", "ang_principal_filtered"]:
    if angle_col not in repetition_stats:
        continue
    
    theta = resampled_df[angle_col].to_numpy()
    ts_plot = resampled_df["ts_s"].to_numpy()
    
    print(f"  • Generating interactive plot data for {angle_col.upper()}...")
    
    # Create interactive figure
    fig = go.Figure()
    
    # Add main signal trace
    fig.add_trace(go.Scatter(
        x=ts_plot,
        y=theta,
        mode='lines',
        name='Signal',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='<b>Time</b>: %{x:.2f}s<br><b>Angle</b>: %{y:.2f}°<extra></extra>'
    ))
    
    # Add repetition markers if available
    rs = repetition_stats[angle_col]
    if rs['num_repetitions'] > 0 and 'trough_indices' in rs:
        trough_indices = rs['trough_indices']
        trough_times = ts_plot[trough_indices]
        trough_values = theta[trough_indices]
        suspicious_flags = np.array(rs.get('suspicious_reps', np.zeros(len(trough_indices), dtype=bool)))
        
        # Valid repetitions
        normal_mask = ~suspicious_flags
        if normal_mask.any():
            fig.add_trace(go.Scatter(
                x=trough_times[normal_mask],
                y=trough_values[normal_mask],
                mode='markers',
                name='Valid Reps',
                marker=dict(
                    size=12,
                    color='#28a745',
                    symbol='triangle-down',
                    line=dict(width=2, color='darkgreen')
                ),
                hovertemplate='<b>Rep #%{text}</b><br>Time: %{x:.2f}s<br>Angle: %{y:.2f}°<extra></extra>',
                text=[f"{i+1}" for i, flag in enumerate(suspicious_flags) if not flag]
            ))
        
        # Suspicious repetitions
        if suspicious_flags.any():
            suspicious_mask = suspicious_flags
            fig.add_trace(go.Scatter(
                x=trough_times[suspicious_mask],
                y=trough_values[suspicious_mask],
                mode='markers',
                name='⚠️ Suspicious',
                marker=dict(
                    size=12,
                    color='#ffc107',
                    symbol='triangle-down',
                    line=dict(width=2, color='#ff6b00')
                ),
                hovertemplate='<b>⚠️ Rep #%{text}</b><br>Time: %{x:.2f}s<br>Angle: %{y:.2f}°<br><i>ROM < 50%</i><extra></extra>',
                text=[f"{i+1}" for i, flag in enumerate(suspicious_flags) if flag]
            ))
    
    # Update layout
    fig.update_layout(
        title=None,  # No title since it's embedded
        xaxis=dict(
            title='Time [s]',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='Angle [°]',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        margin=dict(l=60, r=40, t=20, b=60)
    )
    
    # Store figure for this signal
    plotly_figures[angle_col] = fig
    
    # Extract session ID from filename (e.g., "FILE_EULER2015-1-1-13-06-49.txt" -> "2015-1-1-13-06-49")
    session_id = file_name.replace("FILE_EULER", "").replace(".txt", "")
    
    # Save interactive HTML with session-specific name
    interactive_html_path = os.path.join(os.path.dirname(path_of_this_script), 
                                          f"interactive_{angle_col}_{session_id}.html")
    fig.write_html(interactive_html_path, include_plotlyjs='cdn')
    
    print(f"  ✅ Interactive plot saved: interactive_{angle_col}_{session_id}.html")

print(f"{'='*80}\n")

# === GENERATE COMPREHENSIVE OVERVIEW PLOTS ===
print(f"\n{'='*80}")
print("📊 GENERATING COMPREHENSIVE OVERVIEW PLOTS")
print(f"{'='*80}")
print("  Creating full time-series with all detected regions for ALL signals...")

for axis_col in ["ang1", "ang2", "ang3", "ang_principal",
                  "ang1_filtered", "ang2_filtered", "ang3_filtered", "ang_principal_filtered"]:
    if axis_col not in phase_stats or axis_col not in repetition_stats:
        continue
    
    theta = resampled_df[axis_col].to_numpy()
    ts_plot = resampled_df["ts_s"].to_numpy()
    
    print(f"  • Generating overview plot for {axis_col.upper()}...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    
    # ========== TOP PANEL: Phase Regions ==========
    ax1.plot(ts_plot, theta, linewidth=1.5, color='#2E86AB', alpha=0.8, label='Signal', zorder=3)
    
    # Draw phase regions as background
    if phase_stats[axis_col]['segments']:
        phase_colors = {
            'ascending': '#28a745',      # Green
            'descending': '#dc3545',     # Red
            'stable_high': '#ff9800',    # Orange (standing)
            'stable_low': '#2196f3',     # Blue (bottom)
            'stable': '#ffc107'          # Yellow (backward compatibility)
        }
        phase_segments = phase_stats[axis_col]['segments']
        phase_counts = {}
        
        for segment in phase_segments:
            phase_type = segment['type']
            phase_counts[phase_type] = phase_counts.get(phase_type, 0) + 1
            
            start_idx = segment['start']
            end_idx = segment['end']
            
            ax1.axvspan(ts_plot[start_idx], ts_plot[end_idx], 
                       alpha=0.15, 
                       color=phase_colors.get(phase_type, 'gray'),
                       zorder=1)
        
        # Legend with actual phase types found
        from matplotlib.patches import Patch
        phase_labels = {
            'ascending': '🔼 Ascending',
            'descending': '🔽 Descending',
            'stable_high': '🟧 Stable High (standing)',
            'stable_low': '🟦 Stable Low (bottom)',
            'stable': '➡️ Stable'
        }
        
        phase_legend = []
        for phase_type in ['ascending', 'descending', 'stable_high', 'stable_low', 'stable']:
            if phase_type in phase_counts:
                phase_legend.append(
                    Patch(facecolor=phase_colors[phase_type], alpha=0.3, 
                          label=f"{phase_labels[phase_type]} (n={phase_counts[phase_type]})")
                )
        
        ax1.legend(handles=phase_legend, loc='upper right', fontsize=10, ncol=3)
    
    ax1.set_ylabel('Angle [°]', fontsize=12, fontweight='bold')
    ax1.set_title(f'🎯 COMPLETE ANALYSIS - {axis_col.upper()}\nTop: Phase Regions | Bottom: Repetitions', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    # ========== BOTTOM PANEL: Repetitions ==========
    ax2.plot(ts_plot, theta, linewidth=1.5, color='#2E86AB', alpha=0.7, label='Signal', zorder=3)
    
    # Mark repetitions
    rs = repetition_stats[axis_col]
    if rs['num_repetitions'] > 0 and 'trough_indices' in rs:
        trough_indices = rs['trough_indices']
        trough_times = ts_plot[trough_indices]
        trough_values = theta[trough_indices]
        suspicious_flags = np.array(rs.get('suspicious_reps', np.zeros(len(trough_indices), dtype=bool)))
        
        # Separate valid and suspicious
        normal_mask = ~suspicious_flags
        suspicious_mask = suspicious_flags
        
        # Plot valid reps
        if normal_mask.any():
            ax2.scatter(trough_times[normal_mask], trough_values[normal_mask], 
                       s=200, c='#28a745', marker='v', 
                       edgecolors='darkgreen', linewidths=2.5,
                       label=f"✅ Valid Reps (n={normal_mask.sum()})", 
                       zorder=5)
        
        # Plot suspicious reps
        if suspicious_mask.any():
            ax2.scatter(trough_times[suspicious_mask], trough_values[suspicious_mask], 
                       s=200, c='#ffc107', marker='v', 
                       edgecolors='#ff6b00', linewidths=2.5,
                       label=f"⚠️ SUSPICIOUS (ROM<50%, n={suspicious_mask.sum()})", 
                       zorder=5)
            
            # Highlight suspicious regions
            for t_susp, v_susp in zip(trough_times[suspicious_mask], trough_values[suspicious_mask]):
                ax2.axvspan(t_susp - 0.5, t_susp + 0.5, 
                           alpha=0.2, color='orange', zorder=1)
        
        # Annotate rep numbers
        for i, (t, val, is_susp) in enumerate(zip(trough_times, trough_values, suspicious_flags)):
            label_text = f"⚠️#{i+1}" if is_susp else f"#{i+1}"
            bgcolor = '#fff3cd' if is_susp else '#d4edda'
            edgecolor = '#ff6b00' if is_susp else 'darkgreen'
            
            ax2.annotate(label_text, 
                        xy=(t, val), 
                        xytext=(0, -20), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=9, 
                        fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor=bgcolor, 
                                alpha=0.9, 
                                edgecolor=edgecolor,
                                linewidth=2))
    
    ax2.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Angle [°]', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    plt.tight_layout()
    overview_filename = f"overview_complete_{axis_col}.png"
    plt.savefig(os.path.join(os.path.dirname(path_of_this_script), overview_filename), 
               dpi=150, bbox_inches='tight')
    print(f"  ✅ Complete overview saved: {overview_filename}")
    plt.close()

print(f"{'='*80}\n")

print(f"\n{'='*80}")
print("📊 CROSS-AXIS CORRELATIONS")
print(f"{'='*80}")
for k, v in cross_corrs.items():
    ax1, ax2 = k.split('_')
    print(f"  • {ax1.upper()} ↔ {ax2.upper()}: {v:+.3f}")

# Lagged cross-correlations
print(f"\n🔎 Lagged Cross-Correlations:")
if len(ts) > 1:
    fs = len(ts) / (ts[-1] - ts[0])
else:
    fs = 1.0
max_lag = int(fs * 1.0)  # 1 second max lag

for pair in [('ang1', 'ang2'), ('ang1', 'ang3'), ('ang2', 'ang3')]:
    lags, corr_xy = cross_corr_lags(
        resampled_df[pair[0]].to_numpy(),
        resampled_df[pair[1]].to_numpy(),
        max_lag, fs
    )
    lag_max_idx = np.argmax(np.abs(corr_xy))
    lag_max = lags[lag_max_idx]
    corr_max = corr_xy[lag_max_idx]
    print(f"  • {pair[0].upper()}↔{pair[1].upper()}: max corr={corr_max:+.3f} at lag={lag_max/fs:+.3f}s ({lag_max:+d} samples)")

print(f"\n{'='*80}")
print("✓ ANALYSIS COMPLETE")
print(f"{'='*80}\n")


def export_to_excel(cycle_stats, phase_stats, repetition_stats, resampled_df, output_path):
    """Export all analysis data to comprehensive Excel workbook with multiple sheets.
    
    Args:
        cycle_stats: Dictionary with cycle statistics per axis
        phase_stats: Dictionary with phase statistics per axis
        repetition_stats: Dictionary with repetition statistics per axis
        resampled_df: Resampled dataframe with signals
        output_path: Path for the Excel file
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # ===== SHEET 1: SIGNAL DATA =====
            # Export resampled signals (time series)
            signal_cols = ['ts_s', 'ang1', 'ang2', 'ang3', 'ang_principal',
                          'ang1_filtered', 'ang2_filtered', 'ang3_filtered', 'ang_principal_filtered']
            signal_df = resampled_df[signal_cols].copy()
            signal_df.to_excel(writer, sheet_name='Signal_Data', index=False)
            
            # ===== SHEET 2: CYCLE STATISTICS =====
            cycle_rows = []
            for axis, cs in cycle_stats.items():
                cycle_rows.append({
                    'Axis': axis,
                    'Num Cycles': cs['num_cycles'],
                    'Num Peaks': cs['num_peaks'],
                    'Num Troughs': cs['num_troughs'],
                    'Mean Cycle Duration (s)': cs['mean_cycle_duration'],
                    'CV Cycle Duration': cs['cv_cycle_duration'],
                    'Mean Cycle Amplitude (°)': cs['mean_cycle_amplitude'],
                    'CV Cycle Amplitude': cs['cv_cycle_amplitude'],
                    'Trend Duration (s/cycle)': cs['trend_duration_per_cycle'],
                    'Trend Amplitude (°/cycle)': cs['trend_amplitude_per_cycle']
                })
            
            cycle_df = pd.DataFrame(cycle_rows)
            cycle_df.to_excel(writer, sheet_name='Cycle_Summary', index=False)
            
            # ===== SHEET 3: PHASE STATISTICS =====
            phase_rows = []
            for axis, ps in phase_stats.items():
                stats = ps['statistics']
                for phase_type, pstats in stats.items():
                    if pstats['count'] > 0:
                        phase_rows.append({
                            'Axis': axis,
                            'Phase Type': phase_type,
                            'Count': pstats['count'],
                            'Mean Duration (s)': pstats['mean_duration'],
                            'CV Duration': pstats['cv_duration'],
                            'Total Time (s)': pstats['total_time']
                        })
            
            phase_df = pd.DataFrame(phase_rows)
            phase_df.to_excel(writer, sheet_name='Phase_Summary', index=False)
            
            # ===== SHEET 4-7: REPETITION ANALYSIS PER AXIS =====
            for axis in ['ang1', 'ang2', 'ang3', 'ang_principal',
                        'ang1_filtered', 'ang2_filtered', 'ang3_filtered', 'ang_principal_filtered']:
                if axis not in repetition_stats:
                    continue
                
                rs = repetition_stats[axis]
                
                # Check if required keys exist
                if 'num_repetitions' not in rs:
                    continue
                    
                num_reps = rs['num_repetitions']
                
                if num_reps == 0:
                    continue
                
                # Check if required arrays exist
                if not all(k in rs for k in ['repetition_times', 'rep_durations', 'rep_ranges', 'rep_depths']):
                    continue
                
                # Build per-repetition dataframe
                rep_rows = []
                for i in range(num_reps):
                    row = {
                        'Rep #': i + 1,
                        'Time (s)': rs['repetition_times'][i] if i < len(rs['repetition_times']) else 0,
                        'Duration (s)': rs['rep_durations'][i] if i < len(rs['rep_durations']) else 0,
                        'Range (°)': rs['rep_ranges'][i] if i < len(rs['rep_ranges']) else 0,
                        'Depth (°)': rs['rep_depths'][i] if i < len(rs['rep_depths']) else 0,
                        'Suspicious': rs.get('suspicious_reps', [False]*num_reps)[i],
                        'Valid': True
                    }
                    
                    # Add validation info
                    if rs.get('has_validation', False) and i < len(rs.get('validation_results', [])):
                        val = rs['validation_results'][i]
                        row['Valid'] = val['is_valid']
                        row['Has Proper Phases'] = val['has_proper_phases']
                        row['Is Cyclic'] = val['is_cyclic']
                        row['Angle Recovery (°)'] = val['angle_recovery']
                        row['Start Angle (°)'] = val['start_angle']
                        row['End Angle (°)'] = val['end_angle']
                    
                    # Add temporal parameters if available
                    if 'temporal_parameters' in rs and rs['temporal_parameters']:
                        tp = rs['temporal_parameters']
                        if i < len(tp['per_repetition']):
                            per_rep = tp['per_repetition'][i]
                            row['Eccentric Time (s)'] = per_rep['eccentric_time']
                            row['Concentric Time (s)'] = per_rep['concentric_time']
                            row['Bottom Hold Time (s)'] = per_rep['bottom_hold_time']
                            row['Top Hold Time (s)'] = per_rep['top_hold_time']
                            row['Eccentric %'] = per_rep['eccentric_pct']
                            row['Concentric %'] = per_rep['concentric_pct']
                    
                    rep_rows.append(row)
                
                rep_df = pd.DataFrame(rep_rows)
                
                # Sanitize sheet name (Excel limit: 31 chars, no special chars)
                sheet_name = f'Reps_{axis}'.replace('_filtered', '_filt')[:31]
                rep_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # ===== SHEET 8: QUALITY METRICS =====
            quality_rows = []
            for axis, rs in repetition_stats.items():
                # Skip if missing required data
                if not isinstance(rs, dict) or 'num_repetitions' not in rs:
                    continue
                    
                if rs['num_repetitions'] > 0 and 'quality' in rs:
                    quality = rs['quality']
                    quality_rows.append({
                        'Axis': axis,
                        'Num Repetitions': rs['num_repetitions'],
                        'Num Suspicious': rs.get('num_suspicious', 0),
                        'Num Invalid': rs.get('num_invalid', 0),
                        'Consistency Score': quality.get('consistency_score', 0),
                        'Range CV': quality.get('range_cv', 0),
                        'Timing CV': quality.get('timing_cv', 0),
                        'Tempo CV': quality.get('tempo_cv', 0),
                        'Fatigue Indicator (°/rep)': quality.get('fatigue_indicator', 0)
                    })
            
            if quality_rows:  # Only create sheet if we have data
                quality_df = pd.DataFrame(quality_rows)
                quality_df.to_excel(writer, sheet_name='Quality_Metrics', index=False)
            
            # ===== SHEET 9: TEMPORAL PARAMETERS SUMMARY =====
            temporal_summary_rows = []
            for axis, rs in repetition_stats.items():
                # Skip if missing required data structure
                if not isinstance(rs, dict) or 'temporal_parameters' not in rs or not rs['temporal_parameters']:
                    continue
                
                tp = rs['temporal_parameters']
                
                # Global timing - check structure before accessing
                if 'global_timing' in tp:
                    gt = tp['global_timing']
                    row = {
                        'Axis': axis,
                        'Category': 'Global Timing',
                        'Mean Rep Duration (s)': gt.get('mean_rep_duration', 0),
                        'Mean Cycle Time (s)': gt.get('mean_cycle_time', 0),
                        'Execution Frequency (Hz)': gt.get('execution_frequency_hz', 0),
                        'Mean Density Ratio': gt.get('mean_density_ratio', 0)
                    }
                    temporal_summary_rows.append(row)
                
                # Session totals
                if 'session_totals' in tp:
                    st = tp['session_totals']
                    row = {
                        'Axis': axis,
                        'Category': 'Session Totals',
                        'Total Eccentric (s)': st.get('total_eccentric_time', 0),
                        'Total Concentric (s)': st.get('total_concentric_time', 0),
                        'Total Bottom Hold (s)': st.get('total_bottom_hold_time', 0),
                        'Total Top Hold (s)': st.get('total_top_hold_time', 0),
                        'Total Work Time (s)': st.get('total_work_time', 0),
                        'Total Pause Time (s)': st.get('total_pause_time', 0)
                    }
                    temporal_summary_rows.append(row)
                
                # Variances (summary)
                if 'temporal_variances' in tp:
                    tv = tp['temporal_variances']
                    row = {
                        'Axis': axis,
                        'Category': 'Temporal Variances',
                        'Eccentric CV': tv.get('eccentric', {}).get('cv', 0),
                        'Concentric CV': tv.get('concentric', {}).get('cv', 0),
                        'Bottom Hold CV': tv.get('bottom_hold', {}).get('cv', 0),
                        'Top Hold CV': tv.get('top_hold', {}).get('cv', 0)
                    }
                    temporal_summary_rows.append(row)
                
                # Outliers
                if 'outliers' in tp:
                    out = tp['outliers']
                    row = {
                        'Axis': axis,
                        'Category': 'Outliers',
                        'Slow Eccentric': out.get('num_slow_eccentric', 0),
                        'Fast Concentric': out.get('num_fast_concentric', 0),
                        'Excessive Bottom Hold': out.get('num_excessive_bottom_hold', 0),
                        'Excessive Top Hold': out.get('num_excessive_top_hold', 0)
                    }
                    temporal_summary_rows.append(row)
                
                # Trends
                if 'longitudinal_trends' in tp:
                    lt = tp['longitudinal_trends']
                    row = {
                        'Axis': axis,
                        'Category': 'Longitudinal Trends',
                        'Eccentric Trend (s/rep)': lt.get('eccentric_trend', 0),
                        'Concentric Trend (s/rep)': lt.get('concentric_trend', 0),
                        'Bottom Hold Trend (s/rep)': lt.get('bottom_hold_trend', 0),
                        'Top Hold Trend (s/rep)': lt.get('top_hold_trend', 0),
                        'Rep Duration Trend (s/rep)': lt.get('rep_duration_trend', 0),
                        'Cycle Time Trend (s/rep)': lt.get('cycle_time_trend', 0)
                    }
                    temporal_summary_rows.append(row)
            
            temporal_summary_df = pd.DataFrame(temporal_summary_rows)
            temporal_summary_df.to_excel(writer, sheet_name='Temporal_Parameters', index=False)
            
        return True
    except Exception as e:
        print(f"⚠️  Warning: Excel export failed: {e}")
        return False


# Save comprehensive HTML report
print("-"*80)
print("GENERATING REPORTS")
print("-"*80)
report_path = file_path.replace(".txt", "_report.html")

# Compute sampling statistics for HTML
dt_all = np.diff(df['ts_ms'].to_numpy())
dt_mean_ms = np.mean(dt_all)
dt_std_ms = np.std(dt_all)
dt_min_ms = np.min(dt_all)
dt_max_ms = np.max(dt_all)
sampling_threshold = 3.0
gap_indices = np.where(dt_all > sampling_threshold)[0]
num_gaps = len(gap_indices)

with open(report_path, "w") as f:
    # HTML header with modern CSS styling
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Biomechanics Signal Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header .subtitle { font-size: 1.1em; opacity: 0.9; }
        .content { padding: 40px; }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .summary-card h3 { font-size: 0.9em; text-transform: uppercase; opacity: 0.9; margin-bottom: 10px; }
        .summary-card .value { font-size: 2em; font-weight: bold; }
        
        .section {
            margin-bottom: 40px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }
        .section-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: bold;
            font-size: 1.2em;
        }
        .section-header:hover { background: linear-gradient(90deg, #764ba2 0%, #667eea 100%); }
        .section-content { padding: 25px; background: #fafafa; }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .comparison-table thead { background: #667eea; color: white; }
        .comparison-table th, .comparison-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        .comparison-table tr:hover { background: #f5f5f5; }
        .comparison-table .metric-name { font-weight: bold; color: #667eea; }
        .comparison-table .original { background: #fff3cd; }
        .comparison-table .filtered { background: #d4edda; }
        .improvement { color: #28a745; font-weight: bold; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .stat-item .label { color: #666; font-size: 0.9em; margin-bottom: 5px; }
        .stat-item .value { font-size: 1.3em; font-weight: bold; color: #333; }
        
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .plot-container {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .plot-container img { width: 100%; border-radius: 5px; }
        .plot-title { font-weight: bold; color: #667eea; margin-bottom: 10px; text-align: center; }
        
        .correlation-matrix {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        .corr-item {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .corr-item .axes { font-size: 1.1em; color: #667eea; font-weight: bold; }
        .corr-item .value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .corr-item .lag { font-size: 0.9em; color: #666; }
        
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin: 5px;
        }
        .badge-success { background: #d4edda; color: #155724; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-info { background: #d1ecf1; color: #0c5460; }
        
        @media print {
            body { background: white; }
            .section-content { display: block !important; }
        }
        
        /* Global expand/collapse button */
        .global-controls {
            position: sticky;
            top: 20px;
            z-index: 1000;
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 30px;
        }
        .global-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
        }
        .global-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        .global-btn:active {
            transform: translateY(0);
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🔬 Biomechanics Signal Analysis Report</h1>
        <p class="subtitle">Advanced Statistical Analysis with Noise Filtering</p>
        <p class="subtitle">""" + os.path.basename(file_path) + """</p>
    </div>
    
    <div class="content">
        <!-- Global expand/collapse controls -->
        <div class="global-controls">
            <button class="global-btn" onclick="expandAll()">📂 Expand All Sections</button>
            <button class="global-btn" onclick="collapseAll()">📁 Collapse All Sections</button>
        </div>
        
        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <h3>📊 Total Samples</h3>
                <div class="value">""" + f"{len(resampled_df):,}" + """</div>
            </div>
            <div class="summary-card">
                <h3>⏱️ Duration</h3>
                <div class="value">""" + f"{resampled_df['ts_s'].iloc[-1]:.2f}" + """ s</div>
            </div>
            <div class="summary-card">
                <h3>📈 Sampling Rate</h3>
                <div class="value">""" + f"{len(resampled_df) / resampled_df['ts_s'].iloc[-1]:.1f}" + """ Hz</div>
            </div>
            <div class="summary-card">
                <h3>🔧 Filter Window</h3>
                <div class="value">""" + f"{window_size}" + """ samples</div>
            </div>
        </div>
        
        <!-- Introduction & Methodology -->
        <div class="section">
            <div class="section-header" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display=='none'?'block':'none'">
                <span>📖 Introduction & Methodology</span>
                <span>▼</span>
            </div>
            <div class="section-content" style="display: block;">
                <div style="padding: 20px; line-height: 1.8;">
                    <h3 style="color: #667eea; margin-bottom: 15px;">🎯 Analysis Overview</h3>
                    <p style="margin-bottom: 15px;">
                        This report presents a <strong>comprehensive biomechanical analysis</strong> of movement patterns 
                        captured via smartphone IMU sensors. The analysis pipeline processes raw 3D Euler angles 
                        (pitch, roll, yaw) through multiple stages of signal processing, statistical analysis, 
                        and biomechanical interpretation.
                    </p>
                    
                    <h3 style="color: #667eea; margin: 25px 0 15px 0;">🔬 Analysis Pipeline</h3>
                    <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; border-radius: 5px; margin-bottom: 15px;">
                        <ol style="margin-left: 20px;">
                            <li style="margin-bottom: 10px;"><strong>Data Acquisition</strong>: Raw IMU data from Android sensors (100 Hz nominal)</li>
                            <li style="margin-bottom: 10px;"><strong>Resampling</strong>: Uniform time intervals via linear interpolation</li>
                            <li style="margin-bottom: 10px;"><strong>Noise Filtering</strong>: Moving average filter to remove high-frequency noise</li>
                            <li style="margin-bottom: 10px;"><strong>PCA</strong>: Principal Component Analysis to identify dominant movement axis</li>
                            <li style="margin-bottom: 10px;"><strong>Phase Detection</strong>: Automatic segmentation into movement phases (eccentric, concentric, holds)</li>
                            <li style="margin-bottom: 10px;"><strong>Repetition Detection</strong>: Identification of exercise repetitions with quality validation</li>
                            <li style="margin-bottom: 10px;"><strong>Temporal Analysis</strong>: 50 temporal parameters across 7 categories</li>
                            <li style="margin-bottom: 10px;"><strong>Quality Assessment</strong>: Multi-criteria validation (ROM, phases, cyclicity)</li>
                        </ol>
                    </div>
                    
                    <h3 style="color: #667eea; margin: 25px 0 15px 0;">📊 Key Metrics Explained</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px;">
                        <div style="background: #e7f3ff; padding: 15px; border-radius: 8px;">
                            <h4 style="color: #0066cc; margin-bottom: 10px;">🔄 Cycles vs Repetitions</h4>
                            <p style="font-size: 0.9em;">
                                <strong>Cycles</strong>: Mathematical detection of periodic patterns in the signal (peak-to-peak).<br>
                                <strong>Repetitions</strong>: Biomechanically valid exercise movements with proper phase sequences.
                                Not all cycles are valid repetitions!
                            </p>
                        </div>
                        
                        <div style="background: #fff3e0; padding: 15px; border-radius: 8px;">
                            <h4 style="color: #ff6f00; margin-bottom: 10px;">📏 Range of Motion (ROM)</h4>
                            <p style="font-size: 0.9em;">
                                Angular displacement between maximum and minimum angles during a repetition.
                                Higher ROM indicates deeper movement. Suspicious if ROM < 50% of median (possible partial rep or detection error).
                            </p>
                        </div>
                        
                        <div style="background: #f3e5f5; padding: 15px; border-radius: 8px;">
                            <h4 style="color: #6a1b9a; margin-bottom: 10px;">⏱️ Temporal Parameters</h4>
                            <p style="font-size: 0.9em;">
                                <strong>Eccentric</strong>: Muscle lengthening (descent in squat)<br>
                                <strong>Concentric</strong>: Muscle shortening (ascent in squat)<br>
                                <strong>Holds</strong>: Isometric phases (bottom/top pauses)<br>
                                <strong>Density Ratio</strong>: Work time / Total time (higher = more continuous)
                            </p>
                        </div>
                        
                        <div style="background: #e8f5e9; padding: 15px; border-radius: 8px;">
                            <h4 style="color: #2e7d32; margin-bottom: 10px;">✓ Validation Criteria</h4>
                            <p style="font-size: 0.9em;">
                                <strong>Phase Sequence</strong>: Must contain ascending AND descending phases<br>
                                <strong>Cyclicity</strong>: Must return to start angle within ±20° tolerance<br>
                                <strong>ROM Threshold</strong>: Must exceed 50% of median ROM
                            </p>
                        </div>
                    </div>
                    
                    <h3 style="color: #667eea; margin: 25px 0 15px 0;">📈 Trend Interpretation</h3>
                    <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; border-radius: 5px;">
                        <p style="margin-bottom: 10px;"><strong>Positive Trends (+)</strong>:</p>
                        <ul style="margin-left: 20px; margin-bottom: 15px;">
                            <li>Eccentric/Concentric: Progressive slowing → Fatigue accumulation</li>
                            <li>Holds: Increasing rest periods → Compensatory strategy</li>
                            <li>Rep Duration: Overall slowing → General fatigue</li>
                        </ul>
                        <p style="margin-bottom: 10px;"><strong>Negative Trends (-):</strong></p>
                        <ul style="margin-left: 20px;">
                            <li>Eccentric/Concentric: Progressive speeding → Learning effect or reduced control</li>
                            <li>Holds: Decreasing rest → Pacing strategy or motivation</li>
                            <li>ROM: Decreasing range → Fatigue or pain compensation</li>
                        </ul>
                    </div>
                    
                    <h3 style="color: #667eea; margin: 25px 0 15px 0;">🎯 How to Use This Report</h3>
                    <div style="background: #d1ecf1; padding: 15px; border-radius: 5px; border-left: 4px solid #17a2b8;">
                        <ol style="margin-left: 20px;">
                            <li style="margin-bottom: 8px;"><strong>Review Summary Cards</strong>: Get overall session metrics at a glance</li>
                            <li style="margin-bottom: 8px;"><strong>Check PCA Results</strong>: Identify which axis captures the main movement (typically pitch for squats)</li>
                            <li style="margin-bottom: 8px;"><strong>Examine Repetition Analysis</strong>: Focus on filtered signals (ang*_filtered) for best accuracy</li>
                            <li style="margin-bottom: 8px;"><strong>Validate Quality</strong>: Check suspicious/invalid reps and validation reasons</li>
                            <li style="margin-bottom: 8px;"><strong>Analyze Temporal Patterns</strong>: Look for fatigue indicators in trends and variances</li>
                            <li style="margin-bottom: 8px;"><strong>Compare Across Sessions</strong>: Use Excel export for quantitative comparisons</li>
                        </ol>
                    </div>
                    
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px;">
                        <h4 style="color: #1976d2; margin-bottom: 10px;">💡 Pro Tips</h4>
                        <ul style="margin-left: 20px; font-size: 0.9em;">
                            <li style="margin-bottom: 8px;">✅ <strong>Filtered signals</strong> are more reliable than original (noise-reduced)</li>
                            <li style="margin-bottom: 8px;">✅ <strong>ANG_PRINCIPAL</strong> (PC1) often provides the clearest movement representation</li>
                            <li style="margin-bottom: 8px;">✅ <strong>CV < 15%</strong> indicates excellent consistency</li>
                            <li style="margin-bottom: 8px;">✅ <strong>Density ratio > 70%</strong> indicates continuous execution (good for hypertrophy)</li>
                            <li style="margin-bottom: 8px;">✅ <strong>Excel export</strong> enables easy comparison across training sessions</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Sampling Quality -->
        <div class="section">
            <div class="section-header" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display=='none'?'block':'none'">
                <span>⏱️ Sampling Quality Diagnostics</span>
                <span>▼</span>
            </div>
            <div class="section-content">
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Mean Δt</div>
                        <div class="value">""" + f"{dt_mean_ms:.3f} ms" + """</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Std Dev Δt</div>
                        <div class="value">""" + f"{dt_std_ms:.3f} ms" + """</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Min Δt</div>
                        <div class="value">""" + f"{dt_min_ms:.3f} ms" + """</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Max Δt</div>
                        <div class="value">""" + f"{dt_max_ms:.3f} ms" + """</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Sampling Gaps (>3ms)</div>
                        <div class="value">""" + f"{num_gaps}" + """</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- PCA Analysis -->
        <div class="section">
            <div class="section-header" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display=='none'?'block':'none'">
                <span>🧬 Principal Component Analysis (PCA)</span>
                <span>▼</span>
            </div>
            <div class="section-content">
                <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #1976d2;">
                    <h4 style="color: #1976d2; margin-bottom: 10px;">📐 What is PCA?</h4>
                    <p style="margin-bottom: 10px; line-height: 1.6;">
                        <strong>Principal Component Analysis</strong> identifies the <em>dominant direction of movement</em> 
                        from the 3D Euler angles (ANG1, ANG2, ANG3). The <strong>first principal component (PC1)</strong> 
                        captures the axis with the most variance - typically the primary exercise movement.
                    </p>
                    <p style="margin-bottom: 10px; line-height: 1.6;">
                        <strong>Why it matters</strong>: For squats, PC1 usually corresponds to the pitch axis (forward/backward tilt), 
                        with >80% explained variance indicating a clean, unidirectional movement pattern.
                    </p>
                    <p style="line-height: 1.6;">
                        <strong>Loadings</strong>: Show contribution of each original axis to PC1. 
                        High absolute loading (>0.8) means that axis dominates the movement.
                    </p>
                </div>
                
                <h3>Explained Variance</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">PC1 (Primary Motion)</div>
                        <div class="value">""" + f"{explained[0]*100:.1f}%" + """</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">PC2</div>
                        <div class="value">""" + f"{explained[1]*100:.1f}%" + """</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">PC3</div>
                        <div class="value">""" + f"{explained[2]*100:.1f}%" + """</div>
                    </div>
                </div>
                <h3 style="margin-top: 20px;">PC1 Loadings (ang1, ang2, ang3)</h3>
                <p style="font-size: 1.2em; text-align: center; margin: 15px 0;">
                    <span class="badge badge-info">""" + f"{pca.components_[0][0]:+.3f}" + """</span>
                    <span class="badge badge-info">""" + f"{pca.components_[0][1]:+.3f}" + """</span>
                    <span class="badge badge-info">""" + f"{pca.components_[0][2]:+.3f}" + """</span>
                </p>
            </div>
        </div>
        
        <!-- Noise Filtering Summary -->
        <div class="section">
            <div class="section-header" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display=='none'?'block':'none'">
                <span>🔧 Noise Filtering Summary</span>
                <span>▼</span>
            </div>
            <div class="section-content">
                <p><strong>Filter Type:</strong> Moving Average (Convolution-based)</p>
                <p><strong>Window Size:</strong> """ + f"{window_size}" + """ samples (~""" + f"{window_size / (len(resampled_df) / resampled_df['ts_s'].iloc[-1]) * 1000:.1f}" + """ ms)</p>
                <p><strong>Purpose:</strong> Remove high-frequency noise while preserving biomechanical frequencies (0.1-5 Hz)</p>
                
                <h3 style="margin-top: 20px;">Noise Reduction Effectiveness</h3>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Axis</th>
                            <th>Original Noise (°)</th>
                            <th>Filtered Noise (°)</th>
                            <th>Reduction</th>
                        </tr>
                    </thead>
                    <tbody>
""")
    
    # Noise reduction table
    for axis in ['ang1', 'ang2', 'ang3']:
        noise_orig = np.std(np.diff(resampled_df[axis].to_numpy()))
        noise_filt = np.std(np.diff(resampled_df[f'{axis}_filtered'].to_numpy()))
        reduction = (1 - noise_filt / noise_orig) * 100
        f.write(f"""                        <tr>
                            <td class="metric-name">{axis.upper()}</td>
                            <td class="original">{noise_orig:.4f}</td>
                            <td class="filtered">{noise_filt:.4f}</td>
                            <td class="improvement">{reduction:.1f}%</td>
                        </tr>
""")
    
    f.write("""                    </tbody>
                </table>
            </div>
        </div>
        
""")
    
    # Statistics for each signal (original + filtered)
    all_signals = ["ang1", "ang2", "ang3", "ang_principal", 
                   "ang1_filtered", "ang2_filtered", "ang3_filtered", "ang_principal_filtered"]
    
    for angle_col in all_signals:
        is_filtered = '_filtered' in angle_col
        base_name = angle_col.replace('_filtered', '')
        display_name = angle_col.upper().replace('_', ' ')
        
        theta = resampled_df[angle_col].to_numpy()
        theta_detrended = detrend(theta)
        ts = resampled_df["ts_s"].to_numpy()
        
        # Compute all statistics
        min_val = theta.min()
        max_val = theta.max()
        range_val = max_val - min_val
        mean_val = theta.mean()
        median_val = np.median(theta)
        std_val = theta.std()
        skew_val = pd.Series(theta).skew()
        kurt_val = pd.Series(theta).kurtosis()
        min_idx = np.argmin(theta)
        max_idx = np.argmax(theta)
        
        dynamic_range = np.percentile(theta, 95) - np.percentile(theta, 5)
        noise_est = np.std(np.diff(theta))
        snr_like = dynamic_range / (noise_est + 1e-9)
        
        fs = len(ts) / (ts[-1] - ts[0])
        freqs, power = periodogram(theta_detrended, fs=fs)
        dominant_freq = freqs[np.argmax(power[1:])+1] if len(power) > 1 else 0
        period = 1/dominant_freq if dominant_freq > 0 else None
        
        adf_result = adfuller(theta)
        stationary = adf_result[1] < 0.05
        slope, intercept, r_value, p_value, std_err = linregress(ts, theta)
        
        peaks, _ = find_peaks(theta)
        troughs, _ = find_peaks(-theta)
        num_peaks = len(peaks)
        num_troughs = len(troughs)
        
        autocorr = acf(theta, nlags=100, fft=True)
        peak_lag = np.argmax(autocorr[1:]) + 1
        
        # Sliding window stats
        window_s = 2.0
        step_s = 0.5
        window_samples = int(window_s * fs)
        step_samples = int(step_s * fs)
        window_means = []
        window_stds = []
        for i in range(0, len(theta) - window_samples, step_samples):
            segment = theta[i:i+window_samples]
            window_means.append(np.mean(segment))
            window_stds.append(np.std(segment))
        std_window_means = np.std(window_means) if len(window_means) > 0 else 0
        std_window_stds = np.std(window_stds) if len(window_stds) > 0 else 0
        mean_window_std = np.mean(window_stds) if len(window_stds) > 0 else 0
        
        section_class = "filtered" if is_filtered else "original"
        badge_type = "badge-success" if is_filtered else "badge-info"
        
        f.write(f"""        <!-- {display_name} Statistics -->
        <div class="section">
            <div class="section-header {section_class}" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display=='none'?'block':'none'">
                <span>📊 {display_name} <span class="badge {badge_type}">{'✨ FILTERED' if is_filtered else '📋 ORIGINAL'}</span></span>
                <span>▼</span>
            </div>
            <div class="section-content">
                
                <!-- INTERACTIVE PLOTLY GRAPH EMBEDDED -->
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0 0 10px 0; display: flex; align-items: center;">
                        <span style="font-size: 1.5em; margin-right: 10px;">🔍</span>
                        Interactive Signal Exploration
                    </h4>
                    <p style="margin: 0 0 15px 0; line-height: 1.6; opacity: 0.95; font-size: 0.95em;">
                        Hover over the plot to see precise values • Zoom by dragging • Double-click to reset
                    </p>
                </div>
                <!-- Embed interactive plot using iframe -->
                <iframe src="interactive_{angle_col}_{file_name.replace('FILE_EULER', '').replace('.txt', '')}.html" width="100%" height="500" frameborder="0" style="border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 20px;"></iframe>
                
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Mean</div>
                        <div class="value">{mean_val:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Median</div>
                        <div class="value">{median_val:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Std Dev</div>
                        <div class="value">{std_val:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Range</div>
                        <div class="value">{range_val:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Min / Max</div>
                        <div class="value">{min_val:.2f}° / {max_val:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Skewness</div>
                        <div class="value">{skew_val:.2f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Kurtosis</div>
                        <div class="value">{kurt_val:.2f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">SNR-like Index</div>
                        <div class="value">{snr_like:.2f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Dynamic Range (5-95%)</div>
                        <div class="value">{dynamic_range:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Noise Estimate</div>
                        <div class="value">{noise_est:.4f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Stationary</div>
                        <div class="value">{'✓ Yes' if stationary else '✗ No'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Linear Trend</div>
                        <div class="value">{slope:.4f} °/s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Peaks / Troughs</div>
                        <div class="value">{num_peaks} / {num_troughs}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Dominant Frequency</div>
                        <div class="value">{dominant_freq:.4f} Hz</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Period</div>
                        <div class="value">{period:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">ACF Periodicity</div>
                        <div class="value">{peak_lag} samples</div>
                    </div>
                </div>
                
                <h3 style="margin-top: 25px;">Sliding Window Analysis (2.0s window, 0.5s step)</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Std of Window Means</div>
                        <div class="value">{std_window_means:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Std of Window Stds</div>
                        <div class="value">{std_window_stds:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Mean Window Std</div>
                        <div class="value">{mean_window_std:.2f}°</div>
                    </div>
                </div>
""")
        
        # Cycle and phase statistics
        if angle_col in cycle_stats:
            cs = cycle_stats[angle_col]
            f.write(f"""                
                <h3 style="margin-top: 25px;">🔄 Cycle-Level Metrics</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Total Cycles</div>
                        <div class="value">{cs['num_cycles']}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Mean Cycle Duration</div>
                        <div class="value">{cs['mean_cycle_duration']:.3f} s (CV={cs['cv_cycle_duration']:.1%})</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Mean Cycle Amplitude</div>
                        <div class="value">{cs['mean_cycle_amplitude']:.2f}° (CV={cs['cv_cycle_amplitude']:.1%})</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Trend Duration/Cycle</div>
                        <div class="value">{cs['trend_duration_per_cycle']:+.4f} s/cycle</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Trend Amplitude/Cycle</div>
                        <div class="value">{cs['trend_amplitude_per_cycle']:+.4f} °/cycle</div>
                    </div>
                </div>
""")
        
        if angle_col in phase_stats:
            ps = phase_stats[angle_col]['statistics']
            f.write(f"""                
                <h3 style="margin-top: 25px;">📍 Phase Statistics</h3>
                
                <div style="background: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #ff6f00;">
                    <h4 style="color: #ff6f00; margin-bottom: 10px;">🔄 Understanding Movement Phases</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; font-size: 0.9em;">
                        <div>
                            <strong>🔼 Ascending</strong> (Concentric)<br>
                            <span style="opacity: 0.8;">Upward motion, muscle shortening. E.g., rising from squat bottom.</span>
                        </div>
                        <div>
                            <strong>🔽 Descending</strong> (Eccentric)<br>
                            <span style="opacity: 0.8;">Downward motion, muscle lengthening. E.g., lowering into squat.</span>
                        </div>
                        <div>
                            <strong>🟧 Stable High</strong> (Top Hold)<br>
                            <span style="opacity: 0.8;">Isometric hold in standing position between reps.</span>
                        </div>
                        <div>
                            <strong>🟦 Stable Low</strong> (Bottom Hold)<br>
                            <span style="opacity: 0.8;">Isometric hold at bottom position (pause squats).</span>
                        </div>
                    </div>
                    <p style="margin-top: 10px; font-size: 0.85em; opacity: 0.9;">
                        <strong>CV (Coefficient of Variation)</strong>: Measures consistency. Lower values = more consistent timing across phases.
                    </p>
                </div>
                
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Phase Type</th>
                            <th>Count</th>
                            <th>Mean Duration</th>
                            <th>CV</th>
                            <th>Total Time</th>
                        </tr>
                    </thead>
                    <tbody>
""")
            phase_labels = {
                'ascending': ('🔼', 'Ascending'),
                'descending': ('🔽', 'Descending'),
                'stable_high': ('🟧', 'Stable High (standing)'),
                'stable_low': ('🟦', 'Stable Low (bottom)'),
                'stable': ('➡️', 'Stable')
            }
            
            for phase_type in ['ascending', 'descending', 'stable_high', 'stable_low', 'stable']:
                pstats = ps.get(phase_type, {'count': 0})
                if pstats['count'] > 0:
                    emoji, label = phase_labels[phase_type]
                    f.write(f"""                        <tr>
                            <td class="metric-name">{emoji} {label}</td>
                            <td>{pstats['count']}</td>
                            <td>{pstats['mean_duration']:.3f} s</td>
                            <td>{pstats['cv_duration']:.1%}</td>
                            <td>{pstats['total_time']:.1f} s</td>
                        </tr>
""")
            f.write("""                    </tbody>
                </table>
""")
        
        # === EXERCISE REPETITION ANALYSIS SECTION ===
        if angle_col in repetition_stats:
            rs = repetition_stats[angle_col]
            num_reps = rs['num_repetitions']
            
            f.write(f"""                
                <h3 style="margin-top: 30px;">🏋️ Exercise Repetition Analysis</h3>
                
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h4 style="margin: 0 0 15px 0; font-size: 1.2em;">📊 Summary</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div>
                            <div style="font-size: 0.9em; opacity: 0.9;">Total Repetitions</div>
                            <div style="font-size: 2em; font-weight: bold;">{num_reps}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9em; opacity: 0.9;">Detection Threshold</div>
                            <div style="font-size: 1.5em; font-weight: bold;">{rs['prominence_threshold']:.2f}°</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9em; opacity: 0.9;">Consistency Score</div>
                            <div style="font-size: 1.5em; font-weight: bold;">{rs['quality']['consistency_score']:.1f}/100</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9em; opacity: 0.9;">⚠️ Suspicious Reps</div>
                            <div style="font-size: 1.5em; font-weight: bold;">{rs.get('num_suspicious', 0)}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9em; opacity: 0.9;">📏 Median ROM</div>
                            <div style="font-size: 1.5em; font-weight: bold;">{rs.get('median_rom', 0):.2f}°</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9em; opacity: 0.9;">🎯 ROM Threshold (50%)</div>
                            <div style="font-size: 1.5em; font-weight: bold;">{rs.get('median_rom', 0)*0.5:.2f}°</div>
                        </div>
                    </div>
                </div>
""")
            
            # Add comparison table between base and advanced analysis
            if f"{angle_col}_advanced" in repetition_stats:
                rs_adv = repetition_stats[f"{angle_col}_advanced"]
                num_reps_adv = rs_adv['num_valid_repetitions']
                
                f.write(f"""
                <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #2196f3;">
                    <h4 style="color: #1976d2; margin-bottom: 15px;">📊 Comparison: Base Analysis vs Advanced Filtering</h4>
                    <table class="comparison-table" style="margin-bottom: 15px;">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Base Analysis</th>
                                <th>Advanced Analysis</th>
                                <th>Difference</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="metric-name">Total Repetitions Detected</td>
                                <td><strong>{num_reps}</strong></td>
                                <td><strong>{num_reps_adv}</strong></td>
                                <td><span class="badge {'badge-warning' if num_reps_adv < num_reps else 'badge-success'}">{num_reps_adv - num_reps:+d} ({(num_reps_adv/num_reps*100 if num_reps > 0 else 0):.1f}%)</span></td>
                            </tr>
                            <tr>
                                <td class="metric-name">Filtering Applied</td>
                                <td><span class="badge badge-info">Prominence + Phase Detection</span></td>
                                <td><span class="badge badge-success">ROM Threshold + Outliers + First/Last</span></td>
                                <td>-</td>
                            </tr>
                            <tr>
                                <td class="metric-name">Mean ROM</td>
                                <td>{np.mean(rs['rep_ranges']) if rs['rep_ranges'] else 0:.2f}°</td>
                                <td>{np.mean(rs_adv['rep_ranges']) if len(rs_adv['rep_ranges']) > 0 else 0:.2f}°</td>
                                <td>{(np.mean(rs_adv['rep_ranges']) if len(rs_adv['rep_ranges']) > 0 else 0) - (np.mean(rs['rep_ranges']) if rs['rep_ranges'] else 0):+.2f}°</td>
                            </tr>
                            <tr>
                                <td class="metric-name">ROM CV (Consistency)</td>
                                <td>{rs['quality']['range_cv']:.1%}</td>""")
                
                cv_adv_text = f"{rs_adv['raw_cvs'].get('range_cv', 0):.1%}" if len(rs_adv['raw_cvs']) > 0 else "N/A"
                cv_diff_text = f"{rs_adv['raw_cvs']['range_cv'] - rs['quality']['range_cv']:+.1%}" if len(rs_adv['raw_cvs']) > 0 else "N/A"
                f.write(f"""
                                <td>{cv_adv_text}</td>
                                <td>{cv_diff_text}</td>
                            </tr>
                            <tr>
                                <td class="metric-name">Suspicious Reps</td>
                                <td><span class="badge badge-warning">{rs.get('num_suspicious', 0)} reps</span></td>
                                <td><span class="badge badge-success">Filtered out</span></td>
                                <td>-{rs.get('num_suspicious', 0)} cleaned</td>
                            </tr>
                        </tbody>
                    </table>
                    <p style="margin: 10px 0 0 0; color: #424242; font-size: 0.9em;">
                        <strong>💡 Interpretation:</strong> Advanced analysis applies stricter quality filters to remove half-reps, 
                        statistical outliers, and adjustment phases (first/last reps). This typically results in fewer but 
                        more biomechanically valid repetitions with better consistency metrics.
                    </p>
                </div>
""")
            
            if num_reps > 0:
                # Add validation criteria explanation box
                f.write("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin: 25px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0 0 15px 0; display: flex; align-items: center;">
                        <span style="font-size: 1.5em; margin-right: 10px;">🔍</span>
                        Criteri di Validazione delle Ripetizioni
                    </h4>
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <h5 style="margin: 0 0 10px 0; font-size: 1.1em;">📊 1. Range of Motion (ROM)</h5>
                        <p style="margin: 5px 0; line-height: 1.6;">
                            Una ripetizione è considerata <strong>sospetta per ROM</strong> se:<br>
                            • ROM &lt; 50% della mediana di tutte le ripetizioni<br>
                            • Esempio: Se mediana ROM = 22.87°, la soglia è 11.4°<br>
                            • <em>Motivo</em>: Ripetizioni troppo piccole potrebbero essere artefatti o movimenti incompleti
                        </p>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <h5 style="margin: 0 0 10px 0; font-size: 1.1em;">🔄 2. Sequenza di Fasi</h5>
                        <p style="margin: 5px 0; line-height: 1.6;">
                            Una ripetizione <strong>valida</strong> deve contenere:<br>
                            • Almeno una fase di <strong style="color: #4ade80;">ASCESA</strong> (ascending)<br>
                            • Almeno una fase di <strong style="color: #f87171;">DISCESA</strong> (descending)<br>
                            • <em>Motivo</em>: Un esercizio completo richiede movimento bidirezionale (es: squat = scendi + sali)
                        </p>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                        <h5 style="margin: 0 0 10px 0; font-size: 1.1em;">🔁 3. Ciclicità (Ritorno all'Origine)</h5>
                        <p style="margin: 5px 0; line-height: 1.6;">
                            Una ripetizione è <strong>ciclica</strong> se:<br>
                            • Il <strong>punto più alto DOPO</strong> il minimo ritorna entro <strong>±20°</strong> dal <strong>punto più alto PRIMA</strong> del minimo<br>
                            • Calcolo: <code style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">Δangle = |max_dopo - max_prima|</code><br>
                            • Esempio: Durante uno squat, se parti da -115° (in piedi), scendi a -136° (giù), e risali a -125° (in piedi) → Δ = 10° ✓<br>
                            • <strong>Nota importante</strong>: Si confrontano le posizioni erette (massimi), non i punti intermedi tra ripetizioni<br>
                            • <em>Motivo</em>: Tolleranza di ±20° per tenere conto di drift naturale, fatica e variabilità umana nell'esecuzione
                        </p>
                    </div>
                    <div style="margin-top: 15px; padding: 12px; background: rgba(255,255,255,0.15); border-left: 4px solid #fbbf24; border-radius: 4px;">
                        <strong>⚠️ Nota:</strong> Una ripetizione può essere marcata "suspicious" per uno o più criteri. 
                        I motivi specifici sono indicati nella colonna "Quality".
                    </div>
                </div>
                
                <h4 style="margin-top: 25px;">📋 Per-Repetition Metrics</h4>
                <div style="overflow-x: auto;">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Rep #</th>
                            <th>Time (s)</th>
                            <th>Duration (s)</th>
                            <th>Range (°)</th>
                            <th>Depth (°)</th>
                            <th>Quality</th>
                        </tr>
                    </thead>
                    <tbody>
""")
                
                # Calculate quality indicators for each rep
                rep_ranges = rs['rep_ranges']
                rep_durations = rs['rep_durations']
                mean_range = sum(rep_ranges) / len(rep_ranges) if rep_ranges else 0
                mean_duration = sum(rep_durations) / len(rep_durations) if rep_durations else 0
                median_rom = rs.get('median_rom', mean_range)
                suspicious_flags = rs.get('suspicious_reps', np.zeros(num_reps, dtype=bool))
                invalid_reps = rs.get('invalid_reps', [])
                
                # Build lookup for invalid reps reasons
                invalid_reasons = {}
                for inv_rep in invalid_reps:
                    rep_num = inv_rep.get('rep_num', -1)
                    if rep_num > 0:
                        invalid_reasons[rep_num - 1] = inv_rep.get('reason', [])  # Convert to 0-indexed
                
                for i in range(num_reps):
                    rep_time = rs['repetition_times'][i]
                    rep_dur = rs['rep_durations'][i]
                    rep_range = rs['rep_ranges'][i]
                    rep_depth = rs['rep_depths'][i]
                    is_suspicious = suspicious_flags[i]
                    
                    # Quality indicator: check if within ±20% of mean
                    range_quality = "✓" if abs(rep_range - mean_range) / (mean_range + 1e-9) < 0.2 else "⚠️" if abs(rep_range - mean_range) / (mean_range + 1e-9) < 0.4 else "❌"
                    duration_quality = "✓" if abs(rep_dur - mean_duration) / (mean_duration + 1e-9) < 0.2 else "⚠️"
                    
                    # Override quality if suspicious - show actual reason
                    if is_suspicious:
                        quality_badge = "badge-danger"
                        # Check if this rep has phase validation issues
                        if i in invalid_reasons and invalid_reasons[i]:
                            # Show phase validation failure reason
                            reasons = invalid_reasons[i]
                            quality_text = f"⚠️ {'; '.join(reasons)}"
                        else:
                            # Must be ROM-based suspicious
                            quality_text = f"⚠️ Suspicious (ROM<50% median)"
                    else:
                        quality_badge = "badge-success" if range_quality == "✓" and duration_quality == "✓" else "badge-warning" if "⚠️" in [range_quality, duration_quality] else "badge-danger"
                        quality_text = "Good" if range_quality == "✓" and duration_quality == "✓" else "Fair" if "⚠️" in [range_quality, duration_quality] else "Poor"
                    
                    f.write(f"""                        <tr{' style="background-color: #fff3cd;"' if is_suspicious else ''}>
                            <td class="metric-name">{i+1}</td>
                            <td>{rep_time:.2f}</td>
                            <td>{rep_dur:.2f}</td>
                            <td>{rep_range:.2f}</td>
                            <td>{rep_depth:.2f}</td>
                            <td><span class="badge {quality_badge}">{quality_text}</span></td>
                        </tr>
""")
                
                f.write("""                    </tbody>
                </table>
                </div>
""")
                
                # Statistics grid
                intervals = rs['inter_rep_intervals']
                if intervals:
                    import statistics
                    mean_interval = sum(intervals) / len(intervals)
                    std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
                    min_interval = min(intervals)
                    max_interval = max(intervals)
                    cv_interval = (std_interval / mean_interval * 100) if mean_interval > 0 else 0
                    
                    f.write(f"""
                <h4 style="margin-top: 25px;">⏱️ Timing Analysis</h4>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Mean Interval</div>
                        <div class="value">{mean_interval:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Std Deviation</div>
                        <div class="value">{std_interval:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Min/Max Interval</div>
                        <div class="value">{min_interval:.2f} / {max_interval:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Timing CV</div>
                        <div class="value">{cv_interval:.1f}%</div>
                    </div>
                </div>
""")
                
                # Range analysis
                mean_range_val = sum(rep_ranges) / len(rep_ranges) if rep_ranges else 0
                std_range = statistics.stdev(rep_ranges) if len(rep_ranges) > 1 else 0
                min_range = min(rep_ranges) if rep_ranges else 0
                max_range = max(rep_ranges) if rep_ranges else 0
                
                f.write(f"""
                <h4 style="margin-top: 25px;">📏 Range of Motion Analysis</h4>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Mean Range</div>
                        <div class="value">{mean_range_val:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Std Deviation</div>
                        <div class="value">{std_range:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Min/Max Range</div>
                        <div class="value">{min_range:.2f} / {max_range:.2f}°</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Range Consistency (CV)</div>
                        <div class="value">{rs['quality']['range_cv']*100:.1f}%</div>
                    </div>
                </div>
""")
                
                # Quality metrics with visual indicators
                consistency_score = rs['quality']['consistency_score']
                fatigue = rs['quality']['fatigue_indicator']
                
                fatigue_status = "INCREASING RANGE" if fatigue > 0.5 else "DECREASING RANGE (Fatigue)" if fatigue < -0.5 else "STABLE"
                fatigue_color = "#28a745" if abs(fatigue) < 0.5 else "#ffc107" if fatigue > 0.5 else "#dc3545"
                
                f.write(f"""
                <h4 style="margin-top: 25px;">⭐ Performance Quality</h4>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Overall Consistency</div>
                        <div class="value">{consistency_score:.1f}/100</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Range Consistency</div>
                        <div class="value">{rs['quality']['range_cv']*100:.1f}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Timing Consistency</div>
                        <div class="value">{rs['quality']['timing_cv']*100:.1f}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Tempo Consistency</div>
                        <div class="value">{rs['quality']['tempo_cv']*100:.1f}%</div>
                    </div>
                </div>
                
                <div style="background: {fatigue_color}; color: white; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <h5 style="margin: 0 0 10px 0;">🎯 Fatigue Indicator</h5>
                    <p style="margin: 0; font-size: 1.1em;"><strong>{fatigue_status}</strong></p>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">{fatigue:+.3f}°/rep</p>
                </div>
                
                <h4 style="margin-top: 25px;">🎯 Derived Metrics</h4>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Average Rep Frequency</div>
                        <div class="value">{(num_reps-1)/(rs['repetition_times'][-1] - rs['repetition_times'][0]) if num_reps > 1 else 0:.4f} Hz</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Seconds per Rep</div>
                        <div class="value">{(rs['repetition_times'][-1] - rs['repetition_times'][0])/(num_reps-1) if num_reps > 1 else 0:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Total Exercise Duration</div>
                        <div class="value">{rs['repetition_times'][-1] - rs['repetition_times'][0] if num_reps > 1 else 0:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Work Density</div>
                        <div class="value">{num_reps/(rs['repetition_times'][-1] - rs['repetition_times'][0])*60 if num_reps > 1 else 0:.1f} reps/min</div>
                    </div>
                </div>
""")
                
                # === DETAILED TEMPORAL PARAMETERS SECTION ===
                if 'temporal_parameters' in rs:
                    tp = rs['temporal_parameters']
                    
                    f.write(f"""
                <h4 style="margin-top: 30px;">⏱️ Detailed Temporal Parameters (50 Parameters)</h4>
                
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 15px 0;">
                    <h4 style="margin: 0 0 15px 0; font-size: 1.2em;">📊 Complete Temporal Analysis</h4>
                    <p style="margin: 0 0 15px 0; line-height: 1.6;">
                        This section presents a <strong>comprehensive temporal analysis</strong> of your exercise execution,
                        breaking down each repetition into its constituent phases and analyzing timing patterns, 
                        consistency, and trends throughout the session.
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; font-size: 0.9em;">
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                            <strong>📋 Per-Repetition</strong><br>
                            8 params × {len(tp['per_repetition'])} reps<br>
                            <span style="opacity: 0.9;">Phase timings + percentages</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                            <strong>🌍 Global Timing</strong><br>
                            4 parameters<br>
                            <span style="opacity: 0.9;">Frequency, density, cycles</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                            <strong>📈 Session Totals</strong><br>
                            6 parameters<br>
                            <span style="opacity: 0.9;">Cumulative work & rest</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                            <strong>📊 Variances</strong><br>
                            16 parameters<br>
                            <span style="opacity: 0.9;">Consistency metrics (CV)</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                            <strong>🔄 Regularity</strong><br>
                            6 parameters<br>
                            <span style="opacity: 0.9;">Duration stability</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                            <strong>⚠️ Outliers</strong><br>
                            4 parameters<br>
                            <span style="opacity: 0.9;">Unusual timing patterns</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                            <strong>📈 Trends</strong><br>
                            6 parameters<br>
                            <span style="opacity: 0.9;">Fatigue indicators</span>
                        </div>
                    </div>
                </div>
                
                <!-- B. Global Timing Parameters -->
                <h5 style="margin-top: 20px; color: #667eea;">📊 Global Timing Parameters</h5>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;">
                    <strong>Cycle Time</strong>: Time from start of one rep to start of next (includes work + rest). 
                    <strong>Density Ratio</strong>: Proportion of time spent in active work (ecc+conc) vs total time.
                    Higher density = more continuous training stimulus.
                </div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Mean Rep Duration</div>
                        <div class="value">{tp['global_timing']['mean_rep_duration']:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Mean Cycle Time</div>
                        <div class="value">{tp['global_timing']['mean_cycle_time']:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Execution Frequency</div>
                        <div class="value">{tp['global_timing']['execution_frequency_hz']:.3f} Hz</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Mean Density Ratio</div>
                        <div class="value">{tp['global_timing']['mean_density_ratio']:.1%}</div>
                    </div>
                </div>
                
                <!-- C. Session Total Times -->
                <h5 style="margin-top: 20px; color: #667eea;">📈 Session Total Times</h5>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Total Eccentric Time</div>
                        <div class="value">{tp['session_totals']['total_eccentric_time']:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Total Concentric Time</div>
                        <div class="value">{tp['session_totals']['total_concentric_time']:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Total Bottom Hold</div>
                        <div class="value">{tp['session_totals']['total_bottom_hold_time']:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Total Top Hold</div>
                        <div class="value">{tp['session_totals']['total_top_hold_time']:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Total Work Time</div>
                        <div class="value">{tp['session_totals']['total_work_time']:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Total Pause Time</div>
                        <div class="value">{tp['session_totals']['total_pause_time']:.2f} s</div>
                    </div>
                </div>
                
                <!-- D. Temporal Variances -->
                <h5 style="margin-top: 20px; color: #667eea;">📊 Temporal Variances (Coefficient of Variation)</h5>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Eccentric CV</div>
                        <div class="value">{tp['temporal_variances']['eccentric']['cv']:.1%}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Concentric CV</div>
                        <div class="value">{tp['temporal_variances']['concentric']['cv']:.1%}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Bottom Hold CV</div>
                        <div class="value">{tp['temporal_variances']['bottom_hold']['cv']:.1%}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Top Hold CV</div>
                        <div class="value">{tp['temporal_variances']['top_hold']['cv']:.1%}</div>
                    </div>
                </div>
                
                <!-- E. Regularity & Continuity -->
                <h5 style="margin-top: 20px; color: #667eea;">🔄 Regularity & Continuity</h5>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Rep Duration CV</div>
                        <div class="value">{tp['regularity']['rep_duration_cv']:.1%}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Rep Duration Range</div>
                        <div class="value">{tp['regularity']['rep_duration_range']:.2f} s</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Cycle Time CV</div>
                        <div class="value">{tp['regularity']['cycle_time_cv']:.1%}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Cycle Time Range</div>
                        <div class="value">{tp['regularity']['cycle_time_range']:.2f} s</div>
                    </div>
                </div>
                
                <!-- F. Temporal Outliers -->
                <h5 style="margin-top: 20px; color: #667eea;">⚠️ Temporal Outliers</h5>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Slow Eccentric Reps</div>
                        <div class="value">{tp['outliers']['num_slow_eccentric']}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Fast Concentric Reps</div>
                        <div class="value">{tp['outliers']['num_fast_concentric']}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Excessive Bottom Holds</div>
                        <div class="value">{tp['outliers']['num_excessive_bottom_hold']}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Excessive Top Holds</div>
                        <div class="value">{tp['outliers']['num_excessive_top_hold']}</div>
                    </div>
                </div>
                
                <!-- G. Longitudinal Trends -->
                <h5 style="margin-top: 20px; color: #667eea;">📈 Longitudinal Trends (s/rep)</h5>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">Eccentric Trend</div>
                        <div class="value" style="color: {'#28a745' if tp['longitudinal_trends']['eccentric_trend'] >= 0 else '#dc3545'}">{tp['longitudinal_trends']['eccentric_trend']:+.4f} s/rep</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Concentric Trend</div>
                        <div class="value" style="color: {'#28a745' if tp['longitudinal_trends']['concentric_trend'] >= 0 else '#dc3545'}">{tp['longitudinal_trends']['concentric_trend']:+.4f} s/rep</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Bottom Hold Trend</div>
                        <div class="value" style="color: {'#28a745' if tp['longitudinal_trends']['bottom_hold_trend'] >= 0 else '#dc3545'}">{tp['longitudinal_trends']['bottom_hold_trend']:+.4f} s/rep</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Top Hold Trend</div>
                        <div class="value" style="color: {'#28a745' if tp['longitudinal_trends']['top_hold_trend'] >= 0 else '#dc3545'}">{tp['longitudinal_trends']['top_hold_trend']:+.4f} s/rep</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Rep Duration Trend</div>
                        <div class="value" style="color: {'#28a745' if tp['longitudinal_trends']['rep_duration_trend'] >= 0 else '#dc3545'}">{tp['longitudinal_trends']['rep_duration_trend']:+.4f} s/rep</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Cycle Time Trend</div>
                        <div class="value" style="color: {'#28a745' if tp['longitudinal_trends']['cycle_time_trend'] >= 0 else '#dc3545'}">{tp['longitudinal_trends']['cycle_time_trend']:+.4f} s/rep</div>
                    </div>
                </div>
                
                <!-- TEMPORAL EVOLUTION PLOTS -->
                <h5 style="margin-top: 25px; color: #667eea;">📈 Temporal Evolution Across Repetitions</h5>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <p style="margin: 0 0 10px 0; color: #424242; line-height: 1.6;">
                        <strong>Visual Analysis:</strong> The following plots show how timing metrics evolve throughout your exercise session.
                        Orange dashed lines indicate trends - positive slopes suggest fatigue (slowing down), negative slopes indicate adaptation or pacing strategy.
                    </p>
                    <div style="text-align: center;">
                        <img src="temporal_evolution_{angle_col}.png" alt="Temporal Evolution" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    </div>
                </div>
                
                <!-- A. Per-Repetition Details Table -->
                <h5 style="margin-top: 25px; color: #667eea;">📋 Per-Repetition Timing Breakdown</h5>
                <div style="overflow-x: auto;">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Rep #</th>
                            <th>Eccentric (s)</th>
                            <th>Concentric (s)</th>
                            <th>Bottom Hold (s)</th>
                            <th>Top Hold (s)</th>
                            <th>Total (s)</th>
                            <th>Ecc %</th>
                            <th>Con %</th>
                        </tr>
                    </thead>
                    <tbody>
""")
                    
                    for rep in tp['per_repetition']:
                        f.write(f"""                        <tr>
                            <td class="metric-name">{rep['rep_num']}</td>
                            <td>{rep['eccentric_time']:.3f}</td>
                            <td>{rep['concentric_time']:.3f}</td>
                            <td>{rep['bottom_hold_time']:.3f}</td>
                            <td>{rep['top_hold_time']:.3f}</td>
                            <td><strong>{rep['total_time']:.3f}</strong></td>
                            <td>{rep['eccentric_pct']:.1f}%</td>
                            <td>{rep['concentric_pct']:.1f}%</td>
                        </tr>
""")
                    
                    f.write("""                    </tbody>
                </table>
                </div>
""")
                
                # === PHASE VALIDATION DETAILS ===
                if rs.get('has_validation', False) and rs.get('num_invalid', 0) > 0:
                    invalid_reps = rs.get('invalid_reps', [])
                    f.write(f"""
                <h4 style="margin-top: 30px; color: #dc3545;">⚠️ Phase Validation Results</h4>
                <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; border-radius: 5px; margin: 15px 0;">
                    <p style="margin: 0 0 10px 0;"><strong>⚠️ {rs['num_invalid']} repetitions failed phase sequence validation</strong></p>
                    <p style="margin: 0; font-size: 0.9em; opacity: 0.8;">These repetitions lack proper ascending+descending sequence or cyclic behavior (return to start angle).</p>
                </div>
                
                <div style="overflow-x: auto; margin-top: 15px;">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Rep #</th>
                            <th>Failure Reason</th>
                            <th>Phase Sequence</th>
                            <th>Start Angle (°)</th>
                            <th>End Angle (°)</th>
                            <th>Δ Angle (°)</th>
                        </tr>
                    </thead>
                    <tbody>
""")
                    
                    # Get validation_results for detailed angle info
                    validation_results = rs.get('validation_results', [])
                    
                    for inv_rep in invalid_reps[:10]:  # Show first 10 invalid reps
                        rep_num = inv_rep.get('rep_num', '?')
                        rep_idx = rep_num - 1 if isinstance(rep_num, int) else 0
                        reasons = inv_rep.get('reason', [])
                        reason_text = '<br>'.join(reasons) if reasons else 'Unknown'
                        phase_seq = inv_rep.get('phase_sequence', [])
                        phase_text = ' → '.join(phase_seq) if phase_seq else 'N/A'
                        angle_recovery = inv_rep.get('angle_recovery', 0)
                        
                        # Get detailed angles from validation_results
                        start_angle = '?'
                        end_angle = '?'
                        if rep_idx < len(validation_results):
                            val_result = validation_results[rep_idx]
                            start_angle = f"{val_result.get('start_angle', '?'):.1f}"
                            end_angle = f"{val_result.get('end_angle', '?'):.1f}"
                        
                        f.write(f"""                        <tr style="background-color: #fff3cd;">
                            <td class="metric-name">#{rep_num}</td>
                            <td style="font-size: 0.85em;">{reason_text}</td>
                            <td style="font-size: 0.85em;">{phase_text}</td>
                            <td>{start_angle}</td>
                            <td>{end_angle}</td>
                            <td><strong>{angle_recovery:.1f}°</strong></td>
                        </tr>
""")
                    
                    f.write("""                    </tbody>
                </table>
                </div>
                
                <div style="background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h5 style="margin: 0 0 10px 0;">ℹ️ Validation Criteria</h5>
                    <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                        <li><strong>Phase Sequence:</strong> Must contain BOTH ascending AND descending phases (stable phases allowed)</li>
                        <li><strong>Cyclic Behavior:</strong> Signal must return to starting angle within ±20° tolerance</li>
                        <li><strong>Standing Recovery:</strong> Validates proper exercise execution and position recovery (±20° accounts for natural drift and fatigue)</li>
                    </ul>
                </div>
""")
        
        # === ADVANCED REPETITION ANALYSIS SECTION ===
        # Apply to ALL signals (both original and filtered)
        if f"{angle_col}_advanced" in repetition_stats:
            rs_adv = repetition_stats[f"{angle_col}_advanced"]
            num_reps_adv = rs_adv['num_valid_repetitions']
            filter_stats = rs_adv['filtering_applied']
            quality = rs_adv['quality_indices']
            fatigue_adv = rs_adv['fatigue_analysis']
            
            f.write(f"""
                <div style="margin-top: 50px; padding: 30px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0 0 20px 0; color: white; font-size: 1.5em;">🚀 ADVANCED REPETITION ANALYSIS (Production-Ready)</h3>
                    <div style="background: rgba(255,255,255,0.95); padding: 25px; border-radius: 10px;">
                        
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
                            <h4 style="margin: 0 0 15px 0;">🎯 Key Improvements</h4>
                            <ul style="margin: 0; padding-left: 20px;">
                                <li><strong>Butterworth Filter:</strong> Zero-phase low-pass (10Hz cutoff) preserves signal shape</li>
                                <li><strong>ROM Threshold:</strong> Excludes half-reps and invalid movements</li>
                                <li><strong>Outlier Detection:</strong> IQR method removes statistical anomalies</li>
                                <li><strong>First/Last Exclusion:</strong> Removes adjustment and overshoot phases</li>
                                <li><strong>R² Validation:</strong> Fatigue trend trusted only if R² > 0.5</li>
                            </ul>
                        </div>
                        
                        <h4>📋 Filtering Statistics</h4>
                        <table class="comparison-table" style="margin-bottom: 25px;">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Value</th>
                                    <th>Impact</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td class="metric-name">Total Detected (Before Filtering)</td>
                                    <td>{filter_stats['total_detected_before_filtering']}</td>
                                    <td><span class="badge badge-info">Baseline</span></td>
                                </tr>
                                <tr>
                                    <td class="metric-name">ROM Threshold Applied</td>
                                    <td>{filter_stats['min_rom_threshold']:.1f}°</td>
                                    <td>{'<span class="badge badge-warning">'+str(filter_stats.get('reps_below_rom_threshold', 0))+' removed</span>' if 'reps_below_rom_threshold' in filter_stats else '<span class="badge badge-success">None</span>'}</td>
                                </tr>
                                <tr>
                                    <td class="metric-name">First/Last Reps Removed</td>
                                    <td>{'Yes' if filter_stats['first_last_removed'] else 'No'}</td>
                                    <td><span class="badge badge-warning">2 removed</span></td>
                                </tr>
                                <tr>
                                    <td class="metric-name">Statistical Outliers Removed</td>
                                    <td>{filter_stats['outliers_removed']}</td>
                                    <td>{'<span class="badge badge-warning">IQR method</span>' if filter_stats['outliers_removed'] > 0 else '<span class="badge badge-success">Clean</span>'}</td>
                                </tr>
                                <tr style="background: #e8f5e9;">
                                    <td class="metric-name"><strong>Valid Reps After Filtering</strong></td>
                                    <td><strong>{num_reps_adv}</strong></td>
                                    <td><span class="badge badge-success">{((num_reps_adv/filter_stats['total_detected_before_filtering'])*100 if filter_stats['total_detected_before_filtering'] > 0 else 0):.1f}% retained</span></td>
                                </tr>
""")
            
            if 'iqr_bounds' in filter_stats:
                f.write(f"""                                <tr>
                                    <td class="metric-name">IQR Bounds</td>
                                    <td colspan="2">[{filter_stats['iqr_bounds']['lower']:.1f}°, {filter_stats['iqr_bounds']['upper']:.1f}°]</td>
                                </tr>
""")
            
            f.write("""                            </tbody>
                        </table>
""")
            
            if num_reps_adv > 0:
                # Advanced Quality Indices
                igm_color = "#28a745" if quality['igm_score'] >= 80 else "#ffc107" if quality['igm_score'] >= 60 else "#fd7e14" if quality['igm_score'] >= 40 else "#dc3545"
                
                f.write(f"""
                        <h4>🎯 Advanced Quality Indices (0-100 scale)</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 25px;">
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">ROM Consistency</div>
                                <div style="font-size: 2.5em; font-weight: bold;">{quality['rom_consistency_index']:.1f}</div>
                                <div style="font-size: 0.8em; opacity: 0.8; margin-top: 5px;">Lower CV = Better</div>
                            </div>
                            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Tempo Consistency</div>
                                <div style="font-size: 2.5em; font-weight: bold;">{quality['tempo_consistency_index']:.1f}</div>
                                <div style="font-size: 0.8em; opacity: 0.8; margin-top: 5px;">Execution Rhythm</div>
                            </div>
                            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 5px;">Depth Index</div>
                                <div style="font-size: 2.5em; font-weight: bold;">{quality['depth_index']:.1f}</div>
                                <div style="font-size: 0.8em; opacity: 0.8; margin-top: 5px;">Target Achievement</div>
                            </div>
                        </div>
                        
                        <div style="background: {igm_color}; color: white; padding: 25px; border-radius: 10px; margin-bottom: 25px; text-align: center;">
                            <div style="font-size: 1.1em; opacity: 0.9; margin-bottom: 10px;">⭐ IGM SCORE (Integrated Global Metric)</div>
                            <div style="font-size: 3.5em; font-weight: bold; margin: 10px 0;">{quality['igm_score']:.1f} / 100</div>
                            <div style="font-size: 1.2em; font-weight: 500; background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 5px; display: inline-block;">
                                {quality['qualitative_feedback']}
                            </div>
                            <div style="font-size: 0.85em; opacity: 0.8; margin-top: 15px;">
                                Weighted: ROM×40% + Tempo×30% + Depth×30%
                            </div>
                        </div>
                        
                        <h4>💪 Robust Fatigue Analysis</h4>
                        <table class="comparison-table" style="margin-bottom: 25px;">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Value</th>
                                    <th>Interpretation</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td class="metric-name">Trend</td>
                                    <td><strong>{fatigue_adv['trend'].upper().replace('_', ' ')}</strong></td>
                                    <td>
""")
                
                if fatigue_adv['trend'] == 'fatigue_detected':
                    f.write("""                                        <span class="badge badge-danger">⚠️ Range Decreasing</span>""")
                elif fatigue_adv['trend'] == 'progressive_exploration':
                    f.write("""                                        <span class="badge badge-info">ℹ️ ROM Increasing</span>""")
                elif fatigue_adv['trend'] == 'stable':
                    f.write("""                                        <span class="badge badge-success">✓ Consistent</span>""")
                else:
                    f.write("""                                        <span class="badge badge-warning">⚠️ No Clear Pattern</span>""")
                
                fatigue_color_cell = "#dc3545" if fatigue_adv['trend'] == 'fatigue_detected' else "#17a2b8" if fatigue_adv['trend'] == 'progressive_exploration' else "#28a745" if fatigue_adv['trend'] == 'stable' else "#ffc107"
                r2_status = "GOOD" if fatigue_adv['r_squared'] > 0.7 else "FAIR" if fatigue_adv['r_squared'] > 0.5 else "POOR"
                r2_badge = "badge-success" if fatigue_adv['r_squared'] > 0.7 else "badge-warning" if fatigue_adv['r_squared'] > 0.5 else "badge-danger"
                
                f.write(f"""
                                    </td>
                                </tr>
                                <tr>
                                    <td class="metric-name">Slope</td>
                                    <td>{fatigue_adv['slope']:+.3f}°/rep</td>
                                    <td style="color: {fatigue_color_cell};">{'Decreasing range' if fatigue_adv['slope'] < -0.5 else 'Increasing range' if fatigue_adv['slope'] > 0.5 else 'Stable'}</td>
                                </tr>
                                <tr>
                                    <td class="metric-name">R² (Fit Quality)</td>
                                    <td><strong>{fatigue_adv['r_squared']:.3f}</strong></td>
                                    <td><span class="badge {r2_badge}">{r2_status} FIT</span> {'✓ Trend Reliable' if fatigue_adv['r_squared'] > 0.5 else '✗ High Variability'}</td>
                                </tr>
                                <tr>
                                    <td class="metric-name">Statistical Significance</td>
                                    <td>{'YES' if fatigue_adv['significant'] else 'NO'}</td>
                                    <td>{'<span class="badge badge-success">p < 0.05</span>' if fatigue_adv['significant'] else '<span class="badge badge-secondary">p ≥ 0.05</span>'}</td>
                                </tr>
                            </tbody>
                        </table>
                        
                        <div style="background: {fatigue_color_cell}; color: white; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
                            <h5 style="margin: 0 0 10px 0; font-size: 1.1em;">📊 Interpretation</h5>
                            <p style="margin: 0; font-size: 1.05em; line-height: 1.6;">
""")
                
                if fatigue_adv['trend'] == 'fatigue_detected':
                    f.write("""                                <strong>⚠️ FATIGUE DETECTED:</strong> Range of motion is decreasing over repetitions. 
                                This suggests muscle fatigue or motor control degradation. Consider shorter sets or longer rest periods.
""")
                elif fatigue_adv['trend'] == 'progressive_exploration':
                    f.write("""                                <strong>ℹ️ PROGRESSIVE EXPLORATION:</strong> Range of motion is increasing over repetitions. 
                                The athlete is exploring ROM limits or improving warm-up. This is normal at session start but monitor for overstretching.
""")
                elif fatigue_adv['trend'] == 'stable':
                    f.write("""                                <strong>✓ STABLE PERFORMANCE:</strong> Range of motion remains consistent throughout exercise. 
                                Excellent motor control and no signs of fatigue. Performance is well-maintained.
""")
                else:
                    f.write(f"""                                <strong>⚠️ INCONSISTENT PATTERN:</strong> High variability (R²={fatigue_adv['r_squared']:.2f}) prevents reliable trend identification. 
                                The athlete is executing the movement inconsistently. Focus on technique and movement control before analyzing fatigue.
""")
                
                f.write("""                            </p>
                        </div>
                        
                        <h4>📊 Raw Coefficient of Variation (for reference)</h4>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="label">Range CV</div>
                                <div class="value">{:.1f}%</div>
                            </div>
                            <div class="stat-item">
                                <div class="label">Tempo CV</div>
                                <div class="value">{:.1f}%</div>
                            </div>
                            <div class="stat-item">
                                <div class="label">Timing CV</div>
                                <div class="value">{:.1f}%</div>
                            </div>
                        </div>
                        
                        <h4 style="margin-top: 30px;">✅ Valid Repetitions (After Filtering)</h4>
                        <div style="overflow-x: auto;">
                        <table class="comparison-table">
                            <thead>
                                <tr>
                                    <th>Rep #</th>
                                    <th>Time (s)</th>
                                    <th>Duration (s)</th>
                                    <th>Range (°)</th>
                                    <th>Depth (°)</th>
                                    <th>ROM Status</th>
                                </tr>
                            </thead>
                            <tbody>
""".format(rs_adv['raw_cvs']['range_cv']*100, rs_adv['raw_cvs']['tempo_cv']*100, rs_adv['raw_cvs']['timing_cv']*100))
                
                # Valid reps table
                for i in range(num_reps_adv):
                    rep_time = rs_adv['repetition_times'][i]
                    rep_dur = rs_adv['rep_durations'][i]
                    rep_range = rs_adv['rep_ranges'][i]
                    rep_depth = rs_adv['rep_depths'][i]
                    
                    # Check if within IQR bounds
                    if 'iqr_bounds' in filter_stats:
                        lower = filter_stats['iqr_bounds']['lower']
                        upper = filter_stats['iqr_bounds']['upper']
                        status = "✓ Valid" if lower <= rep_range <= upper else "⚠️ Edge"
                        status_badge = "badge-success" if lower <= rep_range <= upper else "badge-warning"
                    else:
                        status = "✓ Valid"
                        status_badge = "badge-success"
                    
                    f.write(f"""                                <tr>
                                    <td class="metric-name">{i+1}</td>
                                    <td>{rep_time:.2f}</td>
                                    <td>{rep_dur:.2f}</td>
                                    <td><strong>{rep_range:.2f}</strong></td>
                                    <td>{rep_depth:.2f}</td>
                                    <td><span class="badge {status_badge}">{status}</span></td>
                                </tr>
""")
                
                f.write("""                            </tbody>
                        </table>
                        </div>
                        
                        <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; margin-top: 25px; border-left: 5px solid #2196f3;">
                            <h5 style="margin: 0 0 10px 0; color: #1976d2;">💡 Key Takeaways</h5>
                            <ul style="margin: 0; padding-left: 20px; color: #424242;">
                                <li><strong>Base Analysis:</strong> Shows all detected movements (including half-reps, outliers)</li>
                                <li><strong>Advanced Analysis:</strong> Filters for quality - only biomechanically valid repetitions</li>
                                <li><strong>IGM Score:</strong> Single metric for performance evaluation (80+ = Excellent, 60-80 = Good, 40-60 = Fair, <40 = Poor)</li>
                                <li><strong>R² Validation:</strong> Ensures fatigue trends are statistically reliable, not random noise</li>
                                <li><strong>Use Case:</strong> Advanced metrics for athlete feedback, base counts for training volume</li>
                            </ul>
                        </div>
                        
                    </div>
                </div>
""")
            else:
                f.write("""
                        <div style="background: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;">
                            <h5 style="margin: 0 0 10px 0; color: #856404;">⚠️ No Valid Repetitions After Filtering</h5>
                            <p style="margin: 0; color: #856404;">
                                All detected repetitions were filtered out due to:
                                <ul style="margin: 10px 0 0 0; padding-left: 20px;">
                                    <li>ROM below threshold</li>
                                    <li>Statistical outliers (too far from median)</li>
                                    <li>First/last reps excluded (adjustment phase)</li>
                                </ul>
                                <strong>Suggestion:</strong> Review signal quality or lower ROM threshold for this exercise type.
                            </p>
                        </div>
                    </div>
                </div>
""")
        
        # Plots
        f.write(f"""                
                <h3 style="margin-top: 25px;">📊 Visualization</h3>
""")
        
        # Complete overview plot (only for non-filtered signals)
        # Overview plot section
        f.write(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <h4 style="margin: 0 0 10px 0; font-size: 1.1em;">🎯 COMPLETE TIME-SERIES OVERVIEW</h4>
                    <p style="margin: 0; font-size: 0.95em; opacity: 0.95;">
                        Full recording with all detected regions and repetitions. 
                        <strong>Top panel:</strong> Phase regions (ascending/descending/stable_high/stable_low).
                        <strong>Bottom panel:</strong> All detected repetitions with suspicious markers.
                    </p>
                </div>
                <div class="plot-container" style="grid-column: 1 / -1; margin-bottom: 30px;">
                    <img src="overview_complete_{angle_col}.png" alt="Complete Overview" style="width: 100%; max-width: 100%;">
                </div>
""")
        
        f.write(f"""                
                <h4 style="margin-top: 20px; color: #666;">Detailed Analysis Plots</h4>
                <div class="plot-grid">
                    <div class="plot-container">
                        <div class="plot-title">Autocorrelation</div>
                        <img src="autocorr_{angle_col}.png" alt="Autocorrelation">
                    </div>
                    <div class="plot-container">
                        <div class="plot-title">Frequency Spectrum</div>
                        <img src="spectrum_{angle_col}.png" alt="Spectrum">
                    </div>
                    <div class="plot-container">
                        <div class="plot-title">Spectrogram (Time-Frequency)</div>
                        <img src="spectrogram_{angle_col}.png" alt="Spectrogram">
                    </div>
""")
        
        # Phase detection plot - now available for ALL signals
        if angle_col in phase_stats and phase_stats[angle_col]['segments']:
            f.write(f"""                    <div class="plot-container">
                        <div class="plot-title">📍 Phase Detection (Ascending/Descending/Stable High/Low)</div>
                        <img src="phases_{angle_col}.png" alt="Phase Detection">
                    </div>
""")
        
        # Repetition detection plot - now available for ALL signals
        if angle_col in repetition_stats and repetition_stats[angle_col]['num_repetitions'] > 0:
            f.write(f"""                    <div class="plot-container">
                        <div class="plot-title">🏋️ Repetition Detection (Valleys & ROM)</div>
                        <img src="repetitions_{angle_col}.png" alt="Repetitions">
                    </div>
""")
        
        # Advanced repetition comparison plot (only for base axes with advanced analysis)
        if "_filtered" not in angle_col and f"{angle_col}_advanced" in repetition_stats:
            rs_adv = repetition_stats[f"{angle_col}_advanced"]
            if rs_adv['num_valid_repetitions'] > 0:
                f.write(f"""                    <div class="plot-container">
                        <div class="plot-title">🔬 Base vs Advanced Repetition Filtering</div>
                        <img src="repetitions_comparison_{angle_col}.png" alt="Repetition Comparison">
                    </div>
""")
        
        f.write("""                </div>
            </div>
        </div>
        
""")
    
    # Cross-axis correlations
    f.write("""        <!-- Cross-Axis Correlations -->
        <div class="section">
            <div class="section-header" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display=='none'?'block':'none'">
                <span>🔗 Cross-Axis Correlations</span>
                <span>▼</span>
            </div>
            <div class="section-content">
                <h3>Instant Correlations (Pearson)</h3>
                <div class="correlation-matrix">
""")
    
    for k, v in cross_corrs.items():
        ax1, ax2 = k.split('_')
        f.write(f"""                    <div class="corr-item">
                        <div class="axes">{ax1.upper()} ↔ {ax2.upper()}</div>
                        <div class="value" style="color: {'#28a745' if abs(v) > 0.7 else '#ffc107' if abs(v) > 0.4 else '#6c757d'};">{v:+.3f}</div>
                    </div>
""")
    
    f.write("""                </div>
                
                <h3 style="margin-top: 30px;">Lagged Cross-Correlations (Max 1s lag)</h3>
                <div class="correlation-matrix">
""")
    
    # Compute lagged correlations for HTML
    fs = len(ts) / (ts[-1] - ts[0])
    max_lag = int(fs * 1.0)
    
    for pair in [('ang1', 'ang2'), ('ang1', 'ang3'), ('ang2', 'ang3')]:
        lags, corr_xy = cross_corr_lags(
            resampled_df[pair[0]].to_numpy(),
            resampled_df[pair[1]].to_numpy(),
            max_lag, fs
        )
        lag_max_idx = np.argmax(np.abs(corr_xy))
        lag_max = lags[lag_max_idx]
        corr_max = corr_xy[lag_max_idx]
        
        f.write(f"""                    <div class="corr-item">
                        <div class="axes">{pair[0].upper()} ↔ {pair[1].upper()}</div>
                        <div class="value" style="color: {'#28a745' if abs(corr_max) > 0.7 else '#ffc107' if abs(corr_max) > 0.4 else '#6c757d'};">{corr_max:+.3f}</div>
                        <div class="lag">at lag {lag_max/fs:+.3f}s ({lag_max:+d} samples)</div>
                    </div>
""")
    
    f.write("""                </div>
            </div>
        </div>
        
        <!-- Time Series Comparison Plots -->
        <div class="section">
            <div class="section-header" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display=='none'?'block':'none'">
                <span>📈 Time Series Comparison</span>
                <span>▼</span>
            </div>
            <div class="section-content">
                <div class="plot-grid">
                    <div class="plot-container">
                        <div class="plot-title">Original Signals</div>
                        <img src="resampled_signals_original.png" alt="Original">
                    </div>
                    <div class="plot-container">
                        <div class="plot-title">Filtered Signals</div>
                        <img src="resampled_signals_filtered.png" alt="Filtered">
                    </div>
                    <div class="plot-container">
                        <div class="plot-title">ANG1: Original vs Filtered</div>
                        <img src="signal_comparison.png" alt="Comparison">
                    </div>
                </div>
            </div>
        </div>
        
    </div>
</div>

<script>
// Keep all sections OPEN by default (so interactive Plotly graphs are visible)
document.addEventListener('DOMContentLoaded', function() {
    const contents = document.querySelectorAll('.section-content');
    contents.forEach(content => {
        content.style.display = 'block';  // Changed from 'none' to 'block'
    });
});

// Global expand/collapse functions
function expandAll() {
    const contents = document.querySelectorAll('.section-content');
    contents.forEach(content => {
        content.style.display = 'block';
    });
}

function collapseAll() {
    const contents = document.querySelectorAll('.section-content');
    contents.forEach(content => {
        content.style.display = 'none';
    });
}

</script>

</body>
</html>
""")

print(f"HTML report saved to {report_path}")

# Export cycle, phase, and repetition statistics to JSON
cycles_json_path = report_path.replace(".html", "_cycles.json")
phases_json_path = report_path.replace(".html", "_phases.json")
repetitions_json_path = report_path.replace(".html", "_repetitions.json")

with open(cycles_json_path, "w") as fcycle:
    json.dump(cycle_stats, fcycle, indent=2)
print(f"Cycle statistics saved to {cycles_json_path}")

with open(phases_json_path, "w") as fphase:
    json.dump(phase_stats, fphase, indent=2)
print(f"Phase statistics saved to {phases_json_path}")

with open(repetitions_json_path, "w") as frep:
    json.dump(repetition_stats, frep, indent=2)
print(f"Repetition statistics saved to {repetitions_json_path}")

# Export comprehensive Excel workbook
excel_path = report_path.replace(".html", "_report.xlsx")
print(f"\n📊 Generating Excel workbook with all analysis data...")
if export_to_excel(cycle_stats, phase_stats, repetition_stats, resampled_df, excel_path):
    print(f"✅ Excel report saved to {excel_path}")
    print(f"   📄 Sheets: Signal_Data, Cycle_Summary, Phase_Summary, Reps_*, Quality_Metrics, Temporal_Parameters")
else:
    print(f"⚠️  Excel export skipped due to error")

print("\n" + "="*80)
print("🎉 ADVANCED BIOMECHANICS ANALYSIS COMPLETE")
print("="*80)
print("\n📦 Generated outputs:")
print(f"  ✅ Resampled data CSV with principal component (original + filtered)")
print(f"  ✅ Time series plots: original, filtered, and comparison")
print(f"  ✅ Autocorrelation plots (8 signals: 4 original + 4 filtered)")
print(f"  ✅ Frequency spectrum plots (8 signals) 📊 - shows true frequencies vs noise")
print(f"  ✅ Spectrograms (8 signals) 🌈 - time-frequency evolution")
print(f"  ✅ Comprehensive HTML report")
print(f"  ✅ JSON cycle & phase statistics")
print(f"  ✅ Excel workbook with all data (9+ sheets)\n")
print("\n💡 TIP: Compare original vs filtered spectrum plots to identify true frequencies!")
print("   Filtered signals show cleaner frequency peaks without noise artifacts.\n")

