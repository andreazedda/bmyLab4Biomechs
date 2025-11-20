import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, find_peaks, periodogram, spectrogram, coherence
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import adfuller, acf
from scipy.stats import linregress
from sklearn.decomposition import PCA

# Helper functions for cycle and phase detection
def detect_cycles(signal, prominence=1.0, distance=100):
    """Detect cycles using peaks (e.g., squat repetitions)"""
    peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-signal, prominence=prominence, distance=distance)
    cycles = sorted(list(peaks) + list(troughs))
    return cycles, peaks, troughs

def detect_phases(signal, peaks, troughs):
    """Improved phase detection using linear regression on segments"""
    phases = []
    idxs = sorted(list(peaks) + list(troughs))
    for i in range(len(idxs)-1):
        start, end = idxs[i], idxs[i+1]
        seg = signal[start:end]
        if len(seg) < 5:
            phases.append((start, end, 'stable'))
            continue
        x = np.arange(len(seg))
        slope, _, _, _, _ = linregress(x, seg)
        if slope > 0.01:  # threshold to avoid noise
            ph = 'ascending'
        elif slope < -0.01:
            ph = 'descending'
        else:
            ph = 'stable'
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
    phase_durations = {'ascending': [], 'descending': [], 'stable': []}
    
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
    
    return {
        'num_repetitions': num_reps,
        'trough_indices': troughs,
        'repetition_times': rep_times,
        'inter_rep_intervals': inter_rep_intervals,
        'repetition_durations': np.array(rep_durations),
        'rep_ranges': np.array(rep_ranges),
        'rep_depths': np.array(rep_depths),
        'prominence_threshold': min_prominence,
    }

def analyze_repetition_quality(rep_data):
    """Analyze quality metrics for exercise repetitions.
    
    Args:
        rep_data: dict from detect_exercise_repetitions()
    
    Returns:
        dict with quality metrics
    """
    if rep_data['num_repetitions'] == 0:
        return {
            'consistency_score': 0.0,
            'range_cv': 0.0,
            'timing_cv': 0.0,
            'fatigue_indicator': 0.0,
            'tempo_consistency': 0.0,
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
    
    return {
        'consistency_score': consistency_score,
        'range_cv': range_cv,
        'timing_cv': timing_cv,
        'tempo_cv': tempo_cv,
        'fatigue_indicator': fatigue_indicator,
    }

# Main script
print("="*80)
print("BIOMECHANICS SIGNAL ANALYSIS - EULER ANGLES")
print("="*80)

file_path = "/Volumes/nvme/Github/bmyLab4Biomechs/sources/modules/find_time_intervals/FILE_EULER2025-10-28-10-23-31.txt"

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")
print(f"\n📂 File found: {os.path.basename(file_path)}")
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
    rep_quality = analyze_repetition_quality(rep_data)
    
    # Save repetition analysis data
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
        }
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
    for phase_type in ['ascending', 'descending', 'stable']:
        pstats = phase_statistics[phase_type]
        if pstats['count'] > 0:
            print(f"     • {phase_type:>11s}: n={pstats['count']:3d}, mean={pstats['mean_duration']:.3f}s, "
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
    print(f"     ╚{'═'*75}╝")
    
    if rep_data['num_repetitions'] > 0:
        # Repetition timing details
        print(f"\n     📊 PER-REPETITION METRICS:")
        print(f"     ┌{'─'*75}┐")
        print(f"     │ {'Rep':<4} │ {'Time (s)':<10} │ {'Duration (s)':<12} │ {'Range (°)':<11} │ {'Depth (°)':<11} │")
        print(f"     ├{'─'*75}┤")
        
        for i in range(rep_data['num_repetitions']):
            rep_num = i + 1
            rep_time = rep_data['repetition_times'][i]
            rep_dur = rep_data['repetition_durations'][i]
            rep_range = rep_data['rep_ranges'][i]
            rep_depth = rep_data['rep_depths'][i]
            print(f"     │ {rep_num:<4} │ {rep_time:>10.2f} │ {rep_dur:>12.2f} │ {rep_range:>11.2f} │ {rep_depth:>11.2f} │")
        
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
                <span>🧬 Principal Component Analysis</span>
                <span>▼</span>
            </div>
            <div class="section-content">
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
            for phase_type in ['ascending', 'descending', 'stable']:
                pstats = ps[phase_type]
                if pstats['count'] > 0:
                    f.write(f"""                        <tr>
                            <td class="metric-name">{'🔼' if phase_type=='ascending' else '🔽' if phase_type=='descending' else '➡️'} {phase_type.capitalize()}</td>
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
                    </div>
                </div>
""")
            
            if num_reps > 0:
                # Per-repetition details table
                f.write("""
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
                
                for i in range(num_reps):
                    rep_time = rs['repetition_times'][i]
                    rep_dur = rs['rep_durations'][i]
                    rep_range = rs['rep_ranges'][i]
                    rep_depth = rs['rep_depths'][i]
                    
                    # Quality indicator: check if within ±20% of mean
                    range_quality = "✓" if abs(rep_range - mean_range) / (mean_range + 1e-9) < 0.2 else "⚠️" if abs(rep_range - mean_range) / (mean_range + 1e-9) < 0.4 else "❌"
                    duration_quality = "✓" if abs(rep_dur - mean_duration) / (mean_duration + 1e-9) < 0.2 else "⚠️"
                    
                    quality_badge = "badge-success" if range_quality == "✓" and duration_quality == "✓" else "badge-warning" if "⚠️" in [range_quality, duration_quality] else "badge-danger"
                    quality_text = "Good" if range_quality == "✓" and duration_quality == "✓" else "Fair" if "⚠️" in [range_quality, duration_quality] else "Poor"
                    
                    f.write(f"""                        <tr>
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
        
        # Plots
        f.write(f"""                
                <h3 style="margin-top: 25px;">📊 Visualization</h3>
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
                </div>
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
// Auto-collapse sections on load
document.addEventListener('DOMContentLoaded', function() {
    const contents = document.querySelectorAll('.section-content');
    contents.forEach(content => {
        content.style.display = 'none';
    });
    // Open first section by default
    if (contents.length > 0) {
        contents[0].style.display = 'block';
    }
});
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
print(f"  ✅ JSON cycle & phase statistics\n")
print("\n💡 TIP: Compare original vs filtered spectrum plots to identify true frequencies!")
print("   Filtered signals show cleaner frequency peaks without noise artifacts.\n")

