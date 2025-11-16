# Time Parameters Finder - Adaptive Algorithm

## 🎯 Overview

This module analyzes squat movement data from smartphone IMU sensors and automatically detects repetitions with temporal phase segmentation. The algorithm now includes **adaptive parameter optimization** that can target a specific number of expected repetitions.

## ✨ New Features

### 1. Adaptive Parameter Optimization
The algorithm automatically searches through multiple parameter combinations to find the best settings for your data:
- **Velocity thresholds**: 14 different values tested (3-150 deg/s range)
- **Phase durations**: 10 different values tested (0.08-0.4s range)
- **Total combinations**: Up to 140 parameter sets evaluated

### 2. Target-Based Detection
Set `expected_reps` to tell the algorithm how many repetitions you expect:
```python
metrics = analyze_squat_file_auto_axis(
    path="your_data.txt",
    adaptive=True,
    expected_reps=10,  # Algorithm will prioritize finding exactly 10 reps
)
```

When `expected_reps` is specified:
- Parameters that find the exact number get a **10x score boost**
- The algorithm reports how close it got to the target
- Top results show which parameter sets match your expectation

### 3. Improved Visualization
The plot now features:
- **Distinct colors**: Red (eccentric ↓), Orange (bottom ●), Green (concentric ↑), Gray (stand)
- **Higher resolution**: 300 DPI for publication-quality images
- **Better contrast**: Dark blue signal line with enhanced phase overlays
- **Clear legend**: With unicode symbols for phase direction

## 🚀 Quick Start

### Basic Usage (Auto-detect repetitions)
```python
from time_parameters_finder import analyze_squat_file_auto_axis

metrics = analyze_squat_file_auto_axis(
    path="FILE_EULER2025-10-28-10-19-35.txt",
    adaptive=True,  # Enable adaptive optimization
    make_plot=True,
)

print(f"Found {metrics['N_rep']} repetitions")
```

### Target-Based Usage (Find specific number)
```python
metrics = analyze_squat_file_auto_axis(
    path="FILE_EULER2025-10-28-10-19-35.txt",
    adaptive=True,
    expected_reps=8,  # Try to find exactly 8 reps
    make_plot=True,
)
```

### Manual Parameter Control
```python
metrics = analyze_squat_file_auto_axis(
    path="your_data.txt",
    adaptive=False,  # Disable optimization
    vel_thresh=20.0,  # Manual velocity threshold
    min_phase_duration=0.2,  # Manual phase duration
    make_plot=True,
)
```

## 📊 Output

The algorithm returns a comprehensive dictionary with:

### Core Metrics (46 temporal parameters)
- **N_rep**: Number of repetitions detected
- **T_stand_init/final**: Standing time before/after exercise
- **Totals**: T_ecc_tot, T_con_tot, T_work_tot, T_buca_tot, T_top_tot, T_pause_tot, T_TUT_tot
- **Means**: Average time per phase across all reps
- **Variances**: Variance of each temporal parameter
- **CV (Coefficient of Variation)**: Consistency metrics
- **Trends**: Slope of temporal parameters across reps
- **Ratios**: Work/Pause, useful_time_frac, set_density

### Optimization Info (when adaptive=True)
- **tested_combinations**: Number of parameter sets evaluated
- **best_score**: Quality score of the optimal parameters
- **all_results**: Top 10 parameter combinations ranked by score

### Parameters Used
- **vel_thresh**: Optimized velocity threshold
- **min_phase_duration**: Optimized phase duration
- **smooth_window**: Smoothing window size

## 🔧 Configuration

Edit the `__main__` block in `time_parameters_finder.py`:

```python
if __name__ == "__main__":
    path = "your_data_file.txt"
    
    # Change this to your expected number of reps
    EXPECTED_REPS = 10  # Or None for auto-detection
    
    metrics = analyze_squat_file_auto_axis(
        path=path,
        smooth_window=51,  # For ~750Hz data
        adaptive=True,
        expected_reps=EXPECTED_REPS,
        make_plot=True,
        plot_path="squat_fasi.png",
    )
```

## 📈 How It Works

1. **Data Loading**: Reads CSV with timestamp_ms, ang1, ang2, ang3
2. **Axis Selection**: Automatically selects the axis with maximum rotation
3. **Parameter Search** (if adaptive):
   - Tests 14 velocity thresholds × 10 phase durations
   - Scores each combination based on:
     * Number of repetitions found
     * Phase balance (all 4 phases present)
     * Consistency (low coefficient of variation)
     * Target match (if expected_reps specified)
4. **Phase Classification**:
   - STAND: Low velocity near standing baseline
   - ECC (Eccentric): Positive velocity (descending)
   - BOTTOM: Low velocity at squat depth
   - CONC (Concentric): Negative velocity (ascending)
5. **Temporal Filtering**: Removes spurious short phases
6. **Rep Segmentation**: Finds STAND→ECC→BOTTOM→CONC→STAND patterns
7. **Metric Calculation**: Computes all 46 temporal parameters

## 🎨 Plot Interpretation

- **Red zones**: Eccentric (descending) phase
- **Orange zones**: Bottom (pause at depth)
- **Green zones**: Concentric (ascending) phase
- **Gray zones**: Standing (between reps)
- **Dark blue line**: Angular displacement over time

## 🐛 Troubleshooting

### No Repetitions Found
- Try setting `expected_reps=None` to let the algorithm explore more freely
- Check if your data contains actual repetitions
- Verify file format: CSV with timestamp_ms, ang1, ang2, ang3

### Wrong Number Detected
- Set `expected_reps` to your known count
- Check the optimization diagnostics to see alternative parameter sets
- The data quality might limit accurate detection

### Plot Colors Hard to Distinguish
- Updated version uses distinct RGB colors with proper contrast
- View at 100% zoom for best clarity
- Phase symbols (↓●↑) help identify phases in legend

## 📝 Example Output

```
🔍 Analyzing squat data with ADAPTIVE parameter optimization...
🎯 Target: 10 repetitions (algorithm will adapt to find this number)

======================================================================
📊 SQUAT ANALYSIS RESULTS
======================================================================

🎯 Principal axis: ang3 (axis 2)
📈 Number of repetitions detected: 6
⚠️  Found 6 reps, expected 10 (difference: 4)

⚙️  Optimized Parameters:
   • Velocity threshold: 33.42 deg/s
   • Min phase duration: 0.08 s
   • Smoothing window: 51 samples

🔬 Optimization diagnostics:
   • Tested 49 parameter combinations
   • Best score: 2.36
   • Top 5 parameter sets:
     1. vel=33.4, dur=0.08 → 6 reps (score=2.36)
     2. vel=53.5, dur=0.08 → 6 reps (score=2.04)
     ...

⏱️  Timing Metrics:
   • Total work time: 2.02 s
   • Total Time Under Tension (TUT): 5.08 s
   • Work/Pause ratio: 0.66
```

## 🔗 Dependencies

- numpy
- pandas
- matplotlib
- Python 3.10+

## 📄 License

MIT License - Part of bmyLab4Biomechs project
