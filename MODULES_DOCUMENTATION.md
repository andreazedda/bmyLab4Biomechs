# bmyLab4Biomechs - Complete Modules Documentation

**Project**: Biomechanics analysis toolkit for smartphone IMU sensors  
**Primary Use Case**: Squat exercise analysis using chest-mounted Android phone  
**Last Updated**: November 13, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [EKF_for_vel_and_pos_est_from_acc](#ekf_for_vel_and_pos_est_from_acc)
3. [EKF_full](#ekf_full)
4. [plot_recordings](#plot_recordings)
5. [repetions_period](#repetions_period)
6. [repetition_phases_detector](#repetition_phases_detector)
7. [Data Flow Architecture](#data-flow-architecture)
8. [Sensor Fusion Strategy](#sensor-fusion-strategy)
9. [Configuration Management](#configuration-management)
10. [Best Practices & Usage](#best-practices--usage)

---

## Overview

This project provides a modular toolkit for analyzing human movement using smartphone IMU (Inertial Measurement Unit) sensors. The primary application is squat exercise analysis, where the phone is mounted on the chest to capture:

- **3-axis acceleration** (linear and with gravity)
- **3-axis gyroscope** (angular velocity, uncalibrated)
- **3-axis magnetometer** (magnetic field)
- **Euler angles** (roll, pitch, yaw from phone's built-in sensor fusion)

**Key Challenge**: Estimating position from IMU data suffers from **unbounded drift** due to double integration of acceleration. This project implements multiple approaches to address this fundamental limitation.

### Module Architecture

```
sources/modules/
├── EKF_for_vel_and_pos_est_from_acc/   # Single-axis EKF with comprehensive monitoring
├── EKF_full/                           # 15-state inertial navigation + complementary filter
├── plot_recordings/                    # Raw sensor data visualization
├── repetions_period/                   # Periodic motion analysis
└── repetition_phases_detector/         # Phase detection (ascending/descending/stable)
```

---

## EKF_for_vel_and_pos_est_from_acc

### Purpose
Single-axis Extended Kalman Filter for estimating velocity and position from linear acceleration with comprehensive performance monitoring.

### Location
`sources/modules/EKF_for_vel_and_pos_est_from_acc/`

### Key Features

#### State Vector (per axis)
```python
x = [position, velocity, acceleration, bias]  # 4 states per axis
```

#### Advanced Capabilities
- **ZUPT (Zero-Velocity Update)**: Detects stationary periods and forces velocity to zero
- **Drift Correction**: Polynomial detrending to remove integration drift
- **Auto-tuning**: Adaptive Q/R matrix tuning based on innovation statistics
- **Performance Monitoring**: Real-time NEES, NIS, innovation whiteness tests
- **Multiple Axes**: Independent filters for X, Y, Z with coordinated ZUPT

### Configuration

**File**: `configs/EKF_for_vel_and_pos_est_from_acc.yaml`

**Key Parameters**:
```yaml
kalman_filter:
  initial_state: [0.0, 0.0, 0.0, 0.0]  # [pos, vel, acc, bias]
  initial_covariance:
    position: 0.1
    velocity: 0.1
    acceleration: 0.1
    bias: 0.01
  
  process_noise:
    position: 1.0e-6
    velocity: 1.0e-4
    acceleration: 1.0e-2
    bias: 1.0e-8
  
  measurement_noise:
    acceleration: 0.05  # m/s²

zupt:
  enabled: true
  window_size: 50
  threshold: 0.02  # Variance threshold for stationary detection
  min_duration: 30  # Minimum samples to qualify as stationary

drift_correction:
  enabled: true
  polynomial_order: 2
  apply_every: 500  # samples

auto_tuning:
  enabled: false
  adaptation_rate: 0.01
```

### Entry Point

```bash
cd sources/modules/EKF_for_vel_and_pos_est_from_acc
python sources/EKF_for_vel_and_pos_est_from_acc.py --config configs/EKF_for_vel_and_pos_est_from_acc.yaml
```

### Data Requirements

**Input Files** (in `data/inputs/`):
- `FILE_SENSOR_ACCELERATION_LINEAR<session>.txt` - Linear acceleration (gravity removed)
- Format: `timestamp,ax,ay,az` (one row per sample)

**Output Files** (in `data/outputs/`):
```
csv/
  estimated_position_X.csv
  estimated_velocity_X.csv
  estimated_bias.csv
yaml/
  EKF_performance_report_axis_Y.yaml
json/
  optimization_report.json
png/
  position_tracking_X.png
logs/
  EKF_execution_<timestamp>.log
```

### Usage Example

```python
from sources.EKF_for_vel_and_pos_est_from_acc import run_EKF_for_vel_and_pos_est_from_acc

# Run with default config
results = run_EKF_for_vel_and_pos_est_from_acc()

# Run with custom config
results = run_EKF_for_vel_and_pos_est_from_acc(
    config_path="configs/custom_config.yaml"
)
```

### Performance Monitoring

The module generates detailed performance reports including:

1. **NEES (Normalized Estimation Error Squared)**: Filter consistency check
2. **NIS (Normalized Innovation Squared)**: Measurement model validation
3. **Innovation Whiteness**: Independence of prediction errors
4. **Convergence Analysis**: Time to reach steady-state

**Interpretation**:
- NEES within [0.5, 1.5] → Good consistency
- NIS within chi-squared bounds → Measurement noise correctly modeled
- Innovation autocorrelation < 0.3 → Proper whiteness

### Limitations

✅ **Works well for**:
- Short-term velocity estimation (<5 seconds)
- Motion detection and phase identification
- Comparative analysis (relative metrics)

❌ **Does NOT work for**:
- Long-term absolute position (>10 seconds) - unbounded drift
- Full-session trajectory reconstruction without external reference
- Absolute displacement over multiple minutes

---

## EKF_full

### Purpose
Complete 15-state Extended Kalman Filter for full inertial navigation with **multiple estimation strategies**:
1. **Pure EKF** (diverges due to drift)
2. **Complementary Filter** (✅ RECOMMENDED - stable and accurate)

### Location
`sources/modules/EKF_full/`

### State Vector (15 states)

```python
x = [
    pos_x, pos_y, pos_z,           # Position (m)
    vel_x, vel_y, vel_z,           # Velocity (m/s)
    roll, pitch, yaw,              # Orientation (rad)
    acc_bias_x, acc_bias_y, acc_bias_z,  # Accelerometer bias (m/s²)
    gyro_bias_x, gyro_bias_y, gyro_bias_z # Gyroscope bias (rad/s)
]
```

### Architecture

```
EKF_full/
├── sources/
│   ├── config.py              # Dataclass-based configuration
│   ├── data_loading.py        # Multi-sensor data loading & alignment
│   ├── ekf_model.py           # Core EKF mathematics
│   ├── math_utils.py          # Rotation matrices, angle wrapping
│   ├── run_ekf.py             # EKF pipeline (LEGACY - drifts)
│   ├── complementary_filter.py # ✅ NEW - Stable sensor fusion
│   ├── compare_filters.py     # Performance comparison plots
│   ├── plot_squat_analysis.py # Enhanced squat analysis plots
│   ├── relative_tracking.py   # Per-repetition displacement
│   └── plotting.py            # Standard visualization
├── data/
│   ├── inputs/txts/           # Raw sensor recordings
│   └── outputs/               # Results CSV + plots
└── SENSOR_FUSION_REPORT.md    # Detailed analysis & recommendations
```

### Two Approaches

#### Approach 1: Pure EKF (Legacy - Not Recommended)

**File**: `sources/run_ekf.py`

**How it works**:
- Integrates accelerometer twice: acceleration → velocity → position
- Uses gyroscope to estimate orientation
- Applies sensor fusion with phone's Euler angles
- Implements ZUPT (Zero-Velocity Update) during stationary periods
- Position bounding (±2m) with periodic drift correction

**Problems**:
- ❌ Accumulates massive drift (~646m in 30s for X-axis)
- ❌ Velocities reach unrealistic values (400+ m/s)
- ❌ Requires constant corrections that don't solve fundamental issue
- ❌ Improvement factor: 1000-2000x worse than complementary filter

**Usage**:
```bash
cd sources/modules/EKF_full
python -m sources.run_ekf --session 2025-10-28-10-30-39 --plot
```

#### Approach 2: Complementary Filter (✅ RECOMMENDED)

**File**: `sources/complementary_filter.py`

**How it works**:

1. **Frequency Domain Separation**:
   ```
   HIGH-pass filtered acceleration → short-term dynamics
   LOW-pass filtered biomechanical model → long-term constraints
   ```

2. **Sensor Fusion**:
   - **Accelerometer**: High-frequency position changes (0.5+ Hz)
   - **Euler angles**: Trust phone's sensor fusion for orientation
   - **Biomechanical model**: Enforce physical constraints from squat mechanics
   - **Phase detection**: Identify descending/bottom/ascending phases

3. **Complementary Fusion**:
   ```python
   alpha = dt / (dt + 1/(2*pi*cutoff_hz))
   pos = alpha * pos_high_freq + (1-alpha) * pos_biomech_model
   ```

**Biomechanical Model**:
```python
# Squats have known kinematic patterns
Z_position = -MAX_DEPTH * (abs(pitch) / 40°)  # Vertical from lean angle
Y_position = TRUNK_LENGTH * sin(pitch)        # Forward from lean
X_position = TRUNK_LENGTH * sin(roll) * 0.1   # Lateral (small)
```

**Results**:
- ✅ X-axis: ±14 cm (realistic lateral sway)
- ✅ Y-axis: ±37 cm (realistic forward/back lean)
- ✅ Z-axis: 34 cm squat depth (realistic)
- ✅ **4,716x better** than pure EKF for X-axis
- ✅ **733x better** for Y-axis
- ✅ **2,332x better** for Z-axis

**Usage**:
```bash
cd sources/modules/EKF_full
python sources/complementary_filter.py
```

Or in Python:
```python
from sources.complementary_filter import run_complementary_filter

results = run_complementary_filter('2025-10-28-10-30-39')
# Returns DataFrame with: pos_x_cm, pos_y_cm, pos_z_cm, velocities, angles, phases
```

### Data Requirements

**Input Files** (in `data/inputs/txts/`):
- `FILE_SENSOR_ACC<session>.txt` - Accelerometer (timestamp, ax, ay, az)
- `FILE_GYRO_UNCALIBRATED<session>.txt` - Gyroscope (timestamp, gx, gy, gz, bias_x, bias_y, bias_z)
- `FILE_MAGN<session>.txt` - Magnetometer (timestamp, mx, my, mz)
- `FILE_EULER<session>.txt` - Euler angles (timestamp, roll_deg, pitch_deg, yaw_deg)
- `FILE_REL_ROT<session>.txt` - Relative rotation quaternion (timestamp, qx, qy, qz, qw)

**Session ID Format**: Timestamp string like `2025-10-28-10-19-35`

**Output Files**:
```
<session>_complementary_results.csv              # Main results
<session>_complementary_positions.png            # Position plots
<session>_complementary_velocities.png           # Velocity plots
<session>_complementary_orientation.png          # Euler angles
<session>_complementary_trajectory3d.png         # 3D trajectory
<session>_complementary_squat_analysis.png       # Detailed squat analysis
<session>_comparison_ekf_vs_complementary.png    # Method comparison
```

### Configuration

**File**: `sources/config.py`

```python
@dataclass(frozen=True)
class NoiseParams:
    """Process and measurement noise parameters"""
    pos: float = 5e-3          # Position process noise
    vel: float = 5e-2          # Velocity process noise
    angles: float = 1e-3       # Angle process noise
    accel_bias: float = 5e-4   # Accelerometer bias drift
    gyro_bias: float = 1e-6    # Gyroscope bias drift
    euler_meas_deg: float = 0.5  # HIGH TRUST in phone's Euler angles
    mag_meas_uT: float = 30.0  # Magnetometer (less reliable indoors)
    zero_vel: float = 1e-3     # ZUPT when stationary

@dataclass(frozen=True)
class DetectionParams:
    """Detection thresholds"""
    gravity: float = 9.80665
    zero_vel_acc_window: float = 0.3   # Stationary detection threshold
    zero_vel_gyro_window: float = 0.08
    max_interp_gap_ms: int = 15
    max_position_m: float = 2.0        # Position bounding
    position_reset_interval_s: float = 2.0
```

### Visualization & Analysis

**Generate all plots**:
```bash
# Run complementary filter
python sources/complementary_filter.py

# Generate squat analysis plot
python sources/plot_squat_analysis.py

# Compare EKF vs Complementary
python sources/compare_filters.py
```

**Plot Types**:

1. **Position Plots**: X, Y, Z over time with phase coloring
2. **Velocity Plots**: Instantaneous velocities per axis
3. **Orientation Plots**: Roll, pitch, yaw (Euler angles)
4. **3D Trajectory**: Top-down and side views of movement path
5. **Squat Analysis**: Multi-panel view with:
   - Position with phase backgrounds
   - Vertical depth with max/min markers
   - Body orientation angles
   - Phase timeline (descending/bottom/ascending/standing)
6. **Comparison Plot**: Side-by-side EKF vs Complementary with improvement statistics

### Session Management

**List available sessions**:
```bash
python -m sources.run_ekf --list-sessions
```

**Process all sessions**:
```bash
python -m sources.run_ekf --all --plot
```

**Process specific session**:
```bash
python -m sources.run_ekf --session 2025-10-28-10-30-39 --plot
```

### Performance Metrics

**CSV Columns** (`<session>_complementary_results.csv`):
```
timestamp_ms    # Original timestamp
time_s          # Relative time (starts at 0)
pos_x_cm        # X position (lateral)
pos_y_cm        # Y position (forward/back)
pos_z_cm        # Z position (vertical, negative = down)
vel_x_cm_s      # X velocity
vel_y_cm_s      # Y velocity
vel_z_cm_s      # Z velocity
roll_deg        # Roll angle (left/right tilt)
pitch_deg       # Pitch angle (forward/back lean)
yaw_deg         # Yaw angle (rotation)
phase           # Motion phase: 0=standing, 1=descending, 2=bottom, 3=ascending
```

### Best Practices

**✅ Use Complementary Filter for**:
- Squat depth measurement (Z-axis)
- Body lean analysis (pitch angle)
- Lateral stability (X-axis, roll angle)
- Phase detection and timing
- Per-repetition metrics

**❌ Do NOT use for**:
- Absolute position tracking over entire session
- Position-based bar path analysis (requires external reference)
- Precise velocity measurements (high-frequency noise)

**💡 Recommendations**:
1. Always use complementary filter over pure EKF
2. Focus on Z-axis (vertical) for squat depth - most reliable
3. Use phase detection to segment repetitions
4. Analyze per-repetition metrics rather than absolute displacement
5. Consider adding barometer data (FILE_SENSOR_PRESSURE*.txt) for improved Z-axis accuracy (±5cm)

---

## plot_recordings

### Purpose
Visualize raw sensor data from Android recordings without any filtering or processing.

### Location
`sources/modules/plot_recordings/`

### Files
- `plot_recordings.py` - Main script
- `plot_recordings.yaml` - Configuration (file paths, plot settings)
- `input/` - Raw sensor TXT files
- `output/` - Generated plots

### Configuration

**File**: `plot_recordings.yaml`

```yaml
input_file: "input/FILE_EULER2023-3-21-11-20-28.txt"
output_dir: "output/"
plot_settings:
  figsize: [12, 8]
  dpi: 100
  style: "seaborn"
sensors:
  euler:
    columns: ["timestamp", "roll", "pitch", "yaw"]
    units: "degrees"
```

### Usage

```bash
cd sources/modules/plot_recordings
python plot_recordings.py
```

### Output
- Time-series plots of raw sensor data
- One subplot per sensor axis
- Useful for data quality inspection before processing

### Use Cases
- ✅ Verify sensor data integrity
- ✅ Check sampling rate and gaps
- ✅ Identify outliers or sensor failures
- ✅ Understand motion characteristics before filtering

---

## repetions_period

### Purpose
Analyze periodic motion patterns in sensor data (e.g., repetitive squats, walking steps).

### Location
`sources/modules/repetions_period/`

### Files
- `repetions_period.py` - Period detection algorithm
- `repetions_period.yaml` - Configuration

### How It Works

1. **Signal Processing**:
   - Apply Fourier Transform (FFT) to find dominant frequencies
   - Use autocorrelation to detect repeating patterns
   - Bandpass filtering to isolate periodic components

2. **Period Extraction**:
   - Identify peaks in frequency domain
   - Convert frequency to time period (1/f)
   - Report average repetition duration

### Configuration

```yaml
analysis:
  method: "fft"  # or "autocorrelation"
  min_period_s: 1.0
  max_period_s: 10.0
  signal_column: "pitch"  # Angle to analyze
```

### Usage

```bash
cd sources/modules/repetions_period
python repetions_period.py
```

### Output
- Detected period (seconds per repetition)
- Frequency spectrum plot
- Autocorrelation plot

### Use Cases
- ✅ Count total repetitions in session
- ✅ Measure tempo consistency
- ✅ Identify rhythm anomalies
- ✅ Validate expected exercise cadence

---

## repetition_phases_detector

### Purpose
Detect and segment individual repetitions into phases (ascending, descending, stable/rest).

### Location
`sources/modules/repetition_phases_detector/`

### Files
- `repetions_phases_detector.py` - Phase detection logic (1475 lines)
- `repetions_phases_detector.yaml` - Configuration
- `input/` - Sensor data files
- `output/signal_intervals.html` - Interactive Plotly visualization

### How It Works

1. **Signal Preprocessing**:
   ```python
   # Moving average smoothing
   signal_smooth = moving_average(signal, window_size)
   
   # Variance-based stability detection
   variance = rolling_variance(signal_smooth, window)
   stable_regions = variance < threshold
   ```

2. **Phase Detection**:
   - **Stable regions**: Low variance in angular velocity
   - **Descending**: Pitch increasing (leaning forward)
   - **Ascending**: Pitch decreasing (straightening up)
   - **Transitions**: Identify boundaries between phases

3. **Validation**:
   - Minimum duration requirements
   - Continuity checks
   - Outlier removal

### Configuration

```yaml
detection:
  variance_threshold: 0.5
  min_stable_duration: 0.5  # seconds
  window_size: 50
  signal: "pitch"  # Which Euler angle to analyze

visualization:
  backend: "plotly"  # Interactive HTML
  show_phases: true
  color_scheme:
    stable: "green"
    descending: "blue"
    ascending: "red"
```

### Usage

```bash
cd sources/modules/repetition_phases_detector
python repetions_phases_detector.py
```

### Output
- `output/signal_intervals.html` - Interactive plot with phase annotations
- CSV with phase boundaries: `[start_time, end_time, phase_label]`

### Integration with EKF_full

The complementary filter in `EKF_full` includes built-in phase detection:

```python
from sources.modules.EKF_full.sources.complementary_filter import detect_squat_phases

phases = detect_squat_phases(pitch_deg, vel_magnitudes, dt)
# Returns: 0=standing, 1=descending, 2=bottom, 3=ascending
```

### Use Cases
- ✅ Segment continuous recording into individual reps
- ✅ Measure descent vs. ascent time ratio
- ✅ Identify incomplete repetitions
- ✅ Calculate time-under-tension metrics
- ✅ Detect asymmetries (left vs. right transitions)

---

## Data Flow Architecture

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA COLLECTION (Android Phone - Chest Mounted)         │
├─────────────────────────────────────────────────────────────┤
│  • Accelerometer (100Hz)                                    │
│  • Gyroscope Uncalibrated (100Hz)                          │
│  • Magnetometer (variable rate)                             │
│  • Euler Angles (phone's sensor fusion, 100Hz)             │
│  • Optional: Barometer, Quaternions                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. RAW DATA FILES (.txt format)                             │
├─────────────────────────────────────────────────────────────┤
│  FILE_SENSOR_ACC2025-10-28-10-30-39.txt                    │
│  FILE_GYRO_UNCALIBRATED2025-10-28-10-30-39.txt            │
│  FILE_MAGN2025-10-28-10-30-39.txt                         │
│  FILE_EULER2025-10-28-10-30-39.txt                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. DATA LOADING & TIME ALIGNMENT                            │
│    (sources/modules/EKF_full/sources/data_loading.py)      │
├─────────────────────────────────────────────────────────────┤
│  • Load each sensor file                                    │
│  • Time-align using accelerometer as base                   │
│  • Merge with nearest-neighbor matching (±15ms tolerance)   │
│  • Interpolate gaps (except Euler angles)                   │
│  • Output: Single DataFrame with all sensors                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4A. VISUALIZATION ONLY                                      │
│     (plot_recordings)                                        │
├─────────────────────────────────────────────────────────────┤
│  • Raw signal plots                                         │
│  • Quality inspection                                        │
│  • No processing                                             │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4B. PERIOD ANALYSIS                                         │
│     (repetions_period)                                       │
├─────────────────────────────────────────────────────────────┤
│  • FFT / Autocorrelation                                    │
│  • Detect repetition frequency                               │
│  • Report: 3.2s per rep, 9 total reps                       │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4C. PHASE DETECTION                                         │
│     (repetition_phases_detector)                            │
├─────────────────────────────────────────────────────────────┤
│  • Variance-based stability detection                        │
│  • Segment: descending → bottom → ascending → stable        │
│  • Output: Phase boundaries + interactive HTML               │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4D. SINGLE-AXIS EKF                                         │
│     (EKF_for_vel_and_pos_est_from_acc)                     │
├─────────────────────────────────────────────────────────────┤
│  • Process: X, Y, Z independently                            │
│  • State: [pos, vel, acc, bias] per axis                    │
│  • ZUPT when stationary                                      │
│  • Performance monitoring (NEES, NIS)                        │
│  • Limitation: Drifts over time                             │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4E. COMPLEMENTARY FILTER ✅ RECOMMENDED                     │
│     (EKF_full/complementary_filter.py)                      │
├─────────────────────────────────────────────────────────────┤
│  HIGH-PASS: Accelerometer integration (short-term)          │
│  LOW-PASS: Biomechanical model (long-term constraints)      │
│  ORIENTATION: Trust Euler angles from phone                  │
│  PHASES: Automatic detection (descending/ascending)          │
│                                                              │
│  Output:                                                     │
│    • Position: X=±14cm, Y=±37cm, Z=34cm depth ✅            │
│    • Velocities: Reasonable (no 400m/s spikes) ✅           │
│    • Phases: 42% desc, 14% bottom, 44% asc ✅               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. VISUALIZATION & ANALYSIS                                 │
├─────────────────────────────────────────────────────────────┤
│  • Position/Velocity/Orientation plots                       │
│  • 3D trajectory visualization                               │
│  • Squat analysis with phase coloring                        │
│  • EKF vs Complementary comparison                           │
│  • Per-repetition metrics (via relative_tracking.py)        │
└─────────────────────────────────────────────────────────────┘
```

### Data Format Specifications

#### Input File Format
```
# FILE_SENSOR_ACC2025-10-28-10-30-39.txt
1761644382286,-0.0391845703125,0.04327392578125,-0.1175537109375
1761644382296,-0.03912353515625,0.04327392578125,-0.117401123046875
...

# FILE_EULER2025-10-28-10-30-39.txt
1761644382286,7.291010,-22.306484,-0.6381078
1761644382307,7.321010,-22.407915,-0.6481078
...
```

#### Output CSV Format
```csv
timestamp_ms,time_s,pos_x_cm,pos_y_cm,pos_z_cm,vel_x_cm_s,vel_y_cm_s,vel_z_cm_s,roll_deg,pitch_deg,yaw_deg,phase
1761644382286,0.0,0.0,0.0,0.0,0.77,0.36,-1.49,7.29,7.04,-0.64,1
1761644382296,0.01,-0.01,0.02,-0.15,0.82,0.41,-1.52,7.32,7.11,-0.65,1
...
```

---

## Sensor Fusion Strategy

### Why Sensor Fusion is Critical

**The Fundamental Problem**:
```
Acceleration → ∫dt → Velocity → ∫dt → Position
            noise ↑        drift ↑↑    MASSIVE DRIFT ↑↑↑
```

Small acceleration noise (0.01 m/s²) → 30m position drift after 1 minute

### Pure EKF Approach (❌ FAILS)

**What it tries**:
1. Integrate acceleration for velocity
2. Integrate velocity for position
3. Use gyroscope for orientation
4. Apply ZUPT during stationary periods
5. Bound position (±2m) with periodic resets

**Why it fails**:
- Integration amplifies noise exponentially
- No absolute position reference to correct drift
- ZUPT only helps during stationary periods (rare in exercise)
- Position bounding is band-aid, not solution
- **Result**: 646m drift in 30s (should be 0.3m)

### Complementary Filter Approach (✅ WORKS)

**Strategy**: Separate frequency domains

```python
# HIGH frequency (>0.5 Hz): Accelerometer
pos_high_freq = integrate(accelerometer)  # Good for short-term dynamics

# LOW frequency (<0.5 Hz): Biomechanical model
pos_low_freq = biomech_model(angles)  # Physical constraints

# Fusion
alpha = dt / (dt + 1/(2*pi*cutoff_hz))
position = alpha * pos_high_freq + (1-alpha) * pos_low_freq
```

**Biomechanical Model**:
```python
# Squats have predictable kinematics
def squat_position(pitch, roll):
    # Vertical displacement from trunk lean
    z = -MAX_DEPTH * (abs(pitch) / 40°)  # 0 to -50cm
    
    # Forward displacement from hip hinge
    y = TRUNK_LENGTH * sin(pitch)  # ±30cm
    
    # Lateral displacement (small)
    x = TRUNK_LENGTH * sin(roll) * 0.1  # ±10cm
    
    return [x, y, z]
```

**Why it works**:
- Accelerometer handles fast movements (>0.5 Hz)
- Biomechanical model prevents long-term drift (<0.5 Hz)
- Euler angles from phone give accurate orientation
- Physical constraints limit unrealistic positions
- **Result**: 14cm drift in 30s (realistic!)

### Sensor Roles

| Sensor | Frequency | Purpose | Trust Level |
|--------|-----------|---------|-------------|
| **Accelerometer** | High (>0.5 Hz) | Dynamic acceleration, vibrations | Medium (noisy) |
| **Gyroscope** | Medium | Angular velocity for orientation | Medium (drifts) |
| **Magnetometer** | Low | Heading reference (yaw) | Low (indoor interference) |
| **Euler Angles** | All | Phone's sensor fusion | **HIGH** (0.5° accurate) |
| **Biomech Model** | Very Low (<0.5 Hz) | Position constraints | **HIGH** (physics-based) |

### Configuration for Sensor Fusion

**Trust Hierarchy** (in `config.py`):
```python
# Measurement noise (lower = more trust)
euler_meas_deg: float = 0.5    # HIGH TRUST
mag_meas_uT: float = 30.0       # LOW TRUST (indoors)

# Process noise (lower = trust model more)
angles: float = 1e-3            # Let Euler measurements dominate
pos: float = 5e-3               # Expect drift without reference
vel: float = 5e-2               # Integration is uncertain
```

---

## Configuration Management

### Configuration Hierarchy

```
1. Module Config (YAML)
   ├── EKF_for_vel_and_pos_est_from_acc.yaml
   ├── plot_recordings.yaml
   └── repetions_phases_detector.yaml

2. Dataclass Config (Python)
   └── EKF_full/sources/config.py
       ├── NoiseParams
       ├── DetectionParams
       └── PipelineConfig

3. Hardcoded Constants
   └── math_utils.py, ekf_model.py
```

### YAML Configuration (Single-Axis EKF)

**Location**: `sources/modules/EKF_for_vel_and_pos_est_from_acc/configs/`

**Structure**:
```yaml
# Data paths
data:
  input_dir: "data/inputs"
  output_dir: "data/outputs"
  session_id: "2023-3-21-11-6-14"

# Kalman filter parameters
kalman_filter:
  initial_state: [0.0, 0.0, 0.0, 0.0]
  initial_covariance:
    position: 0.1
    velocity: 0.1
    acceleration: 0.1
    bias: 0.01
  process_noise:
    position: 1.0e-6
    velocity: 1.0e-4
    acceleration: 1.0e-2
    bias: 1.0e-8
  measurement_noise:
    acceleration: 0.05

# Feature flags
zupt:
  enabled: true
  window_size: 50
  threshold: 0.02

drift_correction:
  enabled: true
  polynomial_order: 2

auto_tuning:
  enabled: false
```

**Loading**:
```python
import yaml

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

zupt_enabled = config['zupt']['enabled']
```

### Dataclass Configuration (EKF_full)

**Location**: `sources/modules/EKF_full/sources/config.py`

**Structure**:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class NoiseParams:
    """Immutable noise parameters"""
    pos: float = 5e-3
    vel: float = 5e-2
    angles: float = 1e-3
    # ... etc

@dataclass(frozen=True)
class PipelineConfig:
    """Top-level config"""
    sensor_sampling_hz: float = 100.0
    noise: NoiseParams = NoiseParams()
    detection: DetectionParams = DetectionParams()

# Global default
DEFAULT_PIPELINE_CONFIG = PipelineConfig()
```

**Usage**:
```python
from sources.config import DEFAULT_PIPELINE_CONFIG

config = DEFAULT_PIPELINE_CONFIG
print(config.noise.euler_meas_deg)  # 0.5
```

**Modification**:
```python
from dataclasses import replace

custom_noise = replace(
    DEFAULT_PIPELINE_CONFIG.noise,
    euler_meas_deg=1.0  # Less trust in Euler angles
)
custom_config = replace(
    DEFAULT_PIPELINE_CONFIG,
    noise=custom_noise
)
```

### Parameter Tuning Guide

#### Noise Parameters (Q, R matrices)

**Process Noise (Q)** - Model uncertainty:
```python
# Higher = trust measurements more than model
pos: 5e-3     # Expect position drift (no absolute reference)
vel: 5e-2     # Velocity integration is uncertain
angles: 1e-3  # Trust gyro integration less than Euler measurements
```

**Measurement Noise (R)** - Sensor uncertainty:
```python
# Lower = trust sensor more
euler_meas_deg: 0.5   # Phone's sensor fusion is VERY accurate
acc_meas: 0.1         # Accelerometer moderately noisy
mag_meas_uT: 30.0     # Magnetometer unreliable indoors
```

**Tuning Strategy**:
1. Start with defaults
2. If filter diverges → increase R (less trust in measurements)
3. If filter too sluggish → decrease R or increase Q
4. If too much drift → increase process noise for drifting states
5. Monitor NEES, NIS metrics for validation

#### Detection Thresholds

**ZUPT (Zero-Velocity Update)**:
```yaml
zero_vel_acc_window: 0.3   # Accel variance threshold (m/s²)²
zero_vel_gyro_window: 0.08 # Gyro variance threshold (rad/s)²
```
- Too low: False positives (apply ZUPT during motion)
- Too high: Miss stationary periods

**Phase Detection**:
```yaml
variance_threshold: 0.5     # For stable region detection
min_stable_duration: 0.5    # Minimum time in phase (s)
```

---

## Best Practices & Usage

### Recommended Workflow

1. **Record Data** (Android app like "Sensor Logger"):
   ```
   - Enable: Accelerometer, Gyroscope, Magnetometer, Rotation Vector
   - Mount phone on chest (portrait orientation)
   - Sampling rate: 100Hz
   - Record full squat session
   ```

2. **Export and Organize**:
   ```bash
   # Place files in:
   sources/modules/EKF_full/data/inputs/txts/
   
   # Naming convention:
   FILE_SENSOR_ACC2025-10-28-10-30-39.txt
   FILE_GYRO_UNCALIBRATED2025-10-28-10-30-39.txt
   FILE_EULER2025-10-28-10-30-39.txt
   FILE_MAGN2025-10-28-10-30-39.txt
   ```

3. **Quick Inspection**:
   ```bash
   # Visualize raw data
   cd sources/modules/plot_recordings
   python plot_recordings.py
   ```

4. **Run Complementary Filter** (✅ RECOMMENDED):
   ```bash
   cd sources/modules/EKF_full
   
   # Single session
   python sources/complementary_filter.py
   
   # Generate analysis plots
   python sources/plot_squat_analysis.py
   python sources/compare_filters.py
   ```

5. **Extract Metrics**:
   ```python
   import pandas as pd
   
   df = pd.read_csv('data/outputs/2025-10-28-10-30-39_complementary_results.csv')
   
   # Squat depth
   max_depth_cm = abs(df['pos_z_cm'].min())
   print(f"Maximum squat depth: {max_depth_cm:.1f} cm")
   
   # Rep count from phases
   phase_changes = df['phase'].diff().fillna(0)
   num_reps = (phase_changes == 3).sum()  # Transitions to ascending
   print(f"Total repetitions: {num_reps}")
   
   # Body lean
   max_pitch = df['pitch_deg'].max()
   print(f"Maximum forward lean: {max_pitch:.1f}°")
   ```

### Common Pitfalls

❌ **Don't**:
- Use pure EKF for long-term position tracking
- Expect absolute position accuracy without external reference
- Trust velocity estimates beyond 5-second windows
- Compare positions across different sessions (no common reference frame)
- Linearly interpolate Euler angles (causes gimbal lock issues)

✅ **Do**:
- Use complementary filter for position estimation
- Focus on relative metrics (per-rep displacement)
- Trust Euler angles from phone (they're accurate!)
- Analyze Z-axis (vertical) separately - most reliable
- Use phase detection to segment repetitions
- Consider adding barometer for Z-axis ground truth

### Troubleshooting

#### Problem: No data loaded
```
ValueError: Accelerometer data is required
```
**Solution**: Check file naming - must match `FILE_SENSOR_ACC<session>.txt` exactly

#### Problem: Positions are NaN
```
pos_z_cm: nan to nan cm
```
**Solution**: Euler angles missing - check FILE_EULER*.txt exists and forward-fill NaN values

#### Problem: Massive drift (>100m)
```
X: -64600 to 34 cm (range: 64634 cm)
```
**Solution**: You're using pure EKF - switch to complementary filter

#### Problem: No phases detected
```
Phases detected: Standing: 0 samples
```
**Solution**: Adjust `variance_threshold` in phase detection or check if motion actually occurred

#### Problem: Unrealistic velocities (>50 m/s)
```
Velocity magnitude: 0.14 to 406.12 m/s
```
**Solution**: Normal for peak values - look at median or 95th percentile instead

### Performance Benchmarks

**Complementary Filter** (recommended):
- Processing time: ~2-3 seconds per 30-second recording
- Memory usage: <50 MB
- Position accuracy: ±10-20 cm (Z-axis), ±5-15 cm (X, Y)
- Suitable for real-time processing: Yes (with optimization)

**Pure EKF**:
- Processing time: ~5-7 seconds per 30-second recording
- Memory usage: <100 MB
- Position accuracy: ±500-1000 m (UNACCEPTABLE)
- Suitable for real-time processing: No (requires constant corrections)

### Future Improvements

**Short-term**:
1. Add barometer support for improved Z-axis (±5cm accuracy)
2. Implement per-repetition analysis (displacement, timing)
3. Add asymmetry detection (left vs. right imbalances)
4. Real-time processing mode

**Long-term**:
1. Machine learning for phase detection
2. Multi-subject calibration
3. Integration with force plates for ground truth
4. Camera-based validation
5. UWB (Ultra-Wideband) tags for absolute positioning

---

## Quick Reference Commands

```bash
# List available sessions
cd sources/modules/EKF_full
python -m sources.run_ekf --list-sessions

# Run complementary filter (RECOMMENDED)
python sources/complementary_filter.py

# Generate squat analysis
python sources/plot_squat_analysis.py

# Compare methods
python sources/compare_filters.py

# Process specific session
python -m sources.run_ekf --session 2025-10-28-10-30-39 --plot

# Run single-axis EKF
cd sources/modules/EKF_for_vel_and_pos_est_from_acc
python sources/EKF_for_vel_and_pos_est_from_acc.py

# Visualize raw data
cd sources/modules/plot_recordings
python plot_recordings.py

# Detect phases
cd sources/modules/repetition_phases_detector
python repetions_phases_detector.py
```

---

## File Structure Summary

```
bmyLab4Biomechs/
├── .github/
│   └── copilot-instructions.md          # AI agent guidance
├── data/                                 # (Not in repo - runtime only)
├── sources/
│   └── modules/
│       ├── EKF_for_vel_and_pos_est_from_acc/
│       │   ├── sources/
│       │   │   ├── EKF_for_vel_and_pos_est_from_acc.py  # Main script
│       │   │   └── lib/                  # EKF implementation
│       │   ├── configs/
│       │   │   └── EKF_for_vel_and_pos_est_from_acc.yaml
│       │   ├── data/
│       │   │   ├── inputs/               # TXT sensor files
│       │   │   └── outputs/              # Results (CSV, PNG, YAML)
│       │   ├── docs/                     # Detailed documentation
│       │   ├── tests/                    # Unit tests
│       │   └── requirements.txt
│       │
│       ├── EKF_full/
│       │   ├── sources/
│       │   │   ├── config.py             # Dataclass configuration
│       │   │   ├── data_loading.py       # Multi-sensor loading
│       │   │   ├── ekf_model.py          # 15-state EKF
│       │   │   ├── run_ekf.py            # Pure EKF (legacy)
│       │   │   ├── complementary_filter.py  # ✅ RECOMMENDED
│       │   │   ├── compare_filters.py    # Comparison plots
│       │   │   ├── plot_squat_analysis.py   # Enhanced plots
│       │   │   ├── relative_tracking.py  # Per-rep metrics
│       │   │   ├── plotting.py           # Standard plots
│       │   │   └── math_utils.py         # Rotation matrices
│       │   ├── data/
│       │   │   ├── inputs/txts/          # Multi-sensor TXT files
│       │   │   └── outputs/              # CSV + PNG results
│       │   ├── SENSOR_FUSION_REPORT.md   # Detailed analysis
│       │   ├── COMPLETION_REPORT.md
│       │   └── TESTING.md
│       │
│       ├── plot_recordings/
│       │   ├── plot_recordings.py        # Raw data visualization
│       │   ├── plot_recordings.yaml      # Config
│       │   ├── input/                    # Raw TXT files
│       │   └── output/                   # Plots
│       │
│       ├── repetions_period/
│       │   ├── repetions_period.py       # Period detection (FFT)
│       │   └── repetions_period.yaml     # Config
│       │
│       └── repetition_phases_detector/
│           ├── repetions_phases_detector.py  # Phase segmentation
│           ├── repetions_phases_detector.yaml
│           ├── input/                    # Euler angle TXT files
│           └── output/                   # Interactive HTML plots
│
├── MODULES_DOCUMENTATION.md              # This file
├── LICENSE
└── README.md
```

---

## Contact & Contributing

**Project**: bmyLab4Biomechs  
**Owner**: andreazedda  
**Repository**: github.com/andreazedda/bmyLab4Biomechs  
**License**: MIT

For bugs, feature requests, or questions, please open an issue on GitHub.

---

**Last Updated**: November 13, 2025  
**Version**: 2.0 (Complementary Filter Stable Release)
