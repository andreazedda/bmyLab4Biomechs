# bmyLab4Biomechs - AI Coding Agent Instructions

## Project Overview

This is a **biomechanics research toolkit** for analyzing human movement (primarily squats) using smartphone IMU sensors. The project processes Android sensor recordings (accelerometer, gyroscope, magnetometer, Euler angles) through Extended Kalman Filters (EKF) to estimate position, velocity, and motion phases during exercise.

**Key Domain**: Inertial navigation for biomechanical analysis with smartphone sensors at 100Hz sampling.

## Architecture & Module Structure

The project follows a **modular design** under `sources/modules/`:

### Core Modules

1. **`EKF_for_vel_and_pos_est_from_acc/`** - Single-axis EKF with comprehensive monitoring
   - **State**: `[position, velocity, acceleration, bias]` per axis
   - **Features**: ZUPT (Zero-Velocity Updates), drift correction, auto-tuning
   - **Entry**: `sources/EKF_for_vel_and_pos_est_from_acc.py::run_EKF_for_vel_and_pos_est_from_acc()`
   - **Config**: YAML-driven at `configs/EKF_for_vel_and_pos_est_from_acc.yaml`

2. **`EKF_full/`** - 15-state inertial navigation EKF
   - **State**: `[pos(3), vel(3), angles(3), acc_bias(3), gyro_bias(3)]`
   - **Entry**: `sources/run_ekf.py::run_session(session_id)`
   - **Data Flow**: Loads multi-sensor TXT files → time-aligns → EKF prediction/update
   - **Config**: Dataclass-based (`config.py::PipelineConfig`)

3. **`plot_recordings/`** - Visualization of raw sensor data
4. **`repetions_period/`** - Periodic motion analysis
5. **`repetition_phases_detector/`** - Phase detection (ascending/descending/stable)

### Data Flow Pattern

```
Android TXT Files (data/inputs/txts/FILE_*.txt)
  ↓
SessionData (data_loading.py)
  ↓
build_time_aligned_frame() → Merged DataFrame
  ↓
EKF predict/update loop
  ↓
Results CSV + Plots (data/outputs/)
```

## Critical Conventions

### File Naming Standards
- **Input files**: `FILE_SENSOR_ACC<session>.txt`, `FILE_GYRO_UNCALIBRATED<session>.txt`, `FILE_EULER<session>.txt`
  - Example: `FILE_EULER2025-10-28-10-19-35.txt`
- **Session IDs**: Timestamp strings like `2025-10-28-10-19-35`
- **Output files**: `<session>_ekf_results.csv` or axis-specific `estimated_velocity_Y.csv`

### Configuration Management
- **YAML configs** are the single source of truth for parameters
- **Never hardcode** noise parameters (Q, R matrices) - always use config values
- **EKF tuning parameters** live under `kalman_filter:` section with nested `process_noise:` and `measurement_noise:`
- **Biomechanical-specific**: Configs include `zupt:`, `drift_correction:`, `velocity_constraint:` sections

### EKF Implementation Patterns

#### State Updates
```python
# Prediction step ALWAYS precedes update
ekf.predict(dt)
ekf.update(measurement)

# ZUPT application during stationary periods
if is_stationary(data):
    ekf.x[1, 0] = 0.0  # Zero velocity
    ekf.P[1, 1] *= 0.1  # Reduce velocity uncertainty
```

#### Performance Monitoring
- **Real-time diagnostics**: Use `EKFPerformanceMonitor` class for NEES, NIS, innovation analysis
- **Always check convergence** via `monitor.check_convergence()` every N samples
- **Statistical tests**: NEES (filter consistency), NIS (measurement model), innovation whiteness

### Logging & Debugging
- **Colored output**: Uses `colorama` extensively - `print_colored(msg, emoji, color)`
- **Dual logging**: Console + file (`logs/EKF_execution_<timestamp>.log`)
- **Progress tracking**: Show percentage completion for long loops (every 10%)

## Development Workflows

### Running EKF Modules

**Single-axis EKF**:
```bash
cd sources/modules/EKF_for_vel_and_pos_est_from_acc
python sources/EKF_for_vel_and_pos_est_from_acc.py
# Or with custom config:
python sources/EKF_for_vel_and_pos_est_from_acc.py --config path/to/config.yaml
```

**Full 15-state EKF**:
```bash
cd sources/modules/EKF_full
python -m sources.run_ekf --session 2025-10-28-10-19-35 --plots
```

### Testing
- **Unit tests**: `sources/modules/EKF_for_vel_and_pos_est_from_acc/tests/`
- Run with: `python -m pytest tests/`
- **Test coverage**: Focuses on configuration validation, filter initialization, performance monitoring

### Adding New Features
1. **Update config schema** in YAML + validation logic
2. **Extend state vector** carefully - update Jacobian matrices (F, H)
3. **Add monitoring** if new observable state components
4. **Document in** `docs/` folder (API_REFERENCE.md, USER_GUIDE.md)

## Key Technical Details

### ZUPT (Zero-Velocity Update) Logic
- **Detection**: Variance of acceleration < threshold over sliding window
- **Application**: Force velocity to zero, tighten velocity covariance
- **Biomechanics-specific**: More permissive thresholds for exercise motion vs. walking
- **Config keys**: `zupt.window_size`, `zupt.threshold`, `zupt.min_duration`

### Drift Correction
- **Velocity drift**: Polynomial detrending (order 2-3) using `drift_correction.polynomial_order`
- **Position drift**: Cyclic constraints via `position_constraint.apply_every` samples
- **Applied post-EKF** to remove integration drift artifacts

### Coordinate Frames
- **Body frame**: Sensor readings (accelerometer/gyro/magnetometer)
- **World frame**: After rotation by Euler angles or EKF-estimated orientation
- **Gravity handling**: `use_gravity_compensation: false` when using linear acceleration (gravity pre-removed)

## Common Issues & Solutions

### Filter Divergence
- **Symptom**: Covariance trace grows exponentially, `P[i,i] > max_trace` warnings
- **Fix**: Reduce Q (process noise), increase R (measurement noise), check for measurement outliers
- **Prevention**: Enable `numerics.use_joseph` for Joseph form covariance update

### Slow Convergence
- **Symptom**: High initial covariance, many samples before `convergence_check_interval` passes
- **Fix**: Tune `initial_covariance` closer to expected uncertainty, adjust Q/R ratio

### Data Loading Errors
- **Symptom**: `ValueError: Accelerometer data is required`
- **Check**: File names match `SENSOR_METADATA` prefixes in `data_loading.py`
- **Check**: Session ID matches between ACC/GYRO/EULER files

## Dependencies
- **Core**: `numpy`, `pandas`, `scipy`, `matplotlib`
- **Config**: `pyyaml` for YAML parsing
- **UI**: `colorama` for terminal colors, `plotly` for interactive plots
- Install: `pip install -r sources/modules/EKF_for_vel_and_pos_est_from_acc/requirements.txt`

## Documentation
- **`sources/modules/EKF_for_vel_and_pos_est_from_acc/docs/`**:
  - `README.md`: Architecture overview
  - `API_REFERENCE.md`: Class/method signatures
  - `USER_GUIDE.md`: Parameter tuning guide
  - `TECHNICAL.md`: Mathematical formulations
  - `EXAMPLES.md`: Configuration recipes

## Output Artifacts
- **CSV**: Time-series data (`estimated_position_X.csv`, etc.)
- **YAML**: Performance reports (`EKF_performance_report_axis_Y.yaml`)
- **JSON**: Optimization results (`optimization_report.json`)
- **PNG**: Plots (trajectories, residuals)
- **Logs**: `data/outputs/logs/EKF_execution_*.log`

## License
MIT License - see `LICENSE` file
