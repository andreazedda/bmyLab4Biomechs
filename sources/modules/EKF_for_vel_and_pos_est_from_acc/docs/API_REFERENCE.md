# API Reference: EKF Module

## 📚 Core Classes

### ExtendedKalmanFilter

Main implementation of the Extended Kalman Filter for motion estimation.

#### Constructor

```python
ExtendedKalmanFilter(config: dict)
```

**Parameters:**
- `config`: Dictionary containing filter configuration parameters

**Configuration Keys:**
- `kalman_filter.initial_state`: Initial state vector [p, v, a, b]
- `kalman_filter.initial_covariance`: Initial covariance matrix (4x4 flattened)
- `kalman_filter.process_noise`: Process noise parameters
- `kalman_filter.measurement_noise`: Measurement noise variance
- `kalman_filter.gravity`: Gravity constant (m/s²)

#### Methods

##### predict(dt: float) → None

Performs the prediction step of the Kalman filter.

**Parameters:**
- `dt`: Time step in seconds

**Description:**
Updates the state estimate and covariance matrix based on the motion model.

**Mathematical Model:**
```
x(k+1) = F(dt) * x(k)
P(k+1) = F(dt) * P(k) * F(dt)ᵀ + Q
```

##### update(z: np.ndarray) → tuple

Performs the update step with a new measurement.

**Parameters:**
- `z`: Acceleration measurement (scalar or 1D array)

**Returns:**
- `tuple`: (updated_state, updated_covariance, innovation)

**Description:**
Incorporates a new acceleration measurement to refine the state estimate.

**Mathematical Model:**
```
y = z - H * x          # Innovation
S = H * P * Hᵀ + R     # Innovation covariance
K = P * Hᵀ * S⁻¹       # Kalman gain
x = x + K * y          # State update
P = (I - K * H) * P    # Covariance update
```

### EKFPerformanceMonitor

Comprehensive performance monitoring and analysis system.

#### Constructor

```python
EKFPerformanceMonitor(config: dict, logger: logging.Logger)
```

**Parameters:**
- `config`: Configuration dictionary with performance monitoring settings
- `logger`: Logger instance for recording events

**Configuration Keys:**
- `performance_monitoring.max_trace`: Maximum allowable covariance trace
- `performance_monitoring.nees_upper_bound`: Upper bound for NEES test
- `performance_monitoring.nis_upper_bound`: Upper bound for NIS test
- `performance_monitoring.innovation_correlation_threshold`: Innovation whiteness threshold

#### Methods

##### update_metrics(ekf, innovation, measurement, true_state=None) → None

Updates performance metrics with current filter state.

**Parameters:**
- `ekf`: ExtendedKalmanFilter instance
- `innovation`: Current innovation vector
- `measurement`: Current measurement
- `true_state` (optional): True state for NEES calculation

**Metrics Updated:**
- Covariance trace history
- Innovation magnitude history
- Log-likelihood history
- NIS (Normalized Innovation Squared)
- NEES (Normalized Estimation Error Squared) if true state provided

##### check_convergence(window_size=50) → bool

Analyzes filter convergence status.

**Parameters:**
- `window_size`: Number of recent samples to analyze

**Returns:**
- `bool`: True if filter has converged

**Algorithm:**
```python
recent_traces = trace_history[-window_size:]
trace_variance = np.var(recent_traces)
mean_trace = np.mean(recent_traces)
relative_variance = trace_variance / (mean_trace + 1e-10)
is_converged = relative_variance < 0.01  # 1% threshold
```

##### check_filter_consistency() → dict

Performs statistical consistency tests.

**Returns:**
- `dict`: Results of NEES and NIS consistency tests

**Tests Performed:**
- **NEES Test**: Checks if estimation errors follow expected distribution
- **NIS Test**: Validates measurement model consistency

**Expected Ranges:**
- NEES: Should be between χ²₀.₀₅(4) and χ²₀.₉₅(4)
- NIS: Should be below χ²₀.₉₅(1)

##### analyze_innovation_whiteness(max_lag=20) → dict

Tests innovation sequence for whiteness (lack of correlation).

**Parameters:**
- `max_lag`: Maximum lag for autocorrelation analysis

**Returns:**
- `dict`: Whiteness test results including correlations

**Algorithm:**
```python
for lag in range(1, max_lag):
    correlation = np.corrcoef(innovations[:-lag], innovations[lag:])[0, 1]
    correlations.append(abs(correlation))

max_correlation = max(correlations)
is_white = max_correlation < threshold
```

##### evaluate_tuning_quality() → dict

Provides recommendations for parameter tuning.

**Returns:**
- `dict`: Tuning recommendations for Q and R parameters

**Logic:**
- High NIS → Increase R (measurement noise)
- Low NIS → Decrease R
- Growing trace → Decrease Q (process noise)
- Rapidly decreasing trace → Increase Q

##### generate_performance_report() → dict

Creates comprehensive performance analysis report.

**Returns:**
- `dict`: Complete performance analysis with all metrics

**Report Sections:**
- Convergence analysis
- Statistical consistency results
- Innovation whiteness assessment
- Parameter tuning recommendations
- Numerical statistics

## 🔧 Utility Functions

### Data Processing

#### load_data(file_path: str) → pd.DataFrame

Loads acceleration data from text file.

**Parameters:**
- `file_path`: Path to input data file

**Returns:**
- `pd.DataFrame`: DataFrame with columns [timestamp, acc_x, acc_y, acc_z]

**Expected Format:**
```
timestamp   acc_x   acc_y   acc_z
0.000      9.810   0.120   0.450
0.010      9.830   0.150   0.430
```

#### resample_data(data: pd.DataFrame, target_frequency: float = 100) → pd.DataFrame

Resamples data to uniform frequency.

**Parameters:**
- `data`: Input DataFrame
- `target_frequency`: Target sampling rate in Hz

**Returns:**
- `pd.DataFrame`: Resampled data at uniform intervals

**Method:**
- Linear interpolation between existing samples
- Uniform time grid generation
- Extrapolation for boundary conditions

#### detect_stationary_periods(data, acc_column, window_size=50, threshold=0.1) → np.ndarray

Identifies periods when device is stationary for ZUPT application.

**Parameters:**
- `data`: DataFrame with acceleration data
- `acc_column`: Column name for acceleration
- `window_size`: Size of analysis window
- `threshold`: Variance threshold for stationary detection

**Returns:**
- `np.ndarray`: Boolean array indicating stationary periods

**Algorithm:**
```python
for each sample:
    window = acceleration[i-window_size//2 : i+window_size//2]
    variance = np.var(window)
    is_stationary[i] = variance < threshold
```

#### correct_velocity_drift(timestamps, velocities, positions, polynomial_order=2) → tuple

Removes polynomial drift from velocity estimates.

**Parameters:**
- `timestamps`: Time vector
- `velocities`: Velocity estimates
- `positions`: Position estimates  
- `polynomial_order`: Order of drift polynomial

**Returns:**
- `tuple`: (corrected_velocities, corrected_positions)

**Method:**
1. Fit polynomial to velocity time series
2. Remove polynomial trend
3. Re-integrate for corrected positions

### Visualization

#### generate_performance_plots(performance_monitor, output_dir, axis) → None

Creates comprehensive performance visualization.

**Parameters:**
- `performance_monitor`: EKFPerformanceMonitor instance
- `output_dir`: Output directory path
- `axis`: Analysis axis identifier

**Generated Plots:**
1. **Covariance Trace Evolution**: Filter uncertainty over time
2. **Innovation Evolution**: Measurement prediction errors
3. **Innovation Distribution**: Histogram with statistics
4. **NIS Evolution**: Measurement consistency metrics

#### save_performance_report_to_file(report, monitor, output_dir, axis, format='yaml') → None

Saves performance report to file.

**Parameters:**
- `report`: Performance report dictionary
- `monitor`: EKFPerformanceMonitor instance
- `output_dir`: Output directory
- `axis`: Analysis axis
- `format`: Output format ('yaml', 'json', 'txt')

**Output Files:**
- YAML: Structured data format
- JSON: Web-friendly format  
- TXT: Human-readable report

### Configuration

#### load_config(config_path: str) → dict

Loads configuration from YAML file.

**Parameters:**
- `config_path`: Path to configuration file

**Returns:**
- `dict`: Parsed configuration parameters

#### find_config_file() → str

Automatically locates configuration file in standard locations.

**Search Order:**
1. Current directory
2. `configs/` subdirectory
3. Parent directory configs
4. Module configs directory

**Returns:**
- `str`: Path to found configuration file

### Logging

#### setup_enhanced_logging() → tuple

Initializes comprehensive logging system.

**Returns:**
- `tuple`: (logger_instance, log_file_path)

**Features:**
- Timestamped log files
- Multiple log levels
- Console and file output
- Structured formatting

## 📊 Data Structures

### State Vector

4-element numpy array representing system state:
```python
x = np.array([
    [position],     # meters
    [velocity],     # meters/second  
    [acceleration], # meters/second²
    [bias]         # meters/second² (accelerometer bias)
])
```

### Covariance Matrix

4×4 symmetric matrix representing state uncertainty:
```python
P = np.array([
    [σ²_p,  σ_pv, σ_pa, σ_pb],
    [σ_pv,  σ²_v, σ_va, σ_vb],
    [σ_pa,  σ_va, σ²_a, σ_ab],
    [σ_pb,  σ_vb, σ_ab, σ²_b]
])
```

### Configuration Dictionary

Nested dictionary structure for all parameters:
```python
config = {
    'acceleration_type': str,
    'analysis_axis': str,
    'input_files': {
        'linear': str,
        'uncalibrated': str
    },
    'kalman_filter': {
        'initial_state': list[float],
        'initial_covariance': list[float],
        'process_noise': {
            'position': float,
            'velocity': float,
            'acceleration': float,
            'bias': float
        },
        'measurement_noise': float,
        'gravity': float
    },
    'performance_monitoring': {
        'enabled': bool,
        'max_trace': float,
        'nees_upper_bound': float,
        'nis_upper_bound': float,
        # ... additional parameters
    }
}
```

### Performance Report

Structured dictionary containing analysis results:
```python
report = {
    'convergence': {
        'is_converged': bool,
        'convergence_time': int,
        'current_trace': float
    },
    'consistency': {
        'nees_consistent': bool,
        'nis_consistent': bool,
        'overall_consistent': bool
    },
    'innovation_whiteness': {
        'is_white': bool,
        'max_correlation': float,
        'correlations': list[float]
    },
    'tuning': {
        'Q_adjustment': str,  # 'increase', 'decrease', 'none'
        'R_adjustment': str,
        'overall_quality': str  # 'good', 'suboptimal', 'poor'
    },
    'trace_statistics': {
        'mean_trace': float,
        'std_trace': float,
        'min_trace': float,
        'max_trace': float
    }
}
```

## 🎯 Usage Examples

### Basic Filter Usage

```python
# Load configuration
config = load_config('config.yaml')

# Create filter
ekf = ExtendedKalmanFilter(config)

# Initialize performance monitor
monitor = EKFPerformanceMonitor(config, logger)

# Process measurements
for i, measurement in enumerate(measurements):
    dt = timestamps[i] - timestamps[i-1]
    
    # Prediction step
    ekf.predict(dt)
    
    # Update step
    state, covariance, innovation = ekf.update(measurement)
    
    # Monitor performance
    monitor.update_metrics(ekf, innovation, measurement)
    
    # Check convergence periodically
    if i % 100 == 0:
        converged = monitor.check_convergence()

# Generate final report
report = monitor.generate_performance_report()
```

### Custom Parameter Tuning

```python
# Start with baseline parameters
config['kalman_filter']['process_noise']['velocity'] = 0.1

# Run analysis
results = apply_extended_kalman_filter(data, config, 'Y')

# Check performance
monitor = EKFPerformanceMonitor(config, logger)
# ... process data ...
report = monitor.generate_performance_report()

# Adjust based on recommendations
if report['tuning']['Q_adjustment'] == 'decrease':
    config['kalman_filter']['process_noise']['velocity'] *= 0.5
    
if report['tuning']['R_adjustment'] == 'increase':
    config['kalman_filter']['measurement_noise'] *= 1.5

# Re-run with optimized parameters
optimized_results = apply_extended_kalman_filter(data, config, 'Y')
```

This API reference provides complete documentation for all classes, methods, and data structures in the EKF module, enabling developers to effectively use and extend the system.
