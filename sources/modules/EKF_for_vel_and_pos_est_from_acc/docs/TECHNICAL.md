# Technical Documentation: Extended Kalman Filter Implementation

## 🔬 Mathematical Foundation

### State Space Model

The EKF implements a discrete-time state space model for motion estimation:

#### State Vector
```
x = [p, v, a, b]ᵀ
```
Where:
- `p`: position (m)
- `v`: velocity (m/s)
- `a`: acceleration (m/s²)
- `b`: accelerometer bias (m/s²)

#### State Transition Model
```
x(k+1) = F(k) * x(k) + w(k)
```

State transition matrix F:
```
F = [1, dt, 0.5*dt², 0]
    [0, 1,  dt,      0]
    [0, 0,  1,       0]
    [0, 0,  0,       1]
```

#### Process Noise Model
```
Q = diag([σ²_p, σ²_v, σ²_a, σ²_b])
```

#### Measurement Model
```
z(k) = H * x(k) + v(k)
```

Measurement matrix H:
```
H = [0, 0, 1, -1]
```
This models: `measured_acceleration = true_acceleration - bias`

#### Measurement Noise
```
R = σ²_r
```

## 🏭 Implementation Details

### ExtendedKalmanFilter Class

#### Initialization
```python
def __init__(self, config):
    # Extract parameters from config
    kf_config = config['kalman_filter']
    
    # Initialize state vector
    self.x = np.array(kf_config['initial_state']).reshape(4, 1)
    
    # Initialize covariance matrix
    P_flat = kf_config['initial_covariance']
    self.P = np.array(P_flat).reshape(4, 4)
    
    # Process noise matrix
    process_noise = kf_config['process_noise']
    self.Q = np.diag([
        process_noise['position'],
        process_noise['velocity'], 
        process_noise['acceleration'],
        process_noise['bias']
    ])
    
    # Measurement noise
    self.R = np.array([[kf_config['measurement_noise']]])
```

#### Prediction Step
```python
def predict(self, dt):
    # State transition matrix
    F = np.array([
        [1, dt, 0.5*dt**2, 0],
        [0, 1,  dt,        0],
        [0, 0,  1,         0],
        [0, 0,  0,         1]
    ])
    
    # Predict state
    self.x = F @ self.x
    
    # Predict covariance
    self.P = F @ self.P @ F.T + self.Q
```

#### Update Step
```python
def update(self, z):
    # Measurement matrix
    H = np.array([[0, 0, 1, -1]])
    
    # Innovation
    y = z - H @ self.x
    
    # Innovation covariance
    S = H @ self.P @ H.T + self.R
    
    # Kalman gain
    K = self.P @ H.T @ np.linalg.inv(S)
    
    # Update state
    self.x = self.x + K @ y
    
    # Update covariance
    I = np.eye(4)
    self.P = (I - K @ H) @ self.P
    
    return self.x, self.P, y
```

### Zero-Velocity Update (ZUPT)

#### Stationary Period Detection
```python
def detect_stationary_periods(data, acc_column, window_size=50, threshold=0.1):
    # Calculate acceleration magnitude
    acc_magnitude = np.abs(data[acc_column])
    
    # Sliding window variance
    variances = []
    for i in range(len(acc_magnitude)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(acc_magnitude), i + window_size // 2 + 1)
        window_data = acc_magnitude[start_idx:end_idx]
        variances.append(np.var(window_data))
    
    # Identify stationary periods
    is_stationary = np.array(variances) < threshold
    
    return is_stationary
```

#### ZUPT Application
```python
if is_stationary[i]:
    # Force velocity to zero
    ekf.x[1, 0] = 0.0
    
    # Reduce velocity uncertainty
    ekf.P[1, 1] *= 0.1
```

## 📊 Performance Monitoring System

### EKFPerformanceMonitor Class

#### Core Metrics

1. **Covariance Trace Evolution**
```python
trace = np.trace(ekf.P)
self.trace_history.append(trace)
```

2. **Innovation Analysis**
```python
innovation_norm = np.linalg.norm(innovation)
self.innovation_history.append(innovation_norm)
```

3. **Log-Likelihood Calculation**
```python
S = ekf.H @ ekf.P @ ekf.H.T + ekf.R
S_inv = inv(S)
log_likelihood = -0.5 * (innovation.T @ S_inv @ innovation + 
                        np.log(np.linalg.det(2 * np.pi * S)))
```

4. **Normalized Innovation Squared (NIS)**
```python
nis = float(innovation.T @ S_inv @ innovation)
```

5. **Normalized Estimation Error Squared (NEES)**
```python
if true_state is not None:
    estimation_error = ekf.x - true_state
    P_inv = inv(ekf.P)
    nees = float(estimation_error.T @ P_inv @ estimation_error)
```

#### Statistical Tests

##### Convergence Detection
```python
def check_convergence(self, window_size=50):
    if len(self.trace_history) < window_size:
        return False
        
    recent_traces = list(self.trace_history)[-window_size:]
    trace_variance = np.var(recent_traces)
    mean_trace = np.mean(recent_traces)
    
    # Convergence if relative variance is small
    relative_variance = trace_variance / (mean_trace + 1e-10)
    is_converged = relative_variance < 0.01  # 1% threshold
    
    return is_converged
```

##### Innovation Whiteness Test
```python
def analyze_innovation_whiteness(self, max_lag=20):
    innovations = np.array(self.innovation_history)
    correlations = []
    
    for lag in range(1, min(max_lag, len(innovations)//2)):
        if len(innovations) > lag:
            corr = np.corrcoef(innovations[:-lag], innovations[lag:])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
    
    max_correlation = max(correlations) if correlations else 0
    is_white = max_correlation < self.thresholds['innovation_correlation_threshold']
    
    return {'is_white': is_white, 'max_correlation': max_correlation}
```

##### Filter Consistency Check
```python
def check_filter_consistency(self):
    results = {'nees_consistent': True, 'nis_consistent': True}
    
    # NEES test (Chi-squared distribution with DOF = state dimension)
    if len(self.nees_history) > 10:
        mean_nees = np.mean(self.nees_history)
        nees_in_bounds = (self.thresholds['nees_lower_bound'] <= mean_nees <= 
                         self.thresholds['nees_upper_bound'])
        results['nees_consistent'] = nees_in_bounds
    
    # NIS test (Chi-squared distribution with DOF = measurement dimension)
    if len(self.nis_history) > 10:
        mean_nis = np.mean(self.nis_history)
        nis_consistent = mean_nis <= self.thresholds['nis_upper_bound']
        results['nis_consistent'] = nis_consistent
    
    return results
```

#### Tuning Recommendations

##### Q Matrix Adjustment
```python
if len(self.trace_history) > 20:
    trace_trend = np.polyfit(range(len(self.trace_history)), self.trace_history, 1)[0]
    
    if trace_trend > 0.1:
        recommendations['Q_adjustment'] = 'decrease'
        # Trace growing: Q too high
    elif trace_trend < -0.5:
        recommendations['Q_adjustment'] = 'increase'
        # Trace decreasing too fast: Q too low
```

##### R Parameter Adjustment
```python
if len(self.nis_history) > 20:
    mean_nis = np.mean(self.nis_history)
    
    if mean_nis > self.thresholds['nis_upper_bound'] * 1.5:
        recommendations['R_adjustment'] = 'increase'
        # High NIS: R too small
    elif mean_nis < 0.5:
        recommendations['R_adjustment'] = 'decrease'
        # Low NIS: R too large
```

## 🔧 Data Processing Pipeline

### 1. Data Loading
```python
def load_data(file_path):
    data = pd.read_csv(file_path, sep='\s+', header=None)
    data.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
    return data
```

### 2. Resampling
```python
def resample_data(data, target_frequency=100):
    # Create uniform time grid
    time_start, time_end = data['timestamp'].iloc[0], data['timestamp'].iloc[-1]
    dt = 1.0 / target_frequency
    new_timestamps = np.arange(time_start, time_end, dt)
    
    # Interpolate acceleration data
    for axis in ['acc_x', 'acc_y', 'acc_z']:
        f = interpolate.interp1d(data['timestamp'], data[axis], 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
        resampled_data[axis] = f(new_timestamps)
    
    return resampled_data
```

### 3. Signal Trimming
```python
def trim_signal(data, start_offset=50):
    return data.iloc[start_offset:].reset_index(drop=True)
```

### 4. Drift Correction
```python
def correct_velocity_drift(timestamps, velocities, positions, polynomial_order=2):
    # Fit polynomial to velocity
    poly_coeffs = np.polyfit(timestamps, velocities, polynomial_order)
    velocity_trend = np.polyval(poly_coeffs, timestamps)
    
    # Remove trend
    corrected_velocities = velocities - velocity_trend
    
    # Integrate corrected velocities for position
    dt = np.diff(timestamps)
    corrected_positions = np.cumsum(np.concatenate([[0], corrected_velocities[:-1] * dt]))
    
    return corrected_velocities, corrected_positions
```

## 📈 Visualization and Reporting

### Performance Plots Generation
```python
def generate_performance_plots(performance_monitor, output_dir, axis):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Covariance trace evolution
    axes[0, 0].plot(performance_monitor.trace_history)
    axes[0, 0].set_title('Covariance Matrix Trace Evolution')
    
    # 2. Innovation evolution
    axes[0, 1].plot(performance_monitor.innovation_history)
    axes[0, 1].set_title('Innovation Norm Evolution')
    
    # 3. Innovation distribution
    axes[1, 0].hist(performance_monitor.innovation_history, bins=30)
    axes[1, 0].set_title('Innovation Distribution')
    
    # 4. NIS evolution
    axes[1, 1].plot(performance_monitor.nis_history)
    axes[1, 1].axhline(y=performance_monitor.thresholds['nis_upper_bound'], 
                      color='red', linestyle='--')
    axes[1, 1].set_title('Normalized Innovation Squared (NIS)')
```

### Report Generation
```python
def generate_performance_report(performance_monitor):
    report = {
        'convergence': {
            'is_converged': performance_monitor.check_convergence(),
            'convergence_time': performance_monitor.performance_stats['convergence_time'],
            'current_trace': performance_monitor.trace_history[-1] if performance_monitor.trace_history else None
        },
        'consistency': performance_monitor.check_filter_consistency(),
        'innovation_whiteness': performance_monitor.analyze_innovation_whiteness(),
        'tuning': performance_monitor.evaluate_tuning_quality()
    }
    return report
```

## ⚙️ Configuration Schema

### Complete Configuration Structure
```yaml
# Acceleration data type
acceleration_type: 'linear'  # 'linear' or 'uncalibrated'

# Analysis axis
analysis_axis: 'Y'  # 'X', 'Y', 'Z', or 'all'

# File paths
input_files:
  linear: 'data/inputs/FILE_SENSOR_ACCELERATION_LINEAR*.txt'
  uncalibrated: 'data/inputs/FILE_SENSOR_ACCELERATION_UNCALIBRATED*.txt'

output_files:
  velocity: 'data/outputs/estimated_velocity.csv'
  position: 'data/outputs/estimated_position.csv'
  plots: 'data/outputs/ekf_plots.png'

# Resampling parameters
resampling:
  enabled: true
  frequency: 100  # Hz

# Signal processing
signal_trimming:
  enabled: true
  start_offset: 50  # samples

# Kalman filter parameters
kalman_filter:
  initial_state: [0.0, 0.0, 0.0, 0.0]
  initial_covariance: [0.1, 0.0, 0.0, 0.0,
                       0.0, 0.1, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 0.0, 0.1]
  process_noise:
    position: 0.001
    velocity: 0.01
    acceleration: 0.1
    bias: 0.001
  measurement_noise: 0.2
  gravity: 9.81

# ZUPT parameters
zupt:
  enabled: true
  window_size: 50
  threshold: 0.1
  min_duration: 5

# Drift correction
drift_correction:
  enabled: true
  polynomial_order: 2

# Performance monitoring
performance_monitoring:
  enabled: true
  max_trace: 1000.0
  min_log_likelihood: -1000.0
  nees_upper_bound: 15.5
  nees_lower_bound: 0.71
  nis_upper_bound: 6.63
  innovation_correlation_threshold: 0.1
  convergence_check_interval: 100
  generate_performance_plots: true
  save_performance_report: true
  performance_report_format: 'yaml'

# Visualization
visualization:
  show_plots: true
  save_plots: true
  plot_title: 'EKF Estimation from Android Acceleration Data'

# Debug settings
debug:
  enable_debug_output: false
  verbose: true
```

## 🔍 Error Handling and Debugging

### Common Error Scenarios

1. **Matrix Singularity**
```python
try:
    S_inv = inv(S)
except np.linalg.LinAlgError:
    logger.warning("Singular matrix S detected")
    S_inv = np.linalg.pinv(S)  # Use pseudo-inverse
```

2. **Numerical Instability**
```python
if np.trace(self.P) > max_trace_threshold:
    logger.warning("Covariance matrix trace too large, resetting")
    self.P = np.eye(4) * initial_variance
```

3. **Data Quality Issues**
```python
if np.any(np.isnan(data[acc_column])):
    logger.error("NaN values detected in acceleration data")
    data[acc_column] = data[acc_column].interpolate()
```

### Diagnostic Logging
```python
logger.debug(f"State: pos={self.x[0,0]:.4f}, vel={self.x[1,0]:.4f}")
logger.debug(f"Innovation: {innovation_scalar:.6f}")
logger.debug(f"Trace: {np.trace(self.P):.2f}")
```

This technical documentation provides the mathematical foundation, implementation details, and advanced features of the EKF system for biomechanical analysis.
