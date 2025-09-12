# Extended Kalman Filter for Velocity and Position Estimation

## 📖 Overview

This module implements a comprehensive **Extended Kalman Filter (EKF)** system for estimating velocity and position from smartphone acceleration data during squat exercises. The system includes advanced performance monitoring, automatic parameter tuning recommendations, and real-time diagnostics.

## 🎯 Purpose

The EKF module is designed to:

- **Estimate velocity and position** from noisy acceleration measurements
- **Handle smartphone sensor data** with calibrated and uncalibrated acceleration
- **Apply Zero-Velocity Updates (ZUPT)** to reduce drift during stationary periods
- **Monitor filter performance** in real-time with statistical validation
- **Provide tuning recommendations** for optimal filter parameters
- **Generate comprehensive reports** with performance metrics and visualizations

## 🏗️ Architecture

```
EKF_for_vel_and_pos_est_from_acc/
├── sources/
│   └── EKF_for_vel_and_pos_est_from_acc.py    # Main implementation
├── configs/
│   └── EKF_for_vel_and_pos_est_from_acc.yaml  # Configuration file
├── data/
│   ├── inputs/                                # Input acceleration data
│   └── outputs/                               # Results and reports
├── docs/                                      # Documentation
└── requirements.txt                           # Dependencies
```

## 🔧 Key Components

### 1. ExtendedKalmanFilter Class
- **State Vector**: [position, velocity, acceleration, bias]
- **Prediction Step**: Propagates state using motion model
- **Update Step**: Incorporates acceleration measurements
- **ZUPT Integration**: Applies zero-velocity constraints

### 2. EKFPerformanceMonitor Class
- **Real-time Monitoring**: Tracks filter performance during execution
- **Statistical Tests**: NEES, NIS, innovation whiteness analysis
- **Convergence Detection**: Automated convergence assessment
- **Tuning Recommendations**: Suggests Q and R parameter adjustments

### 3. Data Processing Pipeline
- **Resampling**: Uniform 100Hz sampling
- **Signal Trimming**: Removes noisy startup samples
- **ZUPT Detection**: Identifies stationary periods
- **Drift Correction**: Polynomial velocity drift removal

## 📊 Performance Metrics

### Statistical Validation
- **NEES (Normalized Estimation Error Squared)**: Tests filter consistency
- **NIS (Normalized Innovation Squared)**: Validates measurement model
- **Innovation Whiteness**: Checks for unmodeled dynamics
- **Covariance Trace**: Monitors filter stability

### Quality Indicators
- **Convergence Time**: Time to reach steady-state
- **Tuning Quality**: Overall parameter optimization assessment
- **Filter Consistency**: Statistical reliability validation

## 🚀 Quick Start

### Basic Usage

```python
# Run with default configuration
python EKF_for_vel_and_pos_est_from_acc.py
```

### Configuration

Edit `configs/EKF_for_vel_and_pos_est_from_acc.yaml`:

```yaml
# Analysis settings
acceleration_type: 'linear'  # or 'uncalibrated'
analysis_axis: 'Y'           # or 'X', 'Z', 'all'

# Filter parameters
kalman_filter:
  initial_state: [0.0, 0.0, 0.0, 0.0]
  process_noise:
    position: 0.001
    velocity: 0.01
    acceleration: 0.1
    bias: 0.001
  measurement_noise: 0.2

# Performance monitoring
performance_monitoring:
  enabled: true
  generate_performance_plots: true
  save_performance_report: true
```

## 📈 Output Files

### Results
- `estimated_velocity_[axis].csv`: Velocity time series
- `estimated_position_[axis].csv`: Position time series
- `ekf_plots_[axis].png`: Traditional EKF visualization

### Performance Analysis
- `EKF_performance_analysis_axis_[axis].png`: Performance metrics plots
- `EKF_performance_report_axis_[axis].yaml`: Detailed performance report
- `logs/EKF_execution_[timestamp].log`: Complete execution log

## 🎛️ Parameter Tuning

### Process Noise Matrix (Q)
Controls how much the model trusts its predictions:

- **Higher Q**: More trust in measurements, faster adaptation
- **Lower Q**: More trust in model, smoother estimates

### Measurement Noise (R)
Controls how much the filter trusts measurements:

- **Higher R**: Less trust in measurements, smoother estimates
- **Lower R**: More trust in measurements, faster response

### Auto-Tuning Recommendations

The system automatically provides tuning suggestions:

- **"Decrease Q"**: When covariance trace grows too fast
- **"Decrease R"**: When NIS values are consistently low
- **"Increase Q"**: When filter becomes too rigid
- **"Increase R"**: When NIS values are too high

## 🔍 Performance Monitoring Features

### Real-time Diagnostics
- Progress tracking with percentage completion
- Convergence status updates
- Statistical consistency checks
- Trace monitoring with warnings

### Automated Analysis
- Innovation correlation analysis
- Filter consistency validation
- Parameter quality assessment
- Comprehensive reporting

## 📚 Technical Details

### State Model
```
x(k+1) = F*x(k) + w(k)
```
Where:
- `x = [position, velocity, acceleration, bias]ᵀ`
- `F` = state transition matrix
- `w` = process noise

### Measurement Model
```
z(k) = H*x(k) + v(k)
```
Where:
- `z` = acceleration measurement
- `H = [0, 0, 1, -1]` (acceleration minus bias)
- `v` = measurement noise

### ZUPT Implementation
During stationary periods:
- Force velocity to zero: `x[velocity] = 0`
- Reduce velocity uncertainty: `P[velocity,velocity] *= 0.1`

## 🛠️ Troubleshooting

### Common Issues

1. **Filter Divergence**
   - **Symptoms**: Trace grows exponentially, poor estimates
   - **Solution**: Reduce Q parameters, check measurement quality

2. **Slow Convergence**
   - **Symptoms**: High trace values, long convergence time
   - **Solution**: Optimize initial covariance, adjust Q/R ratio

3. **Correlated Innovations**
   - **Symptoms**: Innovation whiteness test fails
   - **Solution**: Check model adequacy, consider additional states

### Parameter Guidelines

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| Q_position | 0.001 - 0.01 | Position smoothness |
| Q_velocity | 0.01 - 0.1 | Velocity responsiveness |
| Q_acceleration | 0.1 - 1.0 | Acceleration tracking |
| Q_bias | 0.001 - 0.01 | Bias adaptation rate |
| R | 0.1 - 1.0 | Measurement trust |

## 📝 Examples

See the examples in the `examples/` folder for:
- Basic filter usage
- Parameter tuning workflows
- Performance analysis interpretation
- Custom configuration setups

## 🤝 Contributing

When modifying the code:
1. Update the configuration schema if needed
2. Add appropriate logging statements
3. Update performance monitoring if new metrics are added
4. Test with various parameter combinations
5. Update documentation accordingly

## 📄 License

This module is part of the igmSquatBiomechanics project.
