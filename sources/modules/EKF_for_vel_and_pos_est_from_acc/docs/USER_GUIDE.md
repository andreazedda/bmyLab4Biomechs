# User Guide: EKF Velocity and Position Estimation

## 🚀 Getting Started

### Prerequisites

1. **Python Environment**: Python 3.8+ with required packages
2. **Data Files**: Android acceleration sensor data in text format
3. **Configuration**: YAML configuration file (provided)

### Installation

```bash
# Navigate to the module directory
cd /path/to/EKF_for_vel_and_pos_est_from_acc

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default settings
python sources/EKF_for_vel_and_pos_est_from_acc.py
```

## 📋 Step-by-Step Workflow

### 1. Prepare Your Data

Your acceleration data should be in text format with columns:
```
timestamp   acc_x   acc_y   acc_z
0.000       9.81    0.12    0.45
0.010       9.83    0.15    0.43
...
```

**Supported formats:**
- Linear acceleration (gravity removed)
- Uncalibrated acceleration (raw sensor data)

### 2. Configure the Analysis

Edit `configs/EKF_for_vel_and_pos_est_from_acc.yaml`:

```yaml
# Choose acceleration type
acceleration_type: 'linear'        # or 'uncalibrated'

# Select analysis axis
analysis_axis: 'Y'                 # 'X', 'Y', 'Z', or 'all'

# Set input file path
input_files:
  linear: 'data/inputs/your_file.txt'
```

### 3. Run the Analysis

```bash
python sources/EKF_for_vel_and_pos_est_from_acc.py
```

### 4. Review Results

The system generates several output files:

#### Estimation Results
- `estimated_velocity_[axis].csv`: Time series of estimated velocity
- `estimated_position_[axis].csv`: Time series of estimated position
- `ekf_plots_[axis].png`: Visualization of results

#### Performance Analysis
- `EKF_performance_analysis_axis_[axis].png`: Performance metrics plots
- `EKF_performance_report_axis_[axis].yaml`: Detailed performance report
- `logs/EKF_execution_[timestamp].log`: Complete execution log

## ⚙️ Configuration Guide

### Essential Parameters

#### Filter Tuning Parameters

```yaml
kalman_filter:
  # Process noise (Q matrix) - How much you trust the model
  process_noise:
    position: 0.001      # Lower = smoother position
    velocity: 0.01       # Lower = smoother velocity
    acceleration: 0.1    # Lower = smoother acceleration
    bias: 0.001          # Lower = slower bias adaptation
  
  # Measurement noise (R) - How much you trust measurements
  measurement_noise: 0.2 # Lower = trust measurements more
```

#### ZUPT (Zero-Velocity Update) Settings

```yaml
zupt:
  enabled: true          # Enable/disable ZUPT
  window_size: 50        # Size of analysis window
  threshold: 0.1         # Variance threshold for stationary detection
  min_duration: 5        # Minimum stationary period length
```

#### Performance Monitoring

```yaml
performance_monitoring:
  enabled: true                           # Enable performance analysis
  generate_performance_plots: true        # Create performance visualizations
  save_performance_report: true          # Save detailed report
  convergence_check_interval: 100        # Check convergence every N iterations
```

### Parameter Tuning Guide

#### When to Adjust Process Noise (Q)

**Increase Q when:**
- Estimates are too smooth/slow to respond
- Real motion is faster than estimates
- System seems "sluggish"

**Decrease Q when:**
- Estimates are too noisy/jittery
- Covariance trace grows too large
- Filter diverges

#### When to Adjust Measurement Noise (R)

**Increase R when:**
- Measurements seem unreliable
- NIS values are consistently high
- Estimates jump around with measurements

**Decrease R when:**
- Measurements are very clean
- NIS values are consistently low
- Filter ignores good measurements

## 📊 Interpreting Results

### Performance Report

The system automatically generates a performance report with key metrics:

#### Convergence Analysis
```yaml
convergence:
  is_converged: true
  convergence_time: 245
  current_trace: 156.789
```
- **is_converged**: Whether filter reached steady-state
- **convergence_time**: Iterations needed to converge
- **current_trace**: Current uncertainty level

#### Statistical Consistency
```yaml
consistency:
  nees_consistent: true
  nis_consistent: true
  overall_consistent: true
```
- **nees_consistent**: State estimation consistency
- **nis_consistent**: Measurement model consistency
- **overall_consistent**: Overall filter reliability

#### Innovation Analysis
```yaml
innovation_whiteness:
  is_white: false
  max_correlation: 0.234
```
- **is_white**: Whether innovations are uncorrelated (good)
- **max_correlation**: Maximum correlation (should be < 0.1)

#### Tuning Quality
```yaml
tuning:
  Q_adjustment: 'none'
  R_adjustment: 'decrease'
  overall_quality: 'good'
```
- **Q_adjustment**: Recommended process noise change
- **R_adjustment**: Recommended measurement noise change
- **overall_quality**: Overall parameter quality assessment

### Performance Plots

The system generates four key plots:

1. **Covariance Trace Evolution**: Shows filter uncertainty over time
2. **Innovation Evolution**: Shows measurement prediction errors
3. **Innovation Distribution**: Histogram of innovation values
4. **NIS Evolution**: Shows measurement consistency over time

### Quality Indicators

#### Good Performance Signs ✅
- Convergence achieved within reasonable time
- Trace stabilizes and doesn't grow
- Innovations are white (uncorrelated)
- NIS values stay within bounds
- NEES values within expected range

#### Warning Signs ⚠️
- No convergence after many iterations
- Exponentially growing trace
- Highly correlated innovations
- Consistently high/low NIS values
- Large bias drift

#### Critical Issues ❌
- Filter divergence (infinite estimates)
- Matrix singularities
- NaN values in estimates
- Extremely high uncertainty

## 🛠️ Troubleshooting

### Common Problems and Solutions

#### Problem: Filter Doesn't Converge
**Symptoms:**
- Trace keeps growing
- Estimates become unrealistic
- Performance report shows "not converged"

**Solutions:**
1. Reduce all Q parameters by 50-90%
2. Check data quality for outliers
3. Verify initial state is reasonable
4. Consider reducing initial covariance

#### Problem: Estimates Too Smooth
**Symptoms:**
- Velocity/position lag behind expected motion
- Filter seems "sluggish"
- Low responsiveness to changes

**Solutions:**
1. Increase Q parameters (especially velocity/acceleration)
2. Decrease R parameter
3. Check if ZUPT is too aggressive
4. Verify sampling rate is adequate

#### Problem: Estimates Too Noisy
**Symptoms:**
- Jittery velocity/position estimates
- High-frequency oscillations
- Unrealistic rapid changes

**Solutions:**
1. Decrease Q parameters
2. Increase R parameter
3. Enable or tune ZUPT parameters
4. Check for measurement outliers

#### Problem: Poor Innovation Whiteness
**Symptoms:**
- Correlated innovations (max correlation > 0.1)
- Performance report shows "not white"

**Solutions:**
1. Check if model captures all dynamics
2. Consider adding more state variables
3. Verify measurement model accuracy
4. Look for systematic measurement errors

### Advanced Troubleshooting

#### Debugging with Logs

Enable detailed logging:
```yaml
debug:
  enable_debug_output: true
  verbose: true
```

Check log file for:
- Matrix singularity warnings
- Unusual innovation values
- Convergence progress
- Parameter update history

#### Custom Diagnostics

Monitor specific metrics during execution:
- Real-time trace values
- Innovation magnitudes
- NIS trends
- Convergence indicators

## 📈 Best Practices

### Data Preparation
1. **Clean your data**: Remove obvious outliers
2. **Check sampling rate**: Ensure consistent timing
3. **Verify axis orientation**: Confirm coordinate system
4. **Validate range**: Check for realistic acceleration values

### Parameter Selection
1. **Start conservative**: Begin with smaller Q values
2. **Tune systematically**: Change one parameter at a time
3. **Use performance metrics**: Let the system guide tuning
4. **Document changes**: Keep track of what works

### Validation
1. **Check convergence**: Ensure filter reaches steady-state
2. **Validate physics**: Verify estimates make physical sense
3. **Compare methods**: Cross-validate with other approaches
4. **Test robustness**: Try different initial conditions

### Reporting
1. **Save configurations**: Document successful parameter sets
2. **Archive results**: Keep performance reports for comparison
3. **Note limitations**: Document any constraints or assumptions
4. **Share insights**: Record lessons learned for future use

## 🔄 Workflow Examples

### Example 1: Basic Analysis
```bash
# 1. Place data file in data/inputs/
# 2. Run with defaults
python sources/EKF_for_vel_and_pos_est_from_acc.py

# 3. Check results in data/outputs/
# 4. Review performance report
```

### Example 2: Parameter Optimization
```bash
# 1. Run initial analysis
python sources/EKF_for_vel_and_pos_est_from_acc.py

# 2. Check performance report recommendations
# 3. Edit config file based on suggestions
# 4. Re-run analysis
python sources/EKF_for_vel_and_pos_est_from_acc.py

# 5. Compare performance metrics
# 6. Iterate until satisfied
```

### Example 3: Multi-Axis Analysis
```yaml
# Set in config file:
analysis_axis: 'all'
```
```bash
# Run analysis for all axes
python sources/EKF_for_vel_and_pos_est_from_acc.py
```

## 🎯 Tips for Success

1. **Start Simple**: Begin with default parameters
2. **Trust the Metrics**: Use performance monitoring guidance
3. **Iterate Carefully**: Make small, systematic changes
4. **Validate Results**: Check that estimates make physical sense
5. **Document Everything**: Keep records of what works
6. **Be Patient**: Good tuning takes time and experimentation

## 📞 Support

For additional help:
1. Check the technical documentation (TECHNICAL.md)
2. Review the performance logs
3. Examine the example configurations
4. Consult the troubleshooting section

Remember: The performance monitoring system is designed to guide you toward optimal parameters. Trust the recommendations and iterate systematically for best results!
