# Performance Tuning Guide: EKF Optimization

## 🎯 Overview

This guide provides systematic approaches for optimizing Extended Kalman Filter parameters to achieve optimal performance for velocity and position estimation from smartphone acceleration data.

## 📊 Understanding Performance Metrics

### Key Performance Indicators

#### 1. Convergence Metrics
- **Convergence Time**: Iterations needed to reach steady-state
- **Trace Stability**: Variance of covariance trace in recent window
- **Target**: Convergence within 200-500 iterations, stable trace

#### 2. Statistical Consistency
- **NEES (Normalized Estimation Error Squared)**: 0.71 < NEES < 15.5
- **NIS (Normalized Innovation Squared)**: NIS < 6.63
- **Target**: Both tests should pass consistently

#### 3. Innovation Quality
- **Whiteness**: Maximum autocorrelation < 0.1
- **Distribution**: Should be approximately Gaussian
- **Target**: Uncorrelated, zero-mean innovations

#### 4. Parameter Quality
- **Q Matrix**: Process noise tuning quality
- **R Parameter**: Measurement noise tuning quality
- **Target**: "Good" overall quality rating

## 🔧 Systematic Tuning Methodology

### Phase 1: Initial Assessment

1. **Run Baseline Analysis**
```bash
python sources/EKF_for_vel_and_pos_est_from_acc.py
```

2. **Review Performance Report**
Check the generated `EKF_performance_report_axis_[axis].yaml`:
```yaml
convergence:
  is_converged: false
  current_trace: 6246281.92

tuning:
  Q_adjustment: 'decrease'
  R_adjustment: 'decrease'
  overall_quality: 'poor'
```

3. **Identify Primary Issues**
- Non-convergence → Q too high
- High trace values → Q too high
- Poor innovation whiteness → Model inadequacy or wrong Q/R ratio
- Inconsistent statistics → Q/R balance issues

### Phase 2: Q Matrix Tuning (Process Noise)

#### Understanding Q Parameters

```yaml
process_noise:
  position: 0.001      # How much position can change unexpectedly
  velocity: 0.01       # How much velocity can change unexpectedly
  acceleration: 0.1    # How much acceleration can change unexpectedly
  bias: 0.001          # How much bias can drift
```

#### Q Tuning Strategy

**Step 1: Reduce All Q Values (if diverging)**
```yaml
# Original values
process_noise:
  position: 0.01
  velocity: 0.1
  acceleration: 1.0
  bias: 0.01

# Reduced by 90%
process_noise:
  position: 0.001
  velocity: 0.01
  acceleration: 0.1
  bias: 0.001
```

**Step 2: Fine-tune Individual Components**

| Symptom | Parameter | Action |
|---------|-----------|--------|
| Position too smooth | position | Increase by 2-5x |
| Velocity too smooth | velocity | Increase by 2-5x |
| Acceleration too smooth | acceleration | Increase by 2-5x |
| Bias changes too slowly | bias | Increase by 2-5x |
| Filter diverges | All Q | Decrease by 50% |
| Trace grows exponentially | All Q | Decrease by 80-90% |

**Step 3: Verify Convergence**
- Target: Trace stabilizes within 500 iterations
- Target: Final trace < 1000 (preferably < 100)

### Phase 3: R Parameter Tuning (Measurement Noise)

#### Understanding R Parameter

```yaml
measurement_noise: 0.2  # Variance of acceleration measurement errors
```

#### R Tuning Strategy

**Use NIS Statistics for Guidance:**

```python
# From performance report
nis_statistics:
  mean_nis: 0.105
  percentage_in_bounds: 99.9%
```

**Tuning Rules:**
- **Mean NIS < 0.5**: R too high → Decrease R by 30-50%
- **Mean NIS > 6.0**: R too low → Increase R by 50-100%
- **Mean NIS ≈ 1.0**: R well-tuned
- **% in bounds < 95%**: R too low → Increase R

**Recommended R Values by Data Quality:**
- **High-quality data (clean, low noise)**: R = 0.1 - 0.3
- **Medium-quality data (typical smartphone)**: R = 0.2 - 0.5
- **Low-quality data (noisy, outdoors)**: R = 0.5 - 1.0

### Phase 4: Advanced Optimization

#### Q/R Ratio Optimization

**Target Ratios (approximate guidelines):**
```python
# For typical smartphone data:
Q_velocity / R ≈ 0.05 - 0.2
Q_acceleration / R ≈ 0.5 - 2.0
Q_position / R ≈ 0.005 - 0.02
Q_bias / R ≈ 0.005 - 0.02
```

#### Initial Covariance Tuning

```yaml
# Conservative (slower initial convergence)
initial_covariance: [0.01, 0.0, 0.0, 0.0,
                     0.0, 0.01, 0.0, 0.0,
                     0.0, 0.0, 0.1, 0.0,
                     0.0, 0.0, 0.0, 0.01]

# Aggressive (faster initial convergence)
initial_covariance: [1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 10.0, 0.0,
                     0.0, 0.0, 0.0, 1.0]
```

## 🔄 Iterative Tuning Workflow

### Automated Tuning Approach

```python
def auto_tune_ekf(data, config, max_iterations=10):
    """
    Automated parameter tuning based on performance metrics.
    """
    
    for iteration in range(max_iterations):
        # Run EKF with current parameters
        results = apply_extended_kalman_filter(data, config, 'Y')
        
        # Get performance recommendations
        report = monitor.generate_performance_report()
        
        # Apply recommendations
        if report['tuning']['Q_adjustment'] == 'decrease':
            for param in config['kalman_filter']['process_noise']:
                config['kalman_filter']['process_noise'][param] *= 0.7
                
        elif report['tuning']['Q_adjustment'] == 'increase':
            for param in config['kalman_filter']['process_noise']:
                config['kalman_filter']['process_noise'][param] *= 1.5
        
        if report['tuning']['R_adjustment'] == 'decrease':
            config['kalman_filter']['measurement_noise'] *= 0.7
            
        elif report['tuning']['R_adjustment'] == 'increase':
            config['kalman_filter']['measurement_noise'] *= 1.5
        
        # Check if tuning is complete
        if report['tuning']['overall_quality'] == 'good':
            break
            
        print(f"Iteration {iteration+1}: Quality = {report['tuning']['overall_quality']}")
    
    return config, report
```

### Manual Tuning Workflow

**Step 1: Baseline Run**
```bash
python sources/EKF_for_vel_and_pos_est_from_acc.py
```

**Step 2: Analyze Performance Report**
```yaml
# Check key metrics
convergence: is_converged
tuning: overall_quality, Q_adjustment, R_adjustment
consistency: overall_consistent
innovation_whiteness: is_white
```

**Step 3: Apply First-Order Corrections**
```python
# Based on recommendations
if Q_adjustment == 'decrease':
    multiply_all_Q_by(0.1)  # Aggressive reduction
if R_adjustment == 'decrease':
    R = R * 0.5            # Moderate reduction
```

**Step 4: Iterative Refinement**
```python
for iteration in range(5):
    run_ekf()
    check_metrics()
    apply_fine_tuning()
    if quality == 'good':
        break
```

## 📈 Performance Optimization Strategies

### Strategy 1: Conservative Tuning (Stable but Slow)

```yaml
kalman_filter:
  process_noise:
    position: 0.0001
    velocity: 0.001
    acceleration: 0.01
    bias: 0.0001
  measurement_noise: 0.5
```

**Characteristics:**
- Very stable, no divergence risk
- Slower response to real changes
- Good for noisy data
- Suitable for post-processing applications

### Strategy 2: Aggressive Tuning (Fast but Risky)

```yaml
kalman_filter:
  process_noise:
    position: 0.01
    velocity: 0.1
    acceleration: 1.0
    bias: 0.01
  measurement_noise: 0.1
```

**Characteristics:**
- Fast response to changes
- Risk of instability with poor data
- Good for clean, high-quality data
- Suitable for real-time applications

### Strategy 3: Balanced Tuning (Recommended)

```yaml
kalman_filter:
  process_noise:
    position: 0.001
    velocity: 0.01
    acceleration: 0.1
    bias: 0.001
  measurement_noise: 0.2
```

**Characteristics:**
- Good compromise between stability and responsiveness
- Suitable for most smartphone data
- Robust to moderate noise levels
- Good starting point for tuning

## 🛠️ Troubleshooting Specific Issues

### Issue 1: Filter Divergence

**Symptoms:**
- Exponentially growing covariance trace
- Unrealistic state estimates
- No convergence after many iterations

**Solutions:**
1. Reduce all Q parameters by 90%
2. Increase R parameter by 2-3x
3. Check data for outliers
4. Verify initial state is reasonable

**Example Fix:**
```yaml
# Before (diverging)
process_noise:
  position: 0.1
  velocity: 1.0
  acceleration: 10.0
  bias: 0.1
measurement_noise: 0.1

# After (stable)
process_noise:
  position: 0.001
  velocity: 0.01
  acceleration: 0.1
  bias: 0.001
measurement_noise: 0.3
```

### Issue 2: Poor Innovation Whiteness

**Symptoms:**
- High autocorrelation in innovations (> 0.1)
- Systematic patterns in residuals
- Model inadequacy warnings

**Root Causes:**
1. Missing dynamics in state model
2. Incorrect Q/R ratio
3. Systematic measurement biases
4. Non-Gaussian noise

**Solutions:**
1. **Adjust Q/R ratio**:
```yaml
# Increase process noise relative to measurement noise
process_noise:
  velocity: 0.02      # Increased
  acceleration: 0.2   # Increased
measurement_noise: 0.1  # Decreased
```

2. **Check for systematic biases**:
- Verify gravity compensation
- Check axis alignment
- Look for temperature effects

3. **Consider model extensions**:
- Add jerk (acceleration derivative) to state
- Include environmental factors
- Model non-linear dynamics

### Issue 3: Slow Convergence

**Symptoms:**
- Takes >1000 iterations to converge
- Initial estimates very poor
- Long settling time

**Solutions:**
1. **Optimize initial covariance**:
```yaml
# Reduce initial uncertainty
initial_covariance: [0.01, 0.0, 0.0, 0.0,
                     0.0, 0.01, 0.0, 0.0,
                     0.0, 0.0, 0.1, 0.0,
                     0.0, 0.0, 0.0, 0.01]
```

2. **Improve initial state estimate**:
```yaml
# Use first few measurements for initialization
initial_state: [0.0, 0.0, mean_first_10_acc, 0.0]
```

3. **Increase process noise temporarily**:
```yaml
# Higher Q for first 100 iterations, then reduce
```

### Issue 4: Overly Smooth Estimates

**Symptoms:**
- Estimates lag behind true motion
- Missing rapid changes
- Too much smoothing

**Solutions:**
1. **Increase process noise**:
```yaml
process_noise:
  position: 0.005    # 5x increase
  velocity: 0.05     # 5x increase
  acceleration: 0.5  # 5x increase
```

2. **Decrease measurement noise**:
```yaml
measurement_noise: 0.1  # Reduce to trust measurements more
```

3. **Reduce ZUPT aggressiveness**:
```yaml
zupt:
  threshold: 0.2      # Increase threshold
  window_size: 25     # Reduce window
```

## 📊 Performance Benchmarks

### Target Performance Values

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| Convergence Time | < 200 iter | < 500 iter | < 1000 iter | > 1000 iter |
| Final Trace | < 10 | < 100 | < 1000 | > 1000 |
| NIS Mean | 0.8-1.2 | 0.5-2.0 | 0.2-4.0 | < 0.2 or > 4.0 |
| Innovation Correlation | < 0.05 | < 0.1 | < 0.2 | > 0.2 |
| Consistency Tests | Both pass | Both pass | 1 passes | Both fail |

### Typical Parameter Ranges

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| Q_position | 0.0001 | 0.001 | 0.01 |
| Q_velocity | 0.001 | 0.01 | 0.1 |
| Q_acceleration | 0.01 | 0.1 | 1.0 |
| Q_bias | 0.0001 | 0.001 | 0.01 |
| R | 0.5-1.0 | 0.2-0.5 | 0.1-0.2 |

## 🎯 Best Practices Summary

1. **Start Conservative**: Begin with low Q values to ensure stability
2. **Use Performance Metrics**: Let the system guide your tuning
3. **Iterate Systematically**: Change one parameter type at a time
4. **Validate Results**: Check that estimates make physical sense
5. **Document Settings**: Record successful parameter combinations
6. **Monitor Continuously**: Use real-time diagnostics during tuning
7. **Test Robustness**: Verify performance across different data sets

## 🔍 Advanced Topics

### Adaptive Parameter Tuning

Consider implementing online parameter adaptation:
```python
# Adjust Q based on innovation magnitude
if innovation_magnitude > threshold:
    Q *= adaptation_factor
```

### Multi-Objective Optimization

Balance multiple criteria:
- Estimation accuracy
- Computational efficiency
- Robustness to noise
- Convergence speed

### Cross-Validation

Use separate datasets for:
- Parameter tuning
- Performance validation
- Robustness testing

This comprehensive tuning guide provides the tools and knowledge needed to optimize EKF performance for any smartphone-based motion estimation application.
