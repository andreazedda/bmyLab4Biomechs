# Examples: EKF Usage and Configuration

This directory contains practical examples for using the Extended Kalman Filter module for velocity and position estimation.

## 📁 Example Files

### Configuration Examples

1. **`config_basic.yaml`** - Simple configuration for getting started
2. **`config_high_performance.yaml`** - Optimized parameters for clean data
3. **`config_robust.yaml`** - Conservative parameters for noisy data
4. **`config_multi_axis.yaml`** - Multi-axis analysis configuration

### Python Scripts

1. **`basic_usage.py`** - Simple EKF execution example
2. **`parameter_tuning.py`** - Automated parameter optimization
3. **`performance_analysis.py`** - Detailed performance evaluation
4. **`batch_processing.py`** - Processing multiple files

### Data Examples

1. **`sample_data/`** - Example acceleration data files
2. **`expected_outputs/`** - Reference output files

## 🚀 Quick Start Examples

### Example 1: Basic Usage

```python
# basic_usage.py
import sys
import os
sys.path.append('../sources')

from EKF_for_vel_and_pos_est_from_acc import *

# Load configuration
config = load_config('config_basic.yaml')

# Load your data
data = load_data('sample_data/sample_acceleration.txt')

# Run EKF analysis
results = apply_extended_kalman_filter(data, config, 'Y')

print("Analysis complete!")
print(f"Estimated velocity range: {results['velocity'].min():.3f} to {results['velocity'].max():.3f} m/s")
print(f"Estimated position range: {results['position'].min():.3f} to {results['position'].max():.3f} m")
```

### Example 2: Parameter Optimization

```python
# parameter_tuning.py
import numpy as np
from copy import deepcopy

def optimize_parameters(data, base_config, axis='Y'):
    """
    Automatically optimize EKF parameters based on performance metrics.
    """
    best_config = deepcopy(base_config)
    best_quality = 'poor'
    
    # Q parameter search space
    q_multipliers = [0.1, 0.3, 1.0, 3.0, 10.0]
    r_multipliers = [0.1, 0.3, 1.0, 3.0, 10.0]
    
    for q_mult in q_multipliers:
        for r_mult in r_multipliers:
            # Create test configuration
            test_config = deepcopy(base_config)
            
            # Scale Q parameters
            for param in test_config['kalman_filter']['process_noise']:
                test_config['kalman_filter']['process_noise'][param] *= q_mult
            
            # Scale R parameter
            test_config['kalman_filter']['measurement_noise'] *= r_mult
            
            try:
                # Run EKF
                results = apply_extended_kalman_filter(data, test_config, axis)
                
                # Evaluate performance (simplified)
                monitor = EKFPerformanceMonitor(test_config, logger)
                # ... run monitoring ...
                report = monitor.generate_performance_report()
                
                # Check if this is better
                if report['tuning']['overall_quality'] == 'good':
                    if report['convergence']['is_converged']:
                        best_config = test_config
                        best_quality = 'good'
                        print(f"Found good parameters: Q_mult={q_mult}, R_mult={r_mult}")
                        return best_config
                        
            except Exception as e:
                print(f"Failed with Q_mult={q_mult}, R_mult={r_mult}: {e}")
                continue
    
    return best_config

# Usage
base_config = load_config('config_basic.yaml')
data = load_data('sample_data/sample_acceleration.txt')
optimized_config = optimize_parameters(data, base_config)
```

### Example 3: Batch Processing

```python
# batch_processing.py
import glob
import pandas as pd

def process_multiple_files(input_pattern, config_file, output_dir):
    """
    Process multiple acceleration files with the same configuration.
    """
    config = load_config(config_file)
    results_summary = []
    
    files = glob.glob(input_pattern)
    
    for file_path in files:
        print(f"Processing {file_path}...")
        
        try:
            # Load and process data
            data = load_data(file_path)
            results = apply_extended_kalman_filter(data, config, 'Y')
            
            # Extract summary statistics
            summary = {
                'file': os.path.basename(file_path),
                'duration': data['timestamp'].iloc[-1] - data['timestamp'].iloc[0],
                'samples': len(data),
                'max_velocity': results['velocity'].max(),
                'max_position': results['position'].max(),
                'velocity_std': results['velocity'].std(),
                'position_std': results['position'].std()
            }
            results_summary.append(summary)
            
            # Save individual results
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            results.to_csv(f"{output_dir}/{base_name}_results.csv", index=False)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(f"{output_dir}/batch_summary.csv", index=False)
    
    return summary_df

# Usage
summary = process_multiple_files(
    input_pattern="data/inputs/*.txt",
    config_file="config_robust.yaml",
    output_dir="batch_results"
)
```

## 📋 Configuration Examples

### Basic Configuration (config_basic.yaml)

```yaml
# Basic configuration for learning and testing
acceleration_type: 'linear'
analysis_axis: 'Y'

input_files:
  linear: 'sample_data/sample_acceleration.txt'

output_files:
  velocity: 'outputs/basic_velocity.csv'
  position: 'outputs/basic_position.csv'
  plots: 'outputs/basic_plots.png'

resampling:
  enabled: true
  frequency: 100

signal_trimming:
  enabled: true
  start_offset: 50

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

zupt:
  enabled: true
  window_size: 50
  threshold: 0.1
  min_duration: 5

drift_correction:
  enabled: true
  polynomial_order: 2

performance_monitoring:
  enabled: true
  generate_performance_plots: true
  save_performance_report: true
  convergence_check_interval: 100

visualization:
  show_plots: false
  save_plots: true

debug:
  enable_debug_output: false
  verbose: true
```

### High Performance Configuration (config_high_performance.yaml)

```yaml
# Optimized for clean, high-quality data
acceleration_type: 'linear'
analysis_axis: 'Y'

kalman_filter:
  initial_state: [0.0, 0.0, 0.0, 0.0]
  initial_covariance: [0.01, 0.0, 0.0, 0.0,
                       0.0, 0.01, 0.0, 0.0,
                       0.0, 0.0, 0.1, 0.0,
                       0.0, 0.0, 0.0, 0.01]
  process_noise:
    position: 0.005    # Higher for more responsiveness
    velocity: 0.02     # Higher for more responsiveness
    acceleration: 0.2  # Higher for more responsiveness
    bias: 0.002        # Higher for faster bias adaptation
  measurement_noise: 0.1  # Lower due to clean data
  gravity: 9.81

zupt:
  enabled: true
  window_size: 30      # Smaller window for faster detection
  threshold: 0.05      # Lower threshold for cleaner data
  min_duration: 3      # Shorter minimum duration

performance_monitoring:
  enabled: true
  convergence_check_interval: 50  # More frequent checks
  nees_upper_bound: 12.0          # Tighter bounds for high performance
  nis_upper_bound: 5.0
```

### Robust Configuration (config_robust.yaml)

```yaml
# Conservative parameters for noisy or challenging data
acceleration_type: 'linear'
analysis_axis: 'Y'

kalman_filter:
  initial_state: [0.0, 0.0, 0.0, 0.0]
  initial_covariance: [0.001, 0.0, 0.0, 0.0,
                       0.0, 0.001, 0.0, 0.0,
                       0.0, 0.0, 0.01, 0.0,
                       0.0, 0.0, 0.0, 0.001]
  process_noise:
    position: 0.0001   # Very low for stability
    velocity: 0.001    # Very low for stability
    acceleration: 0.01 # Very low for stability
    bias: 0.0001       # Very low for stability
  measurement_noise: 1.0  # High due to noisy data
  gravity: 9.81

zupt:
  enabled: true
  window_size: 100     # Larger window for robust detection
  threshold: 0.2       # Higher threshold for noisy data
  min_duration: 10     # Longer minimum duration

drift_correction:
  enabled: true
  polynomial_order: 3  # Higher order for complex drift

performance_monitoring:
  enabled: true
  convergence_check_interval: 200  # Less frequent checks
  max_trace: 5000.0               # Higher tolerance
  nees_upper_bound: 20.0          # Looser bounds for robustness
  nis_upper_bound: 10.0
```

## 🔍 Advanced Examples

### Custom Performance Monitoring

```python
# custom_monitoring.py
class CustomPerformanceMonitor(EKFPerformanceMonitor):
    """Extended performance monitor with custom metrics."""
    
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.velocity_smoothness = []
        self.acceleration_consistency = []
    
    def update_custom_metrics(self, ekf, velocity, acceleration):
        """Add custom performance metrics."""
        
        # Velocity smoothness (derivative of velocity)
        if len(self.velocity_smoothness) > 0:
            vel_change = abs(velocity - self.last_velocity)
            self.velocity_smoothness.append(vel_change)
        self.last_velocity = velocity
        
        # Acceleration consistency (compare estimated vs measured)
        if hasattr(self, 'last_acceleration'):
            acc_change = abs(acceleration - self.last_acceleration)
            self.acceleration_consistency.append(acc_change)
        self.last_acceleration = acceleration
    
    def generate_custom_report(self):
        """Generate report with custom metrics."""
        report = super().generate_performance_report()
        
        if self.velocity_smoothness:
            report['velocity_smoothness'] = {
                'mean': np.mean(self.velocity_smoothness),
                'std': np.std(self.velocity_smoothness),
                'max': np.max(self.velocity_smoothness)
            }
        
        if self.acceleration_consistency:
            report['acceleration_consistency'] = {
                'mean': np.mean(self.acceleration_consistency),
                'std': np.std(self.acceleration_consistency)
            }
        
        return report
```

### Comparative Analysis

```python
# compare_configurations.py
def compare_configurations(data, config_files, axis='Y'):
    """Compare performance of different configurations."""
    
    results = {}
    
    for config_file in config_files:
        config = load_config(config_file)
        
        # Run EKF
        ekf_results = apply_extended_kalman_filter(data, config, axis)
        
        # Get performance metrics
        monitor = EKFPerformanceMonitor(config, logger)
        # ... run analysis ...
        report = monitor.generate_performance_report()
        
        results[config_file] = {
            'velocity_range': ekf_results['velocity'].max() - ekf_results['velocity'].min(),
            'position_range': ekf_results['position'].max() - ekf_results['position'].min(),
            'convergence_time': report['convergence']['convergence_time'],
            'final_trace': report['convergence']['current_trace'],
            'overall_quality': report['tuning']['overall_quality'],
            'is_consistent': report['consistency']['overall_consistent']
        }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    return comparison_df

# Usage
configs = [
    'config_basic.yaml',
    'config_high_performance.yaml', 
    'config_robust.yaml'
]

data = load_data('sample_data/sample_acceleration.txt')
comparison = compare_configurations(data, configs)
print(comparison)
```

## 📊 Expected Results

### Performance Benchmarks

| Configuration | Convergence Time | Final Trace | Quality | Use Case |
|---------------|-----------------|-------------|---------|----------|
| Basic | 200-500 iter | 10-100 | Good | General purpose |
| High Performance | 100-300 iter | 5-50 | Excellent | Clean data |
| Robust | 300-800 iter | 50-500 | Good | Noisy data |

### Typical Output Values

For a 30-second squat exercise:
- **Velocity Range**: 0.5 - 2.0 m/s
- **Position Range**: 0.2 - 1.0 m
- **Convergence Time**: 200-500 iterations
- **Final Trace**: < 100

## 🛠️ Running the Examples

1. **Copy example files to your working directory**
2. **Modify paths in configuration files as needed**
3. **Run examples**:

```bash
# Basic usage
python basic_usage.py

# Parameter tuning
python parameter_tuning.py

# Batch processing
python batch_processing.py

# Custom monitoring
python custom_monitoring.py
```

## 📝 Notes

- All examples assume the EKF module is in the parent `sources/` directory
- Modify file paths according to your directory structure
- Examples are provided as starting points - adapt to your specific needs
- For production use, add appropriate error handling and validation

These examples provide practical templates for implementing EKF-based motion estimation in various scenarios and applications.
