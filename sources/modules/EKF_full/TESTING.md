# EKF_full Module Testing & Validation

## Current Status (2025-11-12)

### Fixed Issues ✅
1. **Attitude propagation bug** - Added proper Euler rate transformation matrix
2. **Merge artifacts** - Restored `_calibrated_gyro()` function
3. **Numerical stability** - Using `np.linalg.solve` instead of `inv`, symmetrizing covariance
4. **Angle plotting** - Unwrapping angles to prevent sawtooth artifacts
5. **NaN handling** - Dropping NaN values before 3D trajectory plotting
6. **Data interpolation** - Excluding Euler angles from linear interpolation

### Remaining Issues ⚠️

#### 1. Excessive Position Drift
**Problem**: Positions grow to hundreds of meters (e.g., -57562 cm = -575 m) over 35 seconds
**Root Cause**: Double integration of accelerometer noise without sufficient constraints
**Expected**: For chest-mounted during squats, vertical oscillation ~30-50cm, horizontal <20cm

#### 2. Insufficient ZUPT Effectiveness
**Current**: ZUPTs applied but not preventing unbounded drift
**Need**: More aggressive position resets during stationary periods

## Recommended Solutions

### Option A: Stricter Biomechanical Constraints
```python
# During ZUPT, reset position to reference
if zupt_count % 2 == 0:  # Every 2nd ZUPT
    ekf.state[:3] = reference_position  # Standing position
    ekf.covariance[:3, :3] *= 0.5  # Reduce position uncertainty
```

### Option B: Relative Position Tracking
Track position **changes** between squat phases instead of absolute positions:
- Detect squat up/down transitions
- Measure displacement between phases
- Reset to zero at standing

### Option C: Sensor Fusion with Height Constraint
For chest-mounted sensor:
- Use barometer if available
- Constrain Z-axis to physiological limits (±1m from standing)
- Use ZUPT to anchor XY plane

## Expected Output Ranges (Chest-Mounted Squats)

| Axis | Expected Range | Current Range | Status |
|------|---------------|---------------|--------|
| X (lateral) | ±10-20 cm | ~57610 cm | ❌ FAIL |
| Y (anterior-posterior) | ±15-30 cm | ~399352 cm | ❌ FAIL |
| Z (vertical) | 30-60 cm oscillation | ~668162 cm | ❌ FAIL |

## Test Data Analysis

### Session: 2025-10-28-10-19-35
- Duration: 35.66 seconds
- Samples: 3568
- Sample rate: ~100 Hz ✅

### Accelerometer Data
```
Mean: [8.7, 3.4, -3.4] m/s² 
Magnitude: ~9.8 m/s² ✅ (correct gravity)
```

### Orientation Data
- Roll: -90° to +30° (likely sagittal plane flexion)
- Pitch: 0° to 50° (forward lean)
- Yaw: -115° to -0.5° (rotation during movement)

## Next Steps

1. **Validate sensor placement** - Confirm phone orientation relative to body
2. **Add physiological constraints** - Limit position to realistic ranges
3. **Implement phase detection** - Use repetition_phases_detector module
4. **Relative tracking** - Measure displacement per repetition cycle
5. **Validate with ground truth** - Compare to video/motion capture if available

## Running Tests

```bash
# Test single session
cd /Volumes/nvme/Github/bmyLab4Biomechs/sources/modules/EKF_full
python -m sources.run_ekf --session 2025-10-28-10-19-35 --plot

# Check results
cat data/outputs/2025-10-28-10-19-35_ekf_results.csv | \
  awk -F',' 'NR>1 {print $3,$4,$5}' | \
  python3 -c "import sys; import statistics as s; data=[list(map(float,l.split())) for l in sys.stdin]; print('X:',s.mean([d[0] for d in data]),'Y:',s.mean([d[1] for d in data]),'Z:',s.mean([d[2] for d in data]))"
```

## Configuration Tuning Log

### Iteration 1 (Initial)
- pos noise: 1e-6, vel noise: 5e-4
- Result: Massive drift (km scale)

### Iteration 2 (Current)
- pos noise: 1e-4, vel noise: 1e-2
- ZUPT thresholds: relaxed for squat motion
- Added drift correction filters
- Result: Still large drift (~500m)

### Iteration 3 (TODO)
- Implement position bounding
- Add per-repetition reset
- Integrate with phase detection
