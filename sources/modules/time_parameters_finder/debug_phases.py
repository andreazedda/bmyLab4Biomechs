from time_parameters_finder import *

path = "/Volumes/nvme/Github/bmyLab4Biomechs/sources/modules/time_parameters_finder/FILE_EULER2025-10-28-10-19-35.txt"

# Load and process
t, angles = load_euler_from_csv(path)
principal_axis = select_principal_axis(angles)
theta = angles[:, principal_axis]

print(f"Principal axis: {principal_axis} ({'ang1' if principal_axis == 0 else 'ang2' if principal_axis == 1 else 'ang3'})")
print(f"Total samples: {len(theta)}")
print(f"Duration: {t[-1]:.2f} seconds")
print(f"Angle range: {theta.min():.2f} to {theta.max():.2f} (range: {theta.max()-theta.min():.2f})")

# Process with different parameters
smooth_window = 7
vel_thresh = 0.1

theta_s = moving_average(theta, smooth_window)
omega = numerical_derivative(t, theta_s)

mu_stand, sigma_stand = estimate_stand_baseline(t, theta_s)
mu_bottom = estimate_bottom_level(theta_s, omega, mu_stand, vel_thresh)

print(f"\nBaselines:")
print(f"  Stand: {mu_stand:.2f} ± {sigma_stand:.2f}")
print(f"  Bottom: {mu_bottom:.2f}")
print(f"  Distance: {abs(mu_bottom - mu_stand):.2f}")

phases, omega = classify_phases(
    t=t,
    theta=theta_s,
    mu_stand=mu_stand,
    mu_bottom=mu_bottom,
    vel_thresh=vel_thresh,
    smooth_window=smooth_window,
)

# Count phases
from collections import Counter
phase_counts = Counter(phases)
print(f"\nPhase distribution:")
for phase, count in phase_counts.items():
    pct = 100 * count / len(phases)
    print(f"  {phase.name}: {count} samples ({pct:.1f}%)")

print(f"\nVelocity stats:")
print(f"  Min: {omega.min():.3f} deg/s")
print(f"  Max: {omega.max():.3f} deg/s")
print(f"  Mean abs: {np.abs(omega).mean():.3f} deg/s")
print(f"  Std: {omega.std():.3f} deg/s")

# Try to detect repetitions
print(f"\n{'='*60}")
print("Trying to detect repetitions with current parameters...")
rep_times = segment_repetitions(t, phases, min_phase_duration=0.12)
print(f"Found {len(rep_times)} repetitions")

if len(rep_times) == 0:
    print("\n⚠️  No repetitions found. Trying with more relaxed parameters...")
    
    # Try with looser constraints
    for vel_thresh_test in [0.2, 0.5, 1.0]:
        phases_test, omega_test = classify_phases(
            t=t,
            theta=theta_s,
            mu_stand=mu_stand,
            mu_bottom=mu_bottom,
            vel_thresh=vel_thresh_test,
            smooth_window=smooth_window,
        )
        
        rep_times_test = segment_repetitions(t, phases_test, min_phase_duration=0.1)
        print(f"  vel_thresh={vel_thresh_test:.1f}: {len(rep_times_test)} reps")
        
        if len(rep_times_test) > 0:
            print(f"    ✓ Success with vel_thresh={vel_thresh_test}")
            phase_counts_test = Counter(phases_test)
            print(f"    Phase distribution:")
            for phase, count in phase_counts_test.items():
                pct = 100 * count / len(phases_test)
                print(f"      {phase.name}: {count} ({pct:.1f}%)")
            break
