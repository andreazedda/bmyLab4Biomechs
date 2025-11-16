import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Quick diagnostic to understand the data
path = "/Volumes/nvme/Github/bmyLab4Biomechs/sources/modules/time_parameters_finder/FILE_EULER2025-10-28-10-19-35.txt"

df = pd.read_csv(path, header=None, names=["ts_ms", "ang1", "ang2", "ang3"])
df = df.drop_duplicates(subset="ts_ms").sort_values("ts_ms").reset_index(drop=True)

t0 = df["ts_ms"].iloc[0]
t = (df["ts_ms"] - t0).to_numpy(dtype=float) / 1000.0

ang1 = df["ang1"].to_numpy()
ang2 = df["ang2"].to_numpy()
ang3 = df["ang3"].to_numpy()

print(f"Total samples: {len(df)}")
print(f"Duration: {t[-1]:.2f} seconds")
print(f"\nAngle statistics:")
print(f"ang1: min={ang1.min():.2f}, max={ang1.max():.2f}, range={ang1.max()-ang1.min():.2f}, std={ang1.std():.2f}")
print(f"ang2: min={ang2.min():.2f}, max={ang2.max():.2f}, range={ang2.max()-ang2.min():.2f}, std={ang2.std():.2f}")
print(f"ang3: min={ang3.min():.2f}, max={ang3.max():.2f}, range={ang3.max()-ang3.min():.2f}, std={ang3.std():.2f}")

# Plot all three angles
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(t, ang1, label="ang1", linewidth=0.8)
axes[0].set_ylabel("ang1 [deg]")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(t, ang2, label="ang2", color="orange", linewidth=0.8)
axes[1].set_ylabel("ang2 [deg]")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(t, ang3, label="ang3", color="green", linewidth=0.8)
axes[2].set_ylabel("ang3 [deg]")
axes[2].set_xlabel("Time [s]")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig("debug_angles.png", dpi=150)
print("\n✓ Plot saved to debug_angles.png")

# Check first/last second statistics
window_sec = 1.0
start_mask = t <= window_sec
end_mask = t >= (t[-1] - window_sec)

print(f"\nFirst {window_sec}s statistics:")
print(f"  ang1: mean={ang1[start_mask].mean():.2f}, std={ang1[start_mask].std():.2f}")
print(f"  ang2: mean={ang2[start_mask].mean():.2f}, std={ang2[start_mask].std():.2f}")
print(f"  ang3: mean={ang3[start_mask].mean():.2f}, std={ang3[start_mask].std():.2f}")

print(f"\nLast {window_sec}s statistics:")
print(f"  ang1: mean={ang1[end_mask].mean():.2f}, std={ang1[end_mask].std():.2f}")
print(f"  ang2: mean={ang2[end_mask].mean():.2f}, std={ang2[end_mask].std():.2f}")
print(f"  ang3: mean={ang3[end_mask].mean():.2f}, std={ang3[end_mask].std():.2f}")
