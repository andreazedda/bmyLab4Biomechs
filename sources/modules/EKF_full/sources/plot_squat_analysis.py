"""
Generate enhanced plots for complementary filter with squat phases
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    PACKAGE_ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(PACKAGE_ROOT))
    from sources.modules.EKF_full.sources.config import ensure_output_dir
else:
    from .config import ensure_output_dir


def plot_squat_analysis(csv_path: Path):
    """Generate comprehensive squat analysis plots"""
    
    df = pd.read_csv(csv_path)
    session_id = csv_path.stem.replace('_complementary_results', '')
    
    # Phase colors
    phase_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
    phase_names = {0: 'Standing', 1: 'Descending', 2: 'Bottom', 3: 'Ascending'}
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'Squat Analysis - {session_id} (Complementary Filter)', fontsize=16, fontweight='bold')
    
    time = df['time_s'].values
    
    # 1. Position with phases
    ax = axes[0]
    ax.plot(time, df['pos_z_cm'], 'k-', linewidth=2, label='Z (vertical)')
    ax.plot(time, df['pos_y_cm'], 'b-', alpha=0.7, label='Y (forward/back)')
    ax.plot(time, df['pos_x_cm'], 'r-', alpha=0.7, label='X (lateral)')
    
    # Color background by phase
    phases = df['phase'].values
    for phase_id in range(4):
        phase_mask = phases == phase_id
        if np.any(phase_mask):
            phase_times = time[phase_mask]
            ax.axvspan(phase_times[0], phase_times[-1], alpha=0.1, 
                      color=phase_colors[phase_id], label=phase_names[phase_id])
    
    ax.set_ylabel('Position (cm)', fontsize=12)
    ax.set_title('Position During Squats', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 2. Vertical position (Z) with squat depth markers
    ax = axes[1]
    z_pos = df['pos_z_cm'].values
    ax.fill_between(time, 0, z_pos, alpha=0.3, color='purple', label='Squat depth')
    ax.plot(time, z_pos, 'k-', linewidth=2)
    
    # Mark min/max depths
    z_min = np.min(z_pos)
    z_max = np.max(z_pos)
    ax.axhline(z_min, color='red', linestyle='--', linewidth=1.5, label=f'Max depth: {abs(z_min):.1f} cm')
    ax.axhline(0, color='green', linestyle='--', linewidth=1.5, label='Standing')
    
    ax.set_ylabel('Vertical Position (cm)', fontsize=12)
    ax.set_title(f'Squat Depth (Range: {abs(z_min):.1f} cm)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Negative is down
    
    # 3. Orientation angles
    ax = axes[2]
    ax.plot(time, df['pitch_deg'], 'b-', linewidth=1.5, label='Pitch (forward lean)', alpha=0.8)
    ax.plot(time, df['roll_deg'], 'r-', linewidth=1.5, label='Roll (lateral tilt)', alpha=0.8)
    ax.plot(time, df['yaw_deg'], 'g-', linewidth=1.5, label='Yaw (rotation)', alpha=0.6)
    
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_title('Body Orientation', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 4. Squat phases timeline
    ax = axes[3]
    
    # Create color blocks for each phase
    for i in range(len(time) - 1):
        phase = phases[i]
        ax.axvspan(time[i], time[i+1], 
                  color=phase_colors[phase], 
                  alpha=0.6)
    
    # Add phase labels on top
    for phase_id in range(4):
        phase_mask = phases == phase_id
        if np.any(phase_mask):
            mid_time = np.mean(time[phase_mask])
            ax.text(mid_time, 0.5, phase_names[phase_id], 
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Phase', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title('Squat Phase Detection', fontsize=13, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save
    output_dir = ensure_output_dir()
    output_path = output_dir / f'{session_id}_complementary_squat_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[PLOT] Saved squat analysis: {output_path}')
    
    return output_path


if __name__ == "__main__":
    # Find latest complementary results
    output_dir = ensure_output_dir()
    results_files = sorted(output_dir.glob('*_complementary_results.csv'))
    
    if not results_files:
        print("No complementary filter results found")
        sys.exit(1)
    
    latest = results_files[-1]
    print(f"Plotting: {latest}")
    
    plot_path = plot_squat_analysis(latest)
    
    # Open it
    import subprocess
    subprocess.run(['open', str(plot_path)])
