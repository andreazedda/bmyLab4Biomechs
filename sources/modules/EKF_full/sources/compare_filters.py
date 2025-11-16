"""
Side-by-side comparison: EKF vs Complementary Filter
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


def compare_methods(session_id: str):
    """Compare EKF and Complementary Filter results side-by-side"""
    
    output_dir = ensure_output_dir()
    
    ekf_path = output_dir / f'{session_id}_ekf_results.csv'
    comp_path = output_dir / f'{session_id}_complementary_results.csv'
    
    if not ekf_path.exists():
        print(f"EKF results not found: {ekf_path}")
        return
    if not comp_path.exists():
        print(f"Complementary results not found: {comp_path}")
        return
    
    df_ekf = pd.read_csv(ekf_path)
    df_comp = pd.read_csv(comp_path)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'EKF vs Complementary Filter - Session {session_id}', 
                 fontsize=16, fontweight='bold')
    
    time_ekf = df_ekf['time_s'].values
    time_comp = df_comp['time_s'].values
    
    # Row 1: Position X
    ax = axes[0, 0]
    ax.plot(time_ekf, df_ekf['pos_x_cm'], 'r-', linewidth=1.5, label='EKF')
    ax.set_ylabel('X Position (cm)', fontsize=11)
    ax.set_title('EKF: X Position (DRIFTS!)', fontsize=12, fontweight='bold', color='red')
    ax.grid(True, alpha=0.3)
    ekf_x_range = df_ekf['pos_x_cm'].max() - df_ekf['pos_x_cm'].min()
    ax.text(0.02, 0.98, f'Range: {ekf_x_range:.0f} cm', 
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax = axes[0, 1]
    ax.plot(time_comp, df_comp['pos_x_cm'], 'g-', linewidth=1.5, label='Complementary')
    ax.set_ylabel('X Position (cm)', fontsize=11)
    ax.set_title('Complementary: X Position (STABLE!)', fontsize=12, fontweight='bold', color='green')
    ax.grid(True, alpha=0.3)
    comp_x_range = df_comp['pos_x_cm'].max() - df_comp['pos_x_cm'].min()
    ax.text(0.02, 0.98, f'Range: {comp_x_range:.0f} cm', 
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Row 2: Position Z (vertical - most important for squats)
    ax = axes[1, 0]
    ax.plot(time_ekf, df_ekf['pos_z_cm'], 'r-', linewidth=1.5)
    ax.set_ylabel('Z Position (cm)', fontsize=11)
    ax.set_title('EKF: Z Position (Vertical)', fontsize=12, fontweight='bold', color='red')
    ax.grid(True, alpha=0.3)
    ekf_z_range = df_ekf['pos_z_cm'].max() - df_ekf['pos_z_cm'].min()
    ax.text(0.02, 0.98, f'Range: {ekf_z_range:.0f} cm', 
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax = axes[1, 1]
    ax.plot(time_comp, df_comp['pos_z_cm'], 'g-', linewidth=1.5)
    ax.fill_between(time_comp, 0, df_comp['pos_z_cm'], alpha=0.3, color='purple')
    ax.set_ylabel('Z Position (cm)', fontsize=11)
    ax.set_title('Complementary: Z Position (Squat Depth)', fontsize=12, fontweight='bold', color='green')
    ax.grid(True, alpha=0.3)
    comp_z_range = df_comp['pos_z_cm'].max() - df_comp['pos_z_cm'].min()
    ax.text(0.02, 0.98, f'Squat depth: {abs(df_comp["pos_z_cm"].min()):.0f} cm', 
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Row 3: 3D Trajectory comparison
    ax = axes[2, 0]
    ax.plot(df_ekf['pos_x_cm'], df_ekf['pos_z_cm'], 'r-', linewidth=1, alpha=0.6)
    ax.scatter(df_ekf['pos_x_cm'].iloc[0], df_ekf['pos_z_cm'].iloc[0], 
              c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(df_ekf['pos_x_cm'].iloc[-1], df_ekf['pos_z_cm'].iloc[-1], 
              c='red', s=100, marker='X', label='End', zorder=5)
    ax.set_xlabel('X Position (cm)', fontsize=11)
    ax.set_ylabel('Z Position (cm)', fontsize=11)
    ax.set_title('EKF: Side View (X-Z)', fontsize=12, fontweight='bold', color='red')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    ax = axes[2, 1]
    ax.plot(df_comp['pos_x_cm'], df_comp['pos_z_cm'], 'g-', linewidth=2, alpha=0.8)
    ax.scatter(df_comp['pos_x_cm'].iloc[0], df_comp['pos_z_cm'].iloc[0], 
              c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(df_comp['pos_x_cm'].iloc[-1], df_comp['pos_z_cm'].iloc[-1], 
              c='red', s=100, marker='X', label='End', zorder=5)
    ax.set_xlabel('X Position (cm)', fontsize=11)
    ax.set_ylabel('Z Position (cm)', fontsize=11)
    ax.set_title('Complementary: Side View (Squat Pattern)', fontsize=12, fontweight='bold', color='green')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'{session_id}_comparison_ekf_vs_complementary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\n[COMPARISON] Saved: {output_path}')
    
    # Print stats
    print(f'\n{"="*60}')
    print(f'COMPARISON STATISTICS - Session {session_id}')
    print(f'{"="*60}')
    print(f'\n{"Method":<20} {"X Range (cm)":<15} {"Y Range (cm)":<15} {"Z Range (cm)":<15}')
    print(f'{"-"*65}')
    print(f'{"EKF":<20} {ekf_x_range:>12.0f}   {df_ekf["pos_y_cm"].max()-df_ekf["pos_y_cm"].min():>12.0f}   {ekf_z_range:>12.0f}')
    print(f'{"Complementary":<20} {comp_x_range:>12.0f}   {df_comp["pos_y_cm"].max()-df_comp["pos_y_cm"].min():>12.0f}   {comp_z_range:>12.0f}')
    print(f'{"-"*65}')
    print(f'{"Improvement":<20} {ekf_x_range/comp_x_range:>12.0f}x   {(df_ekf["pos_y_cm"].max()-df_ekf["pos_y_cm"].min())/(df_comp["pos_y_cm"].max()-df_comp["pos_y_cm"].min()):>12.0f}x   {ekf_z_range/comp_z_range:>12.0f}x')
    print(f'{"="*60}\n')
    
    return output_path


if __name__ == "__main__":
    import subprocess
    
    session_id = '2025-10-28-10-30-39'
    plot_path = compare_methods(session_id)
    
    if plot_path and plot_path.exists():
        subprocess.run(['open', str(plot_path)])
