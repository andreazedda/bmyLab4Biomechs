"""
Debug utility to add extensive logging to segmentation and plotting.
Run this with: python debug_segmentation.py
"""

import numpy as np
from time_parameters_finder import (
    analyze_squat_file_auto_axis,
    RepSegments,
)

# Monkey-patch with debug version
def debug_segment_repetitions_with_segments(
    original_func, t, phases, min_phase_duration=0.1
):
    """Wrapper that adds debug logging to segment_repetitions_with_segments"""
    from collections import Counter
    from time_parameters_finder import Phase, RepTime
    
    print("\n" + "="*80)
    print("🔍 DEBUG: segment_repetitions_with_segments STARTED")
    print("="*80)
    
    phases_array = np.array(phases)
    N = len(phases_array)
    
    print(f"Total samples: {N}")
    print(f"Time range: {t[0]:.3f}s to {t[-1]:.3f}s (duration: {t[-1]-t[0]:.3f}s)")
    print(f"Min phase duration: {min_phase_duration}s")
    
    # Count phase distribution
    phase_counts = Counter(phases_array)
    print(f"\nPhase distribution:")
    for phase_val in [Phase.STAND, Phase.ECC, Phase.BOTTOM, Phase.CONC, Phase.UNKNOWN]:
        count = phase_counts.get(phase_val, 0)
        pct = 100.0 * count / N if N > 0 else 0.0
        print(f"  {phase_val.name:8s}: {count:6d} samples ({pct:5.1f}%)")
    
    # Call original function
    reps, segments = original_func(t, phases, min_phase_duration)
    
    print(f"\n" + "="*80)
    print(f"✅ SEGMENTATION COMPLETE: Found {len(reps)} valid repetitions")
    print("="*80)
    
    # Log each segment
    for i, (rep, seg) in enumerate(zip(reps, segments)):
        print(f"\n  Rep #{i+1}:")
        print(f"    ECC:    [{seg.t_ecc_start:7.3f}, {seg.t_ecc_end:7.3f}] Δ={rep.T_ecc:.3f}s")
        print(f"    BOTTOM: [{seg.t_bottom_start:7.3f}, {seg.t_bottom_end:7.3f}] Δ={rep.T_buca:.3f}s")
        print(f"    CONC:   [{seg.t_conc_start:7.3f}, {seg.t_conc_end:7.3f}] Δ={rep.T_con:.3f}s")
        print(f"    STAND:  [{seg.t_top_start:7.3f}, {seg.t_top_end:7.3f}] Δ={rep.T_top:.3f}s")
        total = seg.t_conc_end - seg.t_ecc_start + (seg.t_top_end - seg.t_top_start)
        print(f"    Total rep duration: {total:.3f}s")
    
    print("="*80 + "\n")
    return reps, segments


def debug_plot_by_rep(t, theta, rep_segments, title, save_path, show):
    """Debug version of plot_squat_phases_by_rep with extensive logging"""
    import matplotlib.pyplot as plt
    
    print("\n" + "="*80)
    print("🎨 DEBUG: plot_squat_phases_by_rep STARTED")
    print("="*80)
    print(f"Time array: {len(t)} samples, range [{t[0]:.3f}, {t[-1]:.3f}]s")
    print(f"Theta array: {len(theta)} samples, range [{theta.min():.1f}, {theta.max():.1f}]°")
    print(f"Number of rep_segments to plot: {len(rep_segments)}")
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    # Plot signal
    ax.plot(t, theta, "k-", linewidth=0.8, label="Angle", zorder=10)
    print(f"\nPlotted signal line")

    # Color repetition blocks
    print(f"\nAdding colored blocks for {len(rep_segments)} repetitions:")
    for i, seg in enumerate(rep_segments):
        print(f"\n  Rep #{i+1}:")
        
        # ECC
        ax.axvspan(seg.t_ecc_start, seg.t_ecc_end, color="#FF4444", alpha=0.3, zorder=1)
        print(f"    ECC:    [{seg.t_ecc_start:7.3f}, {seg.t_ecc_end:7.3f}] Δ={seg.t_ecc_end-seg.t_ecc_start:.3f}s")
        
        # BOTTOM
        ax.axvspan(seg.t_bottom_start, seg.t_bottom_end, color="#FFA500", alpha=0.3, zorder=1)
        print(f"    BOTTOM: [{seg.t_bottom_start:7.3f}, {seg.t_bottom_end:7.3f}] Δ={seg.t_bottom_end-seg.t_bottom_start:.3f}s")
        
        # CONC
        ax.axvspan(seg.t_conc_start, seg.t_conc_end, color="#44FF44", alpha=0.3, zorder=1)
        print(f"    CONC:   [{seg.t_conc_start:7.3f}, {seg.t_conc_end:7.3f}] Δ={seg.t_conc_end-seg.t_conc_start:.3f}s")
        
        # TOP (STAND)
        ax.axvspan(seg.t_top_start, seg.t_top_end, color="#808080", alpha=0.3, zorder=1)
        print(f"    STAND:  [{seg.t_top_start:7.3f}, {seg.t_top_end:7.3f}] Δ={seg.t_top_end-seg.t_top_start:.3f}s")
        
        total_rep_time = seg.t_conc_end - seg.t_ecc_start
        print(f"    Total work time: {total_rep_time:.3f}s")

    # Create dummy handles for legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color='k', label='Angle'),
        Patch(color='#FF4444', alpha=0.3, label='↓ ECC'),
        Patch(color='#FFA500', alpha=0.3, label='● BOTTOM'),
        Patch(color='#44FF44', alpha=0.3, label='↑ CONC'),
        Patch(color='#808080', alpha=0.3, label='STAND'),
    ]
    
    ax.legend(handles=handles, loc="upper right", ncol=5, framealpha=0.9)
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Angle [deg]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"\n" + "="*80)
    print(f"✅ PLOT COMPLETE")
    if save_path:
        print(f"📁 Saving to: {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved successfully")
    if show:
        print(f"👁️  Displaying plot...")
        plt.show()
    plt.close()
    print("="*80 + "\n")


if __name__ == "__main__":
    import time_parameters_finder as tpf
    
    # Monkey-patch just the plot function, not segmentation (too complex with different signatures)
    tpf.plot_squat_phases_by_rep = debug_plot_by_rep
    
    print("="*80)
    print("🚀 RUNNING WITH DEBUG LOGGING ENABLED")
    print("="*80 + "\n")
    
    # Run analysis
    path = 'FILE_EULER2025-10-28-10-19-35.txt'
    metrics = tpf.analyze_squat_file_auto_axis(
        path=path,
        smooth_window=51,
        adaptive=True,
        expected_reps=10,
        make_plot=True,
        plot_path="output/debug_plot.png",
    )
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"Found {metrics['N_rep']} repetitions")
    print(f"Parameters: vel_thresh={metrics['parameters']['vel_thresh']:.1f}, min_dur={metrics['parameters']['min_phase_duration']:.2f}")
    if 'optimization' in metrics:
        print(f"Score: {metrics['optimization']['best_score']:.2f}")
