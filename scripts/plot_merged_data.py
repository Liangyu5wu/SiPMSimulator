#!/usr/bin/env python3
"""
Quick plotting tool for merged scan data
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_merged_data(file_path="../output/merged_scan_results.npz"):
    """Plot photon statistics from merged NPZ data"""
    data = np.load(file_path)
    
    # Load data
    photon_times = data['photon_times_detected_jittered']
    time_axis = data['time_axis']
    
    # Get waveform time window
    t_min, t_max = time_axis[0], time_axis[-1]
    
    # Filter photons within waveform time window
    valid_mask = ~np.isnan(photon_times)
    time_window_mask = (photon_times >= t_min) & (photon_times <= t_max)
    in_window_mask = valid_mask & time_window_mask
    
    # 1. Photon counts per waveform (only in-window photons)
    photon_counts = in_window_mask.sum(axis=1)
    
    # 2. All photon times within window
    all_photon_times = photon_times[in_window_mask]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Photon counts histogram (log scale)
    ax1.hist(photon_counts, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Photons per Event (In-Window)')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    ax1.set_title(f'Photon Counts Distribution\n(Mean: {photon_counts.mean():.1f}, Events: {len(photon_counts)})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Photon timing distribution (log scale)
    ax2.hist(all_photon_times, bins=100, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Photon Time (ns)')
    ax2.set_ylabel('Frequency')
    ax2.set_yscale('log')
    ax2.set_title(f'In-Window Photon Timing\n(Range: [{t_min:.1f}, {t_max:.1f}] ns)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../output/merged_data_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    total_valid_photons = valid_mask.sum()
    print(f"Analysis complete:")
    print(f"  Events: {len(photon_counts)}")
    print(f"  Total photons (all): {total_valid_photons}")
    print(f"  In-window photons: {len(all_photon_times)} ({len(all_photon_times)/total_valid_photons*100:.1f}%)")
    print(f"  In-window photons/event: {photon_counts.mean():.1f} ± {photon_counts.std():.1f}")
    print(f"  Waveform time window: [{t_min:.1f}, {t_max:.1f}] ns")
    print(f"  Plot saved: ../output/merged_data_analysis.png")
    
    data.close()

if __name__ == "__main__":
    plot_merged_data()