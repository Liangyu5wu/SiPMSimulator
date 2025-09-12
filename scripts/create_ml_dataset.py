#!/usr/bin/env python3
"""
Create ML training dataset from merged scan results with filtering
"""

import numpy as np
import argparse
from pathlib import Path

def create_ml_dataset(input_file="../output/merged_scan_results.npz", 
                     output_file="../output/ml_training_dataset.npz",
                     max_photons=95):
    """Create filtered ML dataset from merged scan results"""
    
    print(f"Loading data from: {input_file}")
    data = np.load(input_file)
    
    # Load original data
    waveforms = data['waveforms']
    photon_times = data['photon_times_detected_jittered']
    time_axis = data['time_axis']
    
    # Get waveform time window
    t_min, t_max = time_axis[0], time_axis[-1]
    print(f"Waveform time window: [{t_min:.1f}, {t_max:.1f}] ns")
    
    # Step 1: Filter photons within time window
    valid_mask = ~np.isnan(photon_times)
    time_window_mask = (photon_times >= t_min) & (photon_times <= t_max)
    in_window_mask = valid_mask & time_window_mask
    
    # Create filtered photon times (set out-of-window to NaN)
    filtered_photon_times = photon_times.copy()
    filtered_photon_times[~in_window_mask] = np.nan
    
    # Step 2: Count in-window photons per event and filter high-photon events
    photon_counts = in_window_mask.sum(axis=1)
    event_filter = photon_counts < max_photons
    
    # Apply event filtering
    final_waveforms = waveforms[event_filter]
    final_photon_times = filtered_photon_times[event_filter]
    final_photon_counts = photon_counts[event_filter]
    
    # Statistics
    print(f"\nFiltering results:")
    print(f"  Original events: {len(waveforms)}")
    print(f"  Events removed (≥{max_photons} photons): {(~event_filter).sum()}")
    print(f"  Final events: {len(final_waveforms)}")
    print(f"  Retention rate: {len(final_waveforms)/len(waveforms)*100:.1f}%")
    print(f"  Final photon count range: [{final_photon_counts.min()}, {final_photon_counts.max()}]")
    print(f"  Mean photons per event: {final_photon_counts.mean():.1f} ± {final_photon_counts.std():.1f}")
    
    # Save ML dataset
    np.savez_compressed(output_file,
                       time_axis=time_axis,
                       waveforms=final_waveforms,
                       photon_times_detected_jittered=final_photon_times,
                       # Metadata
                       n_events=len(final_waveforms),
                       time_window=[t_min, t_max],
                       max_photons_filter=max_photons,
                       photon_count_stats=[final_photon_counts.min(), 
                                         final_photon_counts.max(), 
                                         final_photon_counts.mean(), 
                                         final_photon_counts.std()])
    
    # File size comparison
    output_size = Path(output_file).stat().st_size / 1024**2
    input_size = Path(input_file).stat().st_size / 1024**2
    
    print(f"\nDataset saved:")
    print(f"  Output file: {output_file}")
    print(f"  File size: {output_size:.1f} MB (vs {input_size:.1f} MB original)")
    print(f"  Size reduction: {(1-output_size/input_size)*100:.1f}%")
    
    data.close()
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ML training dataset")
    parser.add_argument("--input", default="../output/merged_scan_results.npz", help="Input merged file")
    parser.add_argument("--output", default="../output/ml_training_dataset.npz", help="Output dataset file")
    parser.add_argument("--max-photons", type=int, default=95, help="Maximum photons per event")
    
    args = parser.parse_args()
    create_ml_dataset(args.input, args.output, args.max_photons)