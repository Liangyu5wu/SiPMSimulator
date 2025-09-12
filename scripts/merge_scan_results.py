#!/usr/bin/env python3
"""
Merge SiPM position scan results into a single HDF5 file
Filters out events with zero detected photons for efficiency
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def merge_scan_results(output_dir="../output", output_file="merged_scan_results.h5", 
                      pattern="run_*_x*_y*", min_photons=1):
    """
    Merge all position scan HDF5 files into one, filtering zero-photon events
    
    Args:
        output_dir: Directory containing scan results
        output_file: Output merged file name
        pattern: Directory pattern to match scan results
        min_photons: Minimum detected photons per event (default=1, filters zeros)
    """
    scan_dirs = sorted(Path(output_dir).glob(pattern))
    if not scan_dirs:
        raise FileNotFoundError(f"No scan directories found matching {pattern}")
    
    print(f"Found {len(scan_dirs)} scan directories")
    
    # Collect all valid waveforms and metadata
    all_waveforms, all_photon_data, all_metadata = [], {}, []
    time_axis = None
    
    for scan_dir in tqdm(scan_dirs):
        h5_file = scan_dir / "waveforms" / "sipm_waveforms.h5"
        if not h5_file.exists():
            continue
            
        with h5py.File(h5_file, 'r') as f:
            # Get photon counts for filtering
            detected = f['photon_times_detected'][:]
            valid_mask = np.sum(~np.isnan(detected), axis=1) >= min_photons
            
            if not np.any(valid_mask):
                continue
                
            # Store valid data
            all_waveforms.append(f['waveforms'][:][valid_mask])
            if time_axis is None:
                time_axis = f['time_axis'][:]
                all_photon_data = {k: [] for k in ['photon_times_original', 'photon_times_jittered', 
                                                  'photon_times_detected', 'photon_times_detected_jittered']}
            
            for key in all_photon_data:
                if key in f:
                    all_photon_data[key].append(f[key][:][valid_mask])
            
            # Extract position from directory name
            parts = scan_dir.name.split('_')
            x_part = next(p for p in parts if p.startswith('x'))
            y_part = next(p for p in parts if p.startswith('y'))
            x_range = [float(x_part[1:].split('to')[0].replace('n', '-').replace('p', '.')), 
                      float(x_part[1:].split('to')[1].replace('n', '-').replace('p', '.'))]
            y_range = [float(y_part[1:].split('to')[0].replace('n', '-').replace('p', '.')), 
                      float(y_part[1:].split('to')[1].replace('n', '-').replace('p', '.'))]
            
            all_metadata.extend([{'x_range': x_range, 'y_range': y_range, 'scan_dir': scan_dir.name}] * np.sum(valid_mask))
    
    if not all_waveforms:
        raise ValueError("No valid waveforms found (all events had zero photons)")
    
    # Combine all data
    merged_waveforms = np.vstack(all_waveforms)
    merged_photon_data = {k: np.vstack(v) if v else np.array([]) for k, v in all_photon_data.items()}
    
    # Save merged results
    output_path = Path(output_dir) / output_file
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('waveforms', data=merged_waveforms, compression='gzip')
        f.create_dataset('time_axis', data=time_axis)
        
        for key, data in merged_photon_data.items():
            if data.size > 0:
                f.create_dataset(key, data=data, compression='gzip')
        
        # Metadata
        meta = f.create_group('metadata')
        meta.attrs['total_events'] = len(merged_waveforms)
        meta.attrs['total_scan_regions'] = len(scan_dirs)
        meta.attrs['min_photons_filter'] = min_photons
        
        # Position metadata
        positions = f.create_group('positions')
        positions.create_dataset('x_ranges', data=[[m['x_range'][0], m['x_range'][1]] for m in all_metadata])
        positions.create_dataset('y_ranges', data=[[m['y_range'][0], m['y_range'][1]] for m in all_metadata])
        positions.create_dataset('scan_dirs', data=[m['scan_dir'].encode() for m in all_metadata])
    
    print(f"Merged {len(merged_waveforms)} events from {len(scan_dirs)} scan regions")
    print(f"Output saved: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge SiPM scan results")
    parser.add_argument("--output-dir", default="../output", help="Output directory")
    parser.add_argument("--output-file", default="merged_scan_results.h5", help="Output file name")
    parser.add_argument("--pattern", default="run_*_x*_y*", help="Scan directory pattern")
    parser.add_argument("--min-photons", type=int, default=1, help="Min photons per event")
    
    args = parser.parse_args()
    merge_scan_results(args.output_dir, args.output_file, args.pattern, args.min_photons)