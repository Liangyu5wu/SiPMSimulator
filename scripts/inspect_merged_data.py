#!/usr/bin/env python3
"""
Simple inspection tool for merged scan results
"""

import numpy as np
import argparse
from pathlib import Path

def inspect_data_file(file_path):
    """Inspect merged data file contents"""
    file_path = Path(file_path)
    print(f"Inspecting: {file_path}")
    print(f"File size: {file_path.stat().st_size / 1024**2:.1f} MB")
    print("=" * 50)
    
    if file_path.suffix.lower() == '.npz':
        # NPZ format
        data = np.load(file_path)
        print("DATA ARRAYS:")
        for key in sorted(data.files):
            arr = data[key]
            if hasattr(arr, 'shape'):
                print(f"  {key:25} | {str(arr.shape):15} | {arr.dtype}")
            else:
                print(f"  {key:25} | Value: {arr}")
        
        if 'waveforms' in data:
            wf = data['waveforms']
            print(f"\nSUMMARY:")
            print(f"  Events: {wf.shape[0]}")
            print(f"  Time points: {wf.shape[1]}")
            
        if 'photon_times_detected' in data:
            pt = data['photon_times_detected']
            valid = ~np.isnan(pt)
            print(f"  Total photons: {valid.sum()}")
            print(f"  Avg photons/event: {valid.sum(axis=1).mean():.1f}")
            
        print(f"  Total events: {data.get('total_events', 'N/A')}")
        print(f"  Scan regions: {data.get('total_scan_regions', 'N/A')}")
        data.close()
        
    elif file_path.suffix.lower() in ['.h5', '.hdf5']:
        # HDF5 format
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                def show_item(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"  {name:25} | {str(obj.shape):15} | {obj.dtype}")
                print("DATA STRUCTURE:")
                f.visititems(show_item)
        except ImportError:
            print("h5py not available")
    else:
        print(f"Unsupported format: {file_path.suffix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect merged data file")
    parser.add_argument("file_path", nargs='?', default="../output/merged_scan_results.npz", 
                       help="Path to data file (default: ../output/merged_scan_results.npz)")
    
    args = parser.parse_args()
    if Path(args.file_path).exists():
        inspect_data_file(args.file_path)
    else:
        print(f"File not found: {args.file_path}")