#!/usr/bin/env python3
"""
Inspection tool for data files - supports both merged scan results and ML training datasets
"""

import numpy as np
import argparse
from pathlib import Path

def inspect_data_file(file_path):
    """Inspect data file contents - detects file type automatically"""
    file_path = Path(file_path)
    print(f"Inspecting: {file_path}")
    print(f"File size: {file_path.stat().st_size / 1024**2:.1f} MB")
    print("=" * 50)
    
    # Detect file type based on filename patterns
    is_ml_dataset = 'ml_' in file_path.name.lower() or 'training' in file_path.name.lower()
    
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
        
        # Universal summary for both merged and ML datasets
        if 'waveforms' in data:
            wf = data['waveforms']
            print(f"\nSUMMARY:")
            print(f"  Events: {wf.shape[0]}")
            print(f"  Time points: {wf.shape[1]}")
            
            # Analyze photon data (handles both naming conventions)
            photon_key = None
            for key in ['photon_times_detected_jittered', 'photon_times_detected']:
                if key in data:
                    photon_key = key
                    break
                    
            if photon_key:
                pt = data[photon_key]
                valid = ~np.isnan(pt)
                photon_counts = valid.sum(axis=1)
                print(f"  Total photons: {valid.sum()}")
                print(f"  Avg photons/event: {photon_counts.mean():.1f}")
                print(f"  Photon count range: {photon_counts.min()} - {photon_counts.max()}")
                
            # Time axis information
            if 'time_axis' in data:
                t_axis = data['time_axis']
                print(f"  Time range: [{t_axis[0]:.2f}, {t_axis[-1]:.2f}] ns")
                print(f"  Time step: {(t_axis[1] - t_axis[0]):.3f} ns")
            
            # Dataset-specific information
            if is_ml_dataset:
                print(f"\nML DATASET INFO:")
                print(f"  Dataset type: ML Training Dataset")
                if photon_key and photon_counts.size > 0:
                    print(f"  Filtered events: {(photon_counts > 0).sum()}")
                    print(f"  Max photons/event: {photon_counts.max()}")
            else:
                print(f"\nMERGED SCAN INFO:")
                print(f"  Dataset type: Merged Scan Results")
                print(f"  Total events: {data.get('total_events', 'N/A')}")
                print(f"  Scan regions: {data.get('total_scan_regions', 'N/A')}")
                if 'x_ranges' in data and 'y_ranges' in data:
                    x_ranges = data['x_ranges']
                    y_ranges = data['y_ranges']
                    print(f"  X range: [{x_ranges.min():.2f}, {x_ranges.max():.2f}] cm")
                    print(f"  Y range: [{y_ranges.min():.2f}, {y_ranges.max():.2f}] cm")
                    
        data.close()
        
    elif file_path.suffix.lower() in ['.h5', '.hdf5']:
        # HDF5 format
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                def show_item(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"  {name:25} | {str(obj.shape):15} | {obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        print(f"  {name:25} | GROUP")
                print("DATA STRUCTURE:")
                f.visititems(show_item)
                
                # Show metadata if available
                if 'metadata' in f:
                    print(f"\nMETADATA:")
                    meta = f['metadata']
                    for attr_name, attr_value in meta.attrs.items():
                        print(f"  {attr_name}: {attr_value}")
                        
        except ImportError:
            print("h5py not available - cannot inspect HDF5 files")
    else:
        print(f"Unsupported format: {file_path.suffix}")
        
def detect_common_files():
    """Detect and list common data files in output directory"""
    output_dir = Path("../output")
    if not output_dir.exists():
        return []
        
    common_patterns = [
        "*merged_scan_results*",
        "*ml_training_dataset*", 
        "*ml_dataset*"
    ]
    
    found_files = []
    for pattern in common_patterns:
        found_files.extend(output_dir.glob(pattern))
    
    return sorted(found_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect data files (merged results or ML datasets)")
    parser.add_argument("file_path", nargs='?', default=None, 
                       help="Path to data file")
    
    args = parser.parse_args()
    
    if args.file_path is None:
        # Try common files
        for candidate in ["../output/ml_training_dataset.npz", "../output/merged_scan_results.npz"]:
            if Path(candidate).exists():
                args.file_path = candidate
                break
        if args.file_path is None:
            print("No data files found. Specify file path.")
            exit(1)
    
    if Path(args.file_path).exists():
        inspect_data_file(args.file_path)
    else:
        print(f"File not found: {args.file_path}")