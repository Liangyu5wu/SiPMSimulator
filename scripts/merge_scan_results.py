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
                      pattern="run_*_x*_y*", min_photons=1, output_format="h5", 
                      chunk_size=None, waveforms_only=False):
    """
    Merge all position scan HDF5 files, with size optimization options
    
    Args:
        output_dir: Directory containing scan results
        output_file: Output merged file name
        pattern: Directory pattern to match scan results
        min_photons: Minimum detected photons per event (default=1, filters zeros)
        output_format: "h5", "npz", or "split" (multiple files)
        chunk_size: Max events per chunk file (for split mode)
        waveforms_only: Save only waveforms, time_axis, and photon timing data (reduces file size ~60%)
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
    
    # Save results with chosen format and optimization
    output_path = Path(output_dir) / output_file
    total_events = len(merged_waveforms)
    
    # Common metadata
    metadata = {
        'total_events': total_events,
        'total_scan_regions': len(scan_dirs),
        'min_photons_filter': min_photons,
        'x_ranges': [[m['x_range'][0], m['x_range'][1]] for m in all_metadata],
        'y_ranges': [[m['y_range'][0], m['y_range'][1]] for m in all_metadata],
        'scan_dirs': [m['scan_dir'] for m in all_metadata]
    }
    
    if output_format == "npz":
        # NPZ format - much smaller, faster loading
        data_dict = {'waveforms': merged_waveforms, 'time_axis': time_axis, **metadata}
        if waveforms_only:
            # Include essential photon timing data
            for key in ['photon_times_detected', 'photon_times_detected_jittered']:
                if key in merged_photon_data and merged_photon_data[key].size > 0:
                    data_dict[key] = merged_photon_data[key]
        else:
            data_dict.update({k: v for k, v in merged_photon_data.items() if v.size > 0})
        np.savez_compressed(output_path.with_suffix('.npz'), **data_dict)
        output_path = output_path.with_suffix('.npz')
        
    elif output_format == "split":
        # Split into chunks to avoid large files
        chunk_size = chunk_size or 10000  # Default 10k events per file
        n_chunks = (total_events + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_events)
            chunk_file = output_path.with_suffix(f'_chunk_{i+1:03d}.h5')
            
            with h5py.File(chunk_file, 'w') as f:
                f.create_dataset('waveforms', data=merged_waveforms[start_idx:end_idx], compression='gzip')
                f.create_dataset('time_axis', data=time_axis)
                
                if waveforms_only:
                    # Include essential photon timing data
                    for key in ['photon_times_detected', 'photon_times_detected_jittered']:
                        if key in merged_photon_data and merged_photon_data[key].size > 0:
                            f.create_dataset(key, data=merged_photon_data[key][start_idx:end_idx], compression='gzip')
                else:
                    for key, data in merged_photon_data.items():
                        if data.size > 0:
                            f.create_dataset(key, data=data[start_idx:end_idx], compression='gzip')
                
                # Metadata for this chunk
                meta = f.create_group('metadata')
                meta.attrs['chunk_events'] = end_idx - start_idx
                meta.attrs['chunk_index'] = i + 1
                meta.attrs['total_chunks'] = n_chunks
                
        print(f"Split into {n_chunks} files with max {chunk_size} events each")
        return f"{len(list(output_path.parent.glob(f'*_chunk_*.h5')))} chunk files created"
        
    else:  # HDF5 format
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('waveforms', data=merged_waveforms, compression='gzip')
            f.create_dataset('time_axis', data=time_axis)
            
            if waveforms_only:
                # Include essential photon timing data
                for key in ['photon_times_detected', 'photon_times_detected_jittered']:
                    if key in merged_photon_data and merged_photon_data[key].size > 0:
                        f.create_dataset(key, data=merged_photon_data[key], compression='gzip')
            else:
                for key, data in merged_photon_data.items():
                    if data.size > 0:
                        f.create_dataset(key, data=data, compression='gzip')
            
            # Metadata
            meta = f.create_group('metadata')
            for key, value in {k: v for k, v in metadata.items() if k not in ['x_ranges', 'y_ranges', 'scan_dirs']}.items():
                meta.attrs[key] = value
                
            if not waveforms_only:
                positions = f.create_group('positions')
                positions.create_dataset('x_ranges', data=metadata['x_ranges'])
                positions.create_dataset('y_ranges', data=metadata['y_ranges'])
                positions.create_dataset('scan_dirs', data=[s.encode() for s in metadata['scan_dirs']])
    
    size_mb = output_path.stat().st_size / 1024**2 if output_path.exists() else 0
    print(f"Merged {total_events} events from {len(scan_dirs)} scan regions")
    print(f"Output saved: {output_path} ({size_mb:.1f} MB)")
    return str(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge SiPM scan results with size optimization")
    parser.add_argument("--output-dir", default="../output", help="Output directory")
    parser.add_argument("--output-file", default="merged_scan_results.h5", help="Output file name")
    parser.add_argument("--pattern", default="run_*_x*_y*", help="Scan directory pattern")
    parser.add_argument("--min-photons", type=int, default=1, help="Min photons per event")
    parser.add_argument("--format", choices=["h5", "npz", "split"], default="h5", 
                       help="Output format: h5 (full), npz (compact), split (chunks)")
    parser.add_argument("--chunk-size", type=int, help="Events per chunk (split mode)")
    parser.add_argument("--waveforms-only", action="store_true", 
                       help="Save only waveforms+time+detected_photon_times (60%% smaller)")
    
    args = parser.parse_args()
    merge_scan_results(args.output_dir, args.output_file, args.pattern, args.min_photons,
                      args.format, args.chunk_size, args.waveforms_only)