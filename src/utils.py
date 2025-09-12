"""
Utility functions for SiPM simulation
"""

import os
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


def load_config(config_path):
    """
    Load YAML configuration file
    
    Args:
        config_path (str): Path to YAML config file
        
    Returns:
        dict: Configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_directory(base_dir):
    """
    Create timestamped output directory structure
    
    Args:
        base_dir (str): Base output directory path
        
    Returns:
        str: Path to created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    
    # Create subdirectories
    (run_dir / "waveforms").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    
    return str(run_dir)


def save_config(config, output_dir):
    """
    Save configuration to output directory
    
    Args:
        config (dict): Configuration parameters
        output_dir (str): Output directory path
    """
    config_file = Path(output_dir) / "metadata" / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def plot_2d_histogram(x_data, y_data, x_bins, y_bins, title, output_path=None):
    """
    Create and save 2D histogram plot
    
    Args:
        x_data (array): X coordinates
        y_data (array): Y coordinates  
        x_bins (array): X bin edges
        y_bins (array): Y bin edges
        title (str): Plot title
        output_path (str, optional): Save path for plot
        
    Returns:
        tuple: (hist, xedges, yedges) from np.histogram2d
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=[x_bins, y_bins])
    
    # Create colormap with white background
    import matplotlib.colors as colors
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('white')
    
    # Mask zero values to show as white
    hist_masked = np.ma.masked_where(hist == 0, hist)
    
    im = ax.imshow(hist_masked.T, origin='lower', aspect='auto',
                   extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                   cmap=cmap)
    
    ax.set_xlabel('X Position (cm)')
    ax.set_ylabel('Y Position (cm)')
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Photon Count')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    
    return hist, xedges, yedges


def save_waveform_h5(waveforms, time_axis, metadata, output_path, 
                    photon_times_original=None, photon_times_jittered=None,
                    photon_times_detected=None, photon_times_detected_jittered=None):
    """
    Save waveforms and photon times to HDF5 file
    
    Args:
        waveforms (array): Waveform data (n_events, n_time_points)
        time_axis (array): Time axis in ns
        metadata (dict): Metadata dictionary
        output_path (str): Output file path
        photon_times_original (array, optional): Original photon times (n_events, 100)
        photon_times_jittered (array, optional): Original photon times with jitter (n_events, 100)
        photon_times_detected (array, optional): QE filtered photon times (n_events, 100)
        photon_times_detected_jittered (array, optional): QE filtered + jitter (n_events, 100)
    """
    with h5py.File(output_path, 'w') as f:
        # Save waveform data
        f.create_dataset('waveforms', data=waveforms, compression='gzip')
        f.create_dataset('time_axis', data=time_axis)
        
        # Save photon time arrays if provided
        if photon_times_original is not None:
            f.create_dataset('photon_times_original', data=photon_times_original, compression='gzip')
        if photon_times_jittered is not None:
            f.create_dataset('photon_times_jittered', data=photon_times_jittered, compression='gzip')
        if photon_times_detected is not None:
            f.create_dataset('photon_times_detected', data=photon_times_detected, compression='gzip')
        if photon_times_detected_jittered is not None:
            f.create_dataset('photon_times_detected_jittered', data=photon_times_detected_jittered, compression='gzip')
        
        # Save metadata
        meta_group = f.create_group('metadata')
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta_group.attrs[key] = value
            elif isinstance(value, (list, tuple)):
                meta_group.attrs[key] = np.array(value)
            else:
                meta_group.attrs[key] = str(value)
    
    print(f"Waveforms saved to: {output_path}")


def save_waveform_npz(waveforms, time_axis, metadata, output_path):
    """
    Save waveforms to NPZ file
    
    Args:
        waveforms (array): Waveform data
        time_axis (array): Time axis in ns
        metadata (dict): Metadata dictionary
        output_path (str): Output file path
    """
    np.savez_compressed(output_path, 
                       waveforms=waveforms,
                       time_axis=time_axis,
                       metadata=metadata)
    
    print(f"Waveforms saved to: {output_path}")


def generate_file_list(config):
    """
    Generate list of ROOT files to process based on config
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        list: List of ROOT file paths
    """
    root_config = config['root_files']
    file_dir = Path(root_config['data_directory'])
    
    if not file_dir.exists():
        raise FileNotFoundError(f"Directory not found: {file_dir}")
    
    pattern = root_config['file_pattern']
    file_list = list(file_dir.glob(pattern))
    
    if not file_list:
        raise FileNotFoundError(f"No ROOT files found matching pattern: {pattern}")
    
    return sorted([str(f) for f in file_list])


def print_simulation_summary(config, n_files, n_total_photons, output_dir):
    """
    Print summary of simulation parameters and results
    
    Args:
        config (dict): Configuration parameters
        n_files (int): Number of ROOT files processed
        n_total_photons (int): Total photons processed
        output_dir (str): Output directory path
    """
    print("\n" + "="*60)
    print("SiPM SIMULATION SUMMARY")
    print("="*60)
    print(f"Files processed: {n_files}")
    print(f"Events processed: {config['simulation']['n_events']}")
    print(f"Total photons: {n_total_photons}")
    print(f"Quantum efficiency: {config['simulation']['quantum_efficiency']:.1%}")
    print(f"Timing jitter: {config['simulation']['timing_jitter']} ns")
    print(f"SiPM area: x{config['photon_filter']['x_range']}, y{config['photon_filter']['y_range']}")
    print(f"Output directory: {output_dir}")
    print("="*60)
