"""
SiPM Simulator Main Class

Simulates Silicon Photomultiplier response to optical photons from 
particle physics experiments using ROOT data input.
"""

import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
import h5py
from tqdm import tqdm

from .utils import (
    setup_output_directory, save_config, plot_2d_histogram,
    save_waveform_h5, save_waveform_npz, generate_file_list,
    print_simulation_summary
)


class SiPMSimulator:
    """
    Main SiPM simulation class
    
    Processes ROOT files containing optical photon data and simulates
    SiPM response including quantum efficiency, timing jitter, and noise.
    """
    
    def __init__(self, config):
        """
        Initialize SiPM simulator
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.sipm_pulse = None
        self.time_axis = None
        self.pulse_interp = None
        self.output_dir = None
        
        # Initialize random number generator
        np.random.seed(42)
        
        # Load SiPM pulse shape
        self._load_simp_pulse()
        
        # Setup time axis for waveforms
        self._setup_time_axis()
    
    def _load_simp_pulse(self):
        """Load SiPM pulse shape from ROOT file"""
        pulse_file = self.config['io']['simp_pulse_file']
        
        if not Path(pulse_file).exists():
            raise FileNotFoundError(f"SiPM pulse file not found: {pulse_file}")
        
        # Open ROOT file and get TSpline3
        root_file = ROOT.TFile.Open(pulse_file)
        spline = root_file.Get("pulseShapeSpline")
        
        if not spline:
            raise ValueError("pulseShapeSpline not found in ROOT file")
        
        # Sample spline to regular grid
        t_min, t_max = spline.GetXmin(), spline.GetXmax()
        time_step = self.config['waveform']['time_step']
        
        t_pulse = np.arange(t_min, t_max + time_step, time_step)
        amplitude = np.array([spline.Eval(t) for t in t_pulse])
        
        # Create interpolation function
        self.pulse_interp = interp1d(t_pulse, amplitude, kind='linear',
                                   bounds_error=False, fill_value=0.0)
        
        # Store pulse for reference
        self.simp_pulse = {'time': t_pulse, 'amplitude': amplitude}
        
        root_file.Close()
        
        if self.config['debug']['verbose']:
            print(f"Loaded SiPM pulse: {len(t_pulse)} points from {t_min:.1f} to {t_max:.1f} ns")
    
    def _setup_time_axis(self):
        """Setup time axis for output waveforms"""
        time_window = self.config['waveform']['time_window']
        time_step = self.config['waveform']['time_step']
        
        self.time_axis = np.arange(time_window[0], time_window[1] + time_step, time_step)
        
        if self.config['debug']['verbose']:
            print(f"Waveform time axis: {len(self.time_axis)} points from {time_window[0]} to {time_window[1]} ns")
    
    def _filter_photons(self, tree, event_idx):
        """
        Filter photons based on position and type criteria
        
        Args:
            tree: ROOT tree object
            event_idx (int): Event index
            
        Returns:
            dict: Filtered photon data
        """
        tree.GetEntry(event_idx)
        
        # Get photon data
        n_photons = tree.OP_pos_final_x.size()
        
        if n_photons == 0:
            return {'x': np.array([]), 'y': np.array([]), 'z': np.array([]), 'time': np.array([])}
        
        # Convert to numpy arrays
        x = np.array([tree.OP_pos_final_x[i] for i in range(n_photons)])
        y = np.array([tree.OP_pos_final_y[i] for i in range(n_photons)])
        z = np.array([tree.OP_pos_final_z[i] for i in range(n_photons)])
        time = np.array([tree.OP_time_final[i] for i in range(n_photons)])
        is_core_c = np.array([tree.OP_isCoreC[i] for i in range(n_photons)])
        
        # Apply filters
        filter_config = self.config['photon_filter']
        
        # Z position filter
        mask = z > filter_config['z_min']
        
        # Core Cherenkov filter
        if filter_config['require_core_c']:
            mask &= is_core_c
        
        # X position filter
        x_range = filter_config['x_range']
        mask &= (x >= x_range[0]) & (x <= x_range[1])
        
        # Y position filter  
        y_range = filter_config['y_range']
        mask &= (y >= y_range[0]) & (y <= y_range[1])
        
        # Apply photon limit if debug mode enabled
        debug_config = self.config.get('debug', {})
        if debug_config.get('verbose', False):
            max_photons = debug_config.get('max_photons_per_event', 10000)
            if np.sum(mask) > max_photons:
                print(f"Event has {np.sum(mask)} photons, limiting to {max_photons}")
                indices = np.random.choice(np.sum(mask), max_photons, replace=False)
                mask_indices = np.where(mask)[0][indices]
                new_mask = np.zeros_like(mask, dtype=bool)
                new_mask[mask_indices] = True
                mask = new_mask
        
        return {
            'x': x[mask],
            'y': y[mask], 
            'z': z[mask],
            'time': time[mask]
        }
    
    def _apply_quantum_efficiency(self, photon_times):
        """
        Apply quantum efficiency to photon detection
        
        Args:
            photon_times (array): Photon arrival times
            
        Returns:
            array: Detected photon times
        """
        qe = self.config['simulation']['quantum_efficiency']
        n_photons = len(photon_times)
        
        # Random detection based on quantum efficiency
        detected = np.random.random(n_photons) < qe
        
        return photon_times[detected]
    
    def _apply_timing_jitter(self, photon_times):
        """
        Apply timing jitter to photon arrival times
        
        Args:
            photon_times (array): Photon arrival times
            
        Returns:
            array: Jittered photon times
        """
        jitter = self.config['simulation']['timing_jitter']
        n_photons = len(photon_times)
        
        # Add Gaussian timing jitter
        jitter_values = np.random.normal(0, jitter, n_photons)
        
        return photon_times + jitter_values
    
    def _generate_waveform(self, photon_times, event_id=None):
        """
        Generate SiPM waveform from photon times
        
        Args:
            photon_times (array): Detected photon arrival times
            event_id (int, optional): Event ID for individual plotting
            
        Returns:
            array: SiPM waveform amplitude vs time
        """
        waveform = np.zeros(len(self.time_axis))
        
        for t_photon in photon_times:
            # Shift pulse to photon arrival time
            pulse_amplitude = self.pulse_interp(self.time_axis - t_photon)
            waveform += pulse_amplitude
        
        return waveform
    
    def _estimate_noise_level(self):
        """
        Estimate noise level from SiPM pulse baseline
        
        Returns:
            float: RMS noise level
        """
        baseline_range = self.config['waveform']['baseline_range']
        pulse_time = self.simp_pulse['time']
        pulse_amp = self.simp_pulse['amplitude']
        
        # Find baseline region
        mask = (pulse_time >= baseline_range[0]) & (pulse_time <= baseline_range[1])
        baseline_data = pulse_amp[mask]
        
        if len(baseline_data) == 0:
            print("Warning: No baseline data found, using default noise level")
            return 0.001
        
        noise_rms = np.std(baseline_data)
        
        if self.config['debug']['verbose']:
            print(f"Estimated noise RMS: {noise_rms:.6f}")
        
        return noise_rms
    
    def _add_noise(self, waveform):
        """
        Add Gaussian background noise to waveform
        
        Args:
            waveform (array): Input waveform
            
        Returns:
            array: Waveform with added noise
        """
        if not self.config['noise']['enable_background']:
            return waveform
        
        noise_rms = self._estimate_noise_level()
        noise_scale = self.config['noise']['noise_scale']
        
        noise = np.random.normal(0, noise_rms * noise_scale, len(waveform))
        
        return waveform + noise
    
    def _plot_individual_event(self, event_id, photon_data, detected_times, 
                              jittered_times, waveform_clean, waveform_noisy):
        """
        Plot individual event diagnostics
        
        Args:
            event_id (int): Event ID
            photon_data (dict): Filtered photon data
            detected_times (array): Photon times after QE
            jittered_times (array): Photon times after jitter
            waveform_clean (array): Waveform before noise
            waveform_noisy (array): Waveform with noise
        """
        # Create event subdirectory
        event_dir = Path(self.output_dir) / "plots" / f"event_{event_id:03d}"
        event_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Photon position scatter plot
        if len(photon_data['x']) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(photon_data['x'], photon_data['y'], 
                               c=photon_data['time'], cmap='viridis', alpha=0.7)
            ax.set_xlabel('X Position (mm)')
            ax.set_ylabel('Y Position (mm)')
            ax.set_title(f'Event {event_id}: Photon Positions (colored by time)')
            plt.colorbar(scatter, label='Time (ns)')
            
            # Add SiPM boundary
            x_range = self.config['photon_filter']['x_range']
            y_range = self.config['photon_filter']['y_range']
            rect = plt.Rectangle((x_range[0], y_range[0]), 
                               x_range[1]-x_range[0], y_range[1]-y_range[0],
                               fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            plt.tight_layout()
            plt.savefig(event_dir / "photon_positions.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Photon time distribution
        if len(photon_data['time']) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(photon_data['time'], bins=50, alpha=0.7, label=f'All photons ({len(photon_data["time"])})')
            if len(detected_times) > 0:
                ax.hist(detected_times, bins=50, alpha=0.7, label=f'Detected ({len(detected_times)})')
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Photon Count')
            ax.set_title(f'Event {event_id}: Photon Time Distribution')
            ax.legend()
            plt.tight_layout()
            plt.savefig(event_dir / "photon_times.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Waveform with photon arrival markers
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: Clean vs noisy waveform
        ax1.plot(self.time_axis, waveform_clean, 'b-', linewidth=1.5, label='Clean signal')
        ax1.plot(self.time_axis, waveform_noisy, 'r-', linewidth=1, alpha=0.8, label='With noise')
        
        # Mark photon arrival times with dashed vertical lines
        if len(jittered_times) > 0:
            for t_photon in jittered_times:
                ax1.axvline(t_photon, color='green', linestyle='--', alpha=0.6, linewidth=1)
        
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Event {event_id}: SiPM Waveform ({len(jittered_times)} detected photons)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Zoomed view around signal
        if len(jittered_times) > 0:
            t_min = min(jittered_times) - 5
            t_max = max(jittered_times) + 10
            mask = (self.time_axis >= t_min) & (self.time_axis <= t_max)
            
            ax2.plot(self.time_axis[mask], waveform_clean[mask], 'b-', linewidth=1.5, label='Clean signal')
            ax2.plot(self.time_axis[mask], waveform_noisy[mask], 'r-', linewidth=1, alpha=0.8, label='With noise')
            
            # Mark photon arrivals in zoomed view
            for t_photon in jittered_times:
                if t_min <= t_photon <= t_max:
                    ax2.axvline(t_photon, color='green', linestyle='--', alpha=0.6, linewidth=1)
            
            ax2.set_xlim(t_min, t_max)
        else:
            # If no detected photons, show a default range
            t_center = 0
            mask = (self.time_axis >= t_center-10) & (self.time_axis <= t_center+10)
            ax2.plot(self.time_axis[mask], waveform_clean[mask], 'b-', linewidth=1.5, label='Clean signal')
            ax2.plot(self.time_axis[mask], waveform_noisy[mask], 'r-', linewidth=1, alpha=0.8, label='With noise')
        
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Zoomed View')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(event_dir / "waveform.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.config['debug']['verbose']:
            print(f"Individual plots saved for event {event_id} in {event_dir}")
    
    def process_events(self, max_events=None):
        """
        Main processing function - simulate SiPM response for events
        
        Args:
            max_events (int, optional): Maximum events to process
        """
        # Setup output directory
        base_output = self.config['io']['output_dir']
        self.output_dir = setup_output_directory(base_output)
        save_config(self.config, self.output_dir)
        
        # Get ROOT file list
        root_files = generate_file_list(self.config)
        
        # Determine number of events to process
        n_events_config = self.config['simulation']['n_events']
        if max_events is not None:
            n_events_to_process = min(max_events, n_events_config)
        else:
            n_events_to_process = n_events_config
        
        print(f"Processing {n_events_to_process} events from {len(root_files)} ROOT files")
        
        # Storage for results
        all_waveforms = []
        all_photon_data = {'x': [], 'y': [], 'event_id': []}
        total_photons = 0
        events_processed = 0
        
        # Process ROOT files
        for file_path in tqdm(root_files, desc="Processing files"):
            if events_processed >= n_events_to_process:
                break
                
            if not Path(file_path).exists():
                print(f"Warning: File not found: {file_path}")
                continue
            
            # Open ROOT file
            root_file = ROOT.TFile.Open(file_path)
            tree = root_file.Get("tree")
            
            if not tree:
                print(f"Warning: No tree found in {file_path}")
                root_file.Close()
                continue
            
            # Process events in this file
            n_events_in_file = tree.GetEntries()
            
            for event_idx in range(n_events_in_file):
                if events_processed >= n_events_to_process:
                    break
                
                # Filter photons
                photon_data = self._filter_photons(tree, event_idx)
                n_photons = len(photon_data['time'])
                
                if n_photons == 0:
                    if self.config['debug']['verbose']:
                        print(f"Event {events_processed}: No photons after filtering")
                    events_processed += 1
                    continue
                
                # Store photon positions for 2D histogram
                all_photon_data['x'].extend(photon_data['x'])
                all_photon_data['y'].extend(photon_data['y'])
                all_photon_data['event_id'].extend([events_processed] * n_photons)
                
                # Apply quantum efficiency
                detected_times = self._apply_quantum_efficiency(photon_data['time'])
                n_detected = len(detected_times)
                
                if n_detected == 0:
                    if self.config['debug']['verbose']:
                        print(f"Event {events_processed}: No photons detected after QE")
                    all_waveforms.append(np.zeros(len(self.time_axis)))
                    events_processed += 1
                    continue
                
                # Apply timing jitter
                jittered_times = self._apply_timing_jitter(detected_times)
                
                # Generate waveform
                waveform = self._generate_waveform(jittered_times, events_processed)
                
                # Add noise
                waveform_with_noise = self._add_noise(waveform)
                
                # Plot individual event if enabled
                if self.config['debug']['plot_individual_events']:
                    self._plot_individual_event(
                        events_processed, photon_data, detected_times, 
                        jittered_times, waveform, waveform_with_noise
                    )
                
                all_waveforms.append(waveform_with_noise)
                
                total_photons += n_photons
                events_processed += 1
                
                if self.config['debug']['verbose'] and events_processed % 100 == 0:
                    print(f"Processed {events_processed} events")
            
            root_file.Close()
        
        # Convert to numpy arrays
        waveforms = np.array(all_waveforms)
        
        # Create and save 2D histogram
        if self.config['io']['save_plots'] and len(all_photon_data['x']) > 0:
            x_range = self.config['photon_filter']['x_range']
            y_range = self.config['photon_filter']['y_range']
            
            x_bins = np.linspace(x_range[0], x_range[1], 50)
            y_bins = np.linspace(y_range[0], y_range[1], 50)
            
            plot_path = Path(self.output_dir) / "plots" / "photon_distribution_2d.png"
            plot_2d_histogram(
                all_photon_data['x'], all_photon_data['y'],
                x_bins, y_bins,
                f"Photon Distribution on SiPM ({events_processed} events)",
                str(plot_path)
            )
        
        # Save waveforms
        metadata = {
            'n_events': events_processed,
            'total_photons': total_photons,
            'quantum_efficiency': self.config['simulation']['quantum_efficiency'],
            'timing_jitter': self.config['simulation']['timing_jitter'],
            'time_step': self.config['waveform']['time_step'],
            'baseline_range': self.config['waveform']['baseline_range']
        }
        
        output_format = self.config['io']['output_format']
        waveform_file = Path(self.output_dir) / "waveforms" / f"simp_waveforms.{output_format}"
        
        if output_format == 'h5':
            save_waveform_h5(waveforms, self.time_axis, metadata, str(waveform_file))
        else:
            save_waveform_npz(waveforms, self.time_axis, metadata, str(waveform_file))
        
        # Print summary
        print_simulation_summary(self.config, len(root_files), total_photons, self.output_dir)
        
        return {
            'waveforms': waveforms,
            'time_axis': self.time_axis,
            'metadata': metadata,
            'output_dir': self.output_dir
        }
