"""
SiPM Simulator Main Class - Optimized with Batch ROOT Reading

Simulates Silicon Photomultiplier response to optical photons from 
particle physics experiments using ROOT data input.

Optimizations:
- Batch ROOT data reading with branch activation
- Vectorized photon filtering and processing
- Reduced memory allocations
"""

import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
import h5py
from tqdm import tqdm

from utils import (
    setup_output_directory, save_config, plot_2d_histogram,
    save_waveform_h5, save_waveform_npz, generate_file_list,
    print_simulation_summary
)


class SiPMSimulator:
    """
    Main SiPM simulation class with batch processing optimization
    
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
        
        # Batch processing parameters
        batch_config = config.get('batch_processing', {})
        self.batch_size = batch_config.get('batch_size', 20)  # Optimized for small ROOT files
        
        # Initialize random number generator
        np.random.seed(42)
        
        # Load SiPM pulse shape
        self._load_sipm_pulse()
        
        # Setup time axis for waveforms
        self._setup_time_axis()
        
        # Calculate noise RMS once during initialization
        self.noise_rms = self._estimate_noise_level()
    
    def _load_sipm_pulse(self):
        """Load SiPM pulse shape from ROOT file"""
        pulse_file = self.config['io']['sipm_pulse_file']
        
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
        self.sipm_pulse = {'time': t_pulse, 'amplitude': amplitude}
        
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
    
    def _setup_tree_branches(self, tree):
        """
        Only activate needed branches to improve reading efficiency
        
        Args:
            tree: ROOT tree object
        """
        # Disable all branches
        tree.SetBranchStatus("*", 0)
        
        # Only activate needed branches
        needed_branches = [
            "OP_pos_final_x",
            "OP_pos_final_y", 
            "OP_pos_final_z",
            "OP_time_final",
            "OP_isCoreC"
        ]
        
        for branch in needed_branches:
            tree.SetBranchStatus(branch, 1)
        
        if self.config['debug']['verbose']:
            print(f"Activated {len(needed_branches)} branches for optimized reading")
    
    def _read_events_batch(self, tree, start_event, batch_size):
        """
        Batch read multiple events' data into numpy arrays
        
        Args:
            tree: ROOT tree object
            start_event (int): Starting event index
            batch_size (int): Number of events to read in this batch
            
        Returns:
            dict: Batch data with all photons from multiple events
        """
        n_events = min(batch_size, tree.GetEntries() - start_event)
        
        if n_events <= 0:
            return None
        
        # Pre-allocate storage for batch data
        batch_data = {
            'x': [], 'y': [], 'z': [], 'time': [], 'is_core_c': [],
            'event_ids': []
        }
        
        total_photons = 0
        
        # Read batch of events
        for i in range(n_events):
            event_idx = start_event + i
            tree.GetEntry(event_idx)
            
            n_photons = tree.OP_pos_final_x.size()
            if n_photons == 0:
                continue
            
            # Apply photon limit if in debug mode
            debug_config = self.config.get('debug', {})
            if debug_config.get('verbose', False):
                max_photons = debug_config.get('max_photons_per_event', 10000)
                if n_photons > max_photons:
                    # Randomly sample photons to stay within limit
                    indices = np.random.choice(n_photons, max_photons, replace=False)
                    n_photons = max_photons
                else:
                    indices = np.arange(n_photons)
            else:
                indices = np.arange(n_photons)
            
            # Vectorized reading - convert ROOT vectors to numpy arrays
            try:
                # Use numpy.frombuffer for efficient conversion
                x_vec = np.array([tree.OP_pos_final_x[idx] for idx in indices])
                y_vec = np.array([tree.OP_pos_final_y[idx] for idx in indices])
                z_vec = np.array([tree.OP_pos_final_z[idx] for idx in indices])
                time_vec = np.array([tree.OP_time_final[idx] for idx in indices])
                core_vec = np.array([tree.OP_isCoreC[idx] for idx in indices])
                
                batch_data['x'].append(x_vec)
                batch_data['y'].append(y_vec)
                batch_data['z'].append(z_vec)
                batch_data['time'].append(time_vec)
                batch_data['is_core_c'].append(core_vec)
                batch_data['event_ids'].append(np.full(n_photons, event_idx))
                
                total_photons += n_photons
                
            except Exception as e:
                if self.config['debug']['verbose']:
                    print(f"Warning: Failed to read event {event_idx}: {e}")
                continue
        
        # Concatenate all events' data into single arrays
        if batch_data['x']:
            concatenated_data = {
                'x': np.concatenate(batch_data['x']),
                'y': np.concatenate(batch_data['y']),
                'z': np.concatenate(batch_data['z']),
                'time': np.concatenate(batch_data['time']),
                'is_core_c': np.concatenate(batch_data['is_core_c']),
                'event_ids': np.concatenate(batch_data['event_ids'])
            }
            
            if self.config['debug']['verbose']:
                print(f"Read batch: {n_events} events, {total_photons} photons")
            
            return concatenated_data
        else:
            return None
    
    def _filter_photons_batch(self, batch_data):
        """
        Apply vectorized filtering to batch photon data
        
        Args:
            batch_data (dict): Batch photon data
            
        Returns:
            dict: Filtered photon data
        """
        if batch_data is None or len(batch_data['time']) == 0:
            return None
        
        filter_config = self.config['photon_filter']
        
        # Vectorized filtering - all conditions applied at once
        mask = (
            (batch_data['z'] > filter_config['z_min']) &
            (batch_data['x'] >= filter_config['x_range'][0]) &
            (batch_data['x'] <= filter_config['x_range'][1]) &
            (batch_data['y'] >= filter_config['y_range'][0]) &
            (batch_data['y'] <= filter_config['y_range'][1])
        )
        
        if filter_config['require_core_c']:
            mask &= (batch_data['is_core_c'] == 1)
        
        n_filtered = np.sum(mask)
        if n_filtered == 0:
            return None
        
        # Apply filtering
        filtered_data = {
            'x': batch_data['x'][mask],
            'y': batch_data['y'][mask],
            'z': batch_data['z'][mask],
            'time': batch_data['time'][mask],
            'event_ids': batch_data['event_ids'][mask]
        }
        
        if self.config['debug']['verbose']:
            print(f"Filtered photons: {len(batch_data['time'])} → {n_filtered}")
        
        return filtered_data
    
    def _process_filtered_batch(self, filtered_data, all_waveforms, all_photon_data,
                               all_photon_times_original, all_photon_times_detected,
                               all_photon_times_jittered, all_photon_times_detected_jittered,
                               events_processed_ref):
        """
        Process filtered batch data and generate waveforms per event
        
        Args:
            filtered_data (dict): Filtered photon data
            all_waveforms (list): List to append waveforms
            all_photon_data (dict): Dict to extend photon position data
            all_photon_times_* (list): Lists to append photon timing data
            events_processed_ref (list): Reference to track processed events count
        """
        # Group photons by event ID
        unique_event_ids = np.unique(filtered_data['event_ids'])
        
        for event_id in unique_event_ids:
            # Get photons for this event
            event_mask = filtered_data['event_ids'] == event_id
            event_photons = {
                'x': filtered_data['x'][event_mask],
                'y': filtered_data['y'][event_mask],
                'z': filtered_data['z'][event_mask],
                'time': filtered_data['time'][event_mask]
            }
            
            n_photons = len(event_photons['time'])
            
            # Store photon positions for 2D histogram
            all_photon_data['x'].extend(event_photons['x'])
            all_photon_data['y'].extend(event_photons['y'])
            all_photon_data['event_id'].extend([event_id] * n_photons)
            
            # Apply timing jitter to original photons (vectorized)
            original_jittered = self._apply_timing_jitter_vectorized(event_photons['time'])
            
            # Apply quantum efficiency (vectorized)
            detected_times = self._apply_quantum_efficiency_vectorized(event_photons['time'])
            n_detected = len(detected_times)
            
            # Store photon times (padded to length 100)
            all_photon_times_original.append(self._pad_photon_times(event_photons['time']))
            all_photon_times_jittered.append(self._pad_photon_times(original_jittered))
            all_photon_times_detected.append(self._pad_photon_times(detected_times))
            
            if n_detected == 0:
                # No detected photons - add zero waveform
                all_photon_times_detected_jittered.append(self._pad_photon_times(np.array([])))
                all_waveforms.append(np.zeros(len(self.time_axis)))
            else:
                # Apply timing jitter to detected photons (vectorized)
                jittered_times = self._apply_timing_jitter_vectorized(detected_times)
                all_photon_times_detected_jittered.append(self._pad_photon_times(jittered_times))
                
                # Generate waveform
                waveform = self._generate_waveform(jittered_times)
                
                # Add noise
                waveform_with_noise = self._add_noise(waveform)
                all_waveforms.append(waveform_with_noise)
            
            events_processed_ref[0] += 1
    
    def _apply_quantum_efficiency_vectorized(self, photon_times):
        """
        Vectorized quantum efficiency application
        
        Args:
            photon_times (array): Photon arrival times
            
        Returns:
            array: Detected photon times
        """
        qe = self.config['simulation']['quantum_efficiency']
        
        # Vectorized random detection
        detected_mask = np.random.random(len(photon_times)) < qe
        
        return photon_times[detected_mask]
    
    def _apply_timing_jitter_vectorized(self, photon_times):
        """
        Vectorized timing jitter application
        
        Args:
            photon_times (array): Photon arrival times
            
        Returns:
            array: Jittered photon times
        """
        jitter = self.config['simulation']['timing_jitter']
        
        # Vectorized Gaussian jitter
        jitter_values = np.random.normal(0, jitter, len(photon_times))
        
        return photon_times + jitter_values
    
    def _generate_waveform(self, photon_times):
        """
        Generate SiPM waveform from photon times
        
        Args:
            photon_times (array): Detected photon arrival times
            
        Returns:
            array: SiPM waveform amplitude vs time
        """
        waveform = np.zeros(len(self.time_axis))
        
        # Get original pulse data
        pulse_time = self.sipm_pulse['time']
        pulse_amp = self.sipm_pulse['amplitude']
        
        for t_photon in photon_times:
            # Shift pulse time axis to photon arrival time
            shifted_time = pulse_time + t_photon
            
            # Interpolate onto output time grid
            shifted_pulse = np.interp(self.time_axis, shifted_time, pulse_amp, 
                                    left=0, right=0)
            waveform += shifted_pulse
        
        return waveform
    
    def _estimate_noise_level(self):
        """
        Estimate noise level from SiPM pulse baseline
        
        Returns:
            float: RMS noise level
        """
        baseline_range = self.config['waveform']['baseline_range']
        pulse_time = self.sipm_pulse['time']
        pulse_amp = self.sipm_pulse['amplitude']
        
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
        
        noise_scale = self.config['noise']['noise_scale']
        
        noise = np.random.normal(0, self.noise_rms * noise_scale, len(waveform))
        
        return waveform + noise
    
    def _pad_photon_times(self, times_array, max_length=100):
        """
        Pad photon times array to fixed length with NaN
        
        Args:
            times_array (array): Photon times
            max_length (int): Maximum array length
            
        Returns:
            array: Padded array of length max_length
        """
        if len(times_array) >= max_length:
            return times_array[:max_length]
        else:
            padded = np.full(max_length, np.nan)
            padded[:len(times_array)] = times_array
            return padded
    
    def process_events(self, max_events=None):
        """
        Main processing function with batch optimization
        
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
        print(f"Batch size: {self.batch_size} events per batch")
        
        # Storage for results
        all_waveforms = []
        all_photon_data = {'x': [], 'y': [], 'event_id': []}
        all_photon_times_original = []
        all_photon_times_jittered = []
        all_photon_times_detected = []
        all_photon_times_detected_jittered = []
        total_photons = 0
        events_processed_ref = [0]  # Use list for mutable reference
        
        # Process ROOT files with batch optimization
        for file_path in tqdm(root_files, desc="Processing files"):
            if events_processed_ref[0] >= n_events_to_process:
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
            
            # Optimize branch reading
            self._setup_tree_branches(tree)
            
            n_events_in_file = tree.GetEntries()
            
            # Batch processing loop
            for start_event in range(0, n_events_in_file, self.batch_size):
                if events_processed_ref[0] >= n_events_to_process:
                    break
                
                # Read batch of events
                batch_data = self._read_events_batch(tree, start_event, self.batch_size)
                
                if batch_data is None:
                    continue
                
                # Count total photons
                total_photons += len(batch_data['time'])
                
                # Filter photons (vectorized)
                filtered_data = self._filter_photons_batch(batch_data)
                
                if filtered_data is None:
                    continue
                
                # Process filtered batch
                self._process_filtered_batch(
                    filtered_data, all_waveforms, all_photon_data,
                    all_photon_times_original, all_photon_times_detected,
                    all_photon_times_jittered, all_photon_times_detected_jittered,
                    events_processed_ref
                )
                
                if self.config['debug']['verbose'] and events_processed_ref[0] % 100 == 0:
                    print(f"Processed {events_processed_ref[0]} events")
            
            root_file.Close()
        
        events_processed = events_processed_ref[0]
        
        # Convert to numpy arrays
        waveforms = np.array(all_waveforms)
        photon_times_original = np.array(all_photon_times_original)
        photon_times_jittered = np.array(all_photon_times_jittered)
        photon_times_detected = np.array(all_photon_times_detected)
        photon_times_detected_jittered = np.array(all_photon_times_detected_jittered)
        
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
            'baseline_range': self.config['waveform']['baseline_range'],
            'batch_size': self.batch_size
        }
        
        output_format = self.config['io']['output_format']
        waveform_file = Path(self.output_dir) / "waveforms" / f"sipm_waveforms.{output_format}"
        
        if output_format == 'h5':
            save_waveform_h5(waveforms, self.time_axis, metadata, str(waveform_file),
                            photon_times_original, photon_times_jittered,
                            photon_times_detected, photon_times_detected_jittered)
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
