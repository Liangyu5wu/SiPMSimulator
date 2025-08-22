#!/usr/bin/env python3
"""
SiPM Simulation Runner

Command-line interface for running SiPM simulations with YAML configuration.

Usage:
    python scripts/run_simulation.py --config configs/default.yaml
    python scripts/run_simulation.py --config configs/default.yaml --events 100
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulator import SiPMSimulator
from utils import load_config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run SiPM simulation with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default configuration
    python scripts/run_simulation.py --config configs/default.yaml
    
    # Override number of events
    python scripts/run_simulation.py --config configs/default.yaml --events 100
    
    # Run with custom output directory
    python scripts/run_simulation.py --config configs/default.yaml --output ../my_results
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--events", "-e",
        type=int,
        help="Override number of events to process"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Override output directory"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and exit without running simulation"
    )
    
    return parser.parse_args()


def validate_config(config):
    """
    Validate configuration parameters
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_sections = ['simulation', 'photon_filter', 'waveform', 'io']
    
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required configuration section: {section}")
            return False
    
    # Check SiPM pulse file exists
    pulse_file = config['io']['sipm_pulse_file']
    if not Path(pulse_file).exists():
        print(f"Error: SiPM pulse file not found: {pulse_file}")
        return False
    
    # Check ROOT files base path exists
    root_base = config['root_files']['base_path']
    if not Path(root_base).exists():
        print(f"Error: ROOT files base path not found: {root_base}")
        return False
    
    # Validate parameter ranges
    qe = config['simulation']['quantum_efficiency']
    if not 0 <= qe <= 1:
        print(f"Error: Quantum efficiency must be between 0 and 1, got {qe}")
        return False
    
    jitter = config['simulation']['timing_jitter']
    if jitter < 0:
        print(f"Error: Timing jitter must be non-negative, got {jitter}")
        return False
    
    return True


def print_config_summary(config):
    """Print summary of configuration parameters"""
    print("\n" + "="*50)
    print("SIMULATION CONFIGURATION")
    print("="*50)
    
    print(f"Events to process: {config['simulation']['n_events']}")
    print(f"Quantum efficiency: {config['simulation']['quantum_efficiency']:.1%}")
    print(f"Timing jitter: {config['simulation']['timing_jitter']} ns")
    
    print(f"\nSiPM active area:")
    print(f"  X range: {config['photon_filter']['x_range']} mm")
    print(f"  Y range: {config['photon_filter']['y_range']} mm")
    print(f"  Z minimum: {config['photon_filter']['z_min']} mm")
    
    print(f"\nWaveform settings:")
    print(f"  Time window: {config['waveform']['time_window']} ns")
    print(f"  Time step: {config['waveform']['time_step']} ns")
    print(f"  Baseline range: {config['waveform']['baseline_range']} ns")
    
    print(f"\nNoise settings:")
    print(f"  Background noise: {'Enabled' if config['noise']['enable_background'] else 'Disabled'}")
    print(f"  Noise scale: {config['noise']['noise_scale']}")
    
    print(f"\nInput/Output:")
    print(f"  SiPM pulse file: {config['io']['sipm_pulse_file']}")
    print(f"  Output directory: {config['io']['output_dir']}")
    print(f"  Output format: {config['io']['output_format']}")
    print(f"  Save plots: {'Yes' if config['io']['save_plots'] else 'No'}")
    
    print("="*50)


def main():
    """Main function"""
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Apply command line overrides
    if args.events:
        config['simulation']['n_events'] = args.events
        print(f"Override: Setting n_events to {args.events}")
    
    if args.output:
        config['io']['output_dir'] = args.output
        print(f"Override: Setting output directory to {args.output}")
    
    if args.verbose:
        config['debug']['verbose'] = True
        print("Override: Enabling verbose output")
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed. Exiting.")
        sys.exit(1)
    
    # Print configuration summary
    print_config_summary(config)
    
    # Dry run mode - exit after showing configuration
    if args.dry_run:
        print("\nDry run mode - configuration validated successfully.")
        print("Run without --dry-run to execute simulation.")
        sys.exit(0)
    
    # Run simulation
    try:
        print(f"\nStarting SiPM simulation...")
        
        # Create simulator instance
        simulator = SiPMSimulator(config)
        
        # Process events
        results = simulator.process_events()
        
        print(f"\nSimulation completed successfully!")
        print(f"Results saved to: {results['output_dir']}")
        print(f"Generated {len(results['waveforms'])} waveforms")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
