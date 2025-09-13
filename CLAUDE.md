# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment Setup
```bash
# Setup environment (required before running)
source setup.sh

# Alternative manual setup
uv venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Running Simulations
```bash
# Basic simulation run
python scripts/run_simulation.py --config configs/default.yaml

# Override number of events
python scripts/run_simulation.py --config configs/default.yaml --events 100

# Validate configuration without running
python scripts/run_simulation.py --config configs/default.yaml --dry-run

# SLURM cluster execution
sbatch jobs/sipm_sim.sh

# Position scanning (individual jobs per region)
./jobs/position_scan.sh                           # Default scan: x[-6.65,-1.85], y[1.12,5.92]
./jobs/position_scan.sh -6.65 -6.25 1.12 1.52 0.4 # Custom: submits 4 individual jobs

# Merge scan results (with size optimization)
python scripts/merge_scan_results.py                    # Default: HDF5 format with all data
python scripts/merge_scan_results.py --format npz       # NPZ format (smaller, faster)
python scripts/merge_scan_results.py --waveforms-only   # Waveforms+time+photon_times (60% smaller)
python scripts/merge_scan_results.py --format split --chunk-size 5000 # Split into 5k-event chunks

# Inspect merged results or ML datasets
python scripts/inspect_merged_data.py                   # Auto-detect and inspect available files
python scripts/inspect_merged_data.py path/to/file.npz  # Inspect specific file

# Create ML dataset (filtered and cleaned for training)
python scripts/create_ml_dataset.py                     # Filter out-of-window photons and high-photon events
python scripts/create_ml_dataset.py --max-photons 80    # Custom photon count threshold

# Machine learning model parameter sweep
cd ml_training/
python param_sweep.py                                   # Local execution for parameter optimization (216 combinations)
sbatch ml_param_sweep.sh                                # SLURM cluster execution (100 parallel jobs, ~2 combinations each)
python merge_results.py                                 # Merge parallel results after jobs complete
```

### Pulse Shape Creation
```bash
# Create normalized TSpline3 from feather file
python scripts/make_spline.py

# This script automatically:
# - Determines normalization parameters by comparing with existing pulse
# - Creates compatible ROOT TSpline3 format
# - Generates overlay comparison plot in ../SPR_data/pulse_overlay.png
# - Outputs ../SPR_data/SiPM_TTU_pulse.root ready for use
```

## Architecture

### Core Components

**SiPM Simulator** (`src/simulator.py`): Main simulation engine that processes ROOT files containing optical photon data and simulates Silicon Photomultiplier response. Key features include quantum efficiency modeling, timing jitter, pulse shape convolution, and background noise generation.

**Configuration System** (`configs/default.yaml`): YAML-based configuration covering simulation parameters (events, quantum efficiency, timing jitter), photon filtering (active area, z-position), waveform generation, noise modeling, and I/O settings.

**Command Line Interface** (`scripts/run_simulation.py`): Provides argument parsing, configuration validation, parameter overrides, and dry-run functionality.

**Pulse Shape Creator** (`scripts/make_spline.py`): Converts feather pulse data to normalized ROOT TSpline3 format. Automatically determines normalization parameters, ensures compatibility with existing code, and generates comparison plots.

### Data Flow

1. **Input Processing**: Reads ROOT files from specified directories using configurable file patterns
2. **Photon Filtering**: Filters optical photons by position (SiPM active area) and type (Cherenkov core)
3. **Physics Simulation**: Applies quantum efficiency, timing jitter, and pulse shape convolution
4. **Output Generation**: Saves waveforms (H5/NPZ format), 2D photon distribution plots, and metadata

### Performance Optimizations

The simulator implements several critical optimizations:
- **ROOT Branch Activation**: Only loads necessary data branches (50-70% I/O reduction)
- **Efficient Array Conversion**: Uses `np.array(ROOT_vector)` instead of element-by-element access (30x speedup)
- **Vectorized Operations**: Numpy operations replace Python loops where possible

Typical performance: 2-3 seconds for 10 events (vs 60+ seconds in original implementation).

### Configuration Structure

- `simulation`: Core simulation parameters (events, quantum efficiency=0.4, timing jitter)
- `photon_filter`: SiPM active area and photon selection criteria
- `waveform`: Time window, sampling, and baseline parameters (baseline_range=[-15,-3])
- `noise`: Background noise modeling settings
- `io`: File paths, output formats, and save options
- `root_files`: ROOT file directory and pattern matching
- `debug`: Verbose output and visualization controls

### Output Structure

Results are saved in directories with descriptive naming:

**Position Scanning:**
```
output/run_YYYYMMDD_xNaNbtoNcNd_yNaNbtoNcNd/
├── waveforms/     # SiPM waveform data (H5/NPZ)
├── plots/         # 2D photon distribution histograms
└── metadata/      # Configuration and simulation statistics
```

**Regular Runs:**
```
output/run_YYYYMMDD_HHMMSS/
├── waveforms/     # SiPM waveform data (H5/NPZ)
├── plots/         # 2D photon distribution histograms
└── metadata/      # Configuration and simulation statistics
```

Directory naming convention for position scans:
- Date format: YYYYMMDD  
- Coordinate format: 'n' = negative sign, 'p' = decimal point
- Example: `run_20250912_xn6p65ton1p85_y1p12to5p92` represents x[-6.65,-1.85], y[1.12,5.92]

### SiPM Waveforms HDF5 File Structure

The main output file `sipm_waveforms.h5` contains:

**Datasets:**
- `waveforms` (n_events × n_time_points): SiPM response waveforms with noise, gzip compressed
- `time_axis` (n_time_points,): Time axis in nanoseconds
- `photon_times_original` (n_events × 100): Original photon arrival times, padded/truncated to 100, gzip compressed
- `photon_times_jittered` (n_events × 100): Original photon times with timing jitter applied, gzip compressed  
- `photon_times_detected` (n_events × 100): Photon times after quantum efficiency filtering, gzip compressed
- `photon_times_detected_jittered` (n_events × 100): Final detected photon times with jitter, gzip compressed

**Metadata Group:**
- `n_events`: Number of processed events
- `total_photons`: Total photons processed across all events
- `quantum_efficiency`: QE value used in simulation
- `timing_jitter`: Timing jitter sigma in ns
- `time_step`: Waveform sampling time step in ns
- `baseline_range`: Time range used for noise estimation

**Data Types:**
- All photon time arrays use NaN padding for events with <100 photons
- Waveforms are float64 arrays with time-domain voltage response
- Time axis matches waveform sampling (typically 0.05ns steps)

### Environment Dependencies

Requires ROOT framework (loaded via `/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el8-gcc11-opt/setup.sh`) and Python packages listed in `requirements.txt`. Uses UV for fast Python package management.