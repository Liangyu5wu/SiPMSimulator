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

- `simulation`: Core simulation parameters (events, quantum efficiency, timing jitter)
- `photon_filter`: SiPM active area and photon selection criteria
- `waveform`: Time window, sampling, and baseline parameters
- `noise`: Background noise modeling settings
- `io`: File paths, output formats, and save options
- `root_files`: ROOT file directory and pattern matching
- `debug`: Verbose output and visualization controls

### Output Structure

Results are saved in timestamped directories:
```
output/run_YYYYMMDD_HHMMSS/
├── waveforms/     # SiPM waveform data (H5/NPZ)
├── plots/         # 2D photon distribution histograms
└── metadata/      # Configuration and simulation statistics
```

### Environment Dependencies

Requires ROOT framework (loaded via `/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el8-gcc11-opt/setup.sh`) and Python packages listed in `requirements.txt`. Uses UV for fast Python package management.