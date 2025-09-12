# SiPM Simulator

SiPM simulation for particle physics experiments. Processes ROOT files containing optical photon data and simulates Silicon Photomultiplier response.

## Setup

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.9
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Project Structure

```
SiPMSimulator/
├── src/                    # Core simulation code
│   ├── simulator.py        # Main SiPM simulation engine
│   ├── utils.py            # Utility functions
│   └── init.py
├── scripts/                # Executable scripts
│   ├── run_simulation.py   # Main simulation runner
│   └── make_spline.py      # Pulse shape creator
├── configs/                # Configuration files
│   └── default.yaml        # Default simulation parameters
├── jobs/                   # SLURM job scripts
│   └── sipm_sim.sh         # Cluster job submission
├── requirements.txt        # Python dependencies
└── setup.sh                # Environment setup

External directories:
├── ../output/              # Simulation results (timestamped runs)
├── ../logs/                # Log files for cluster jobs  
└── ../SPR_data/            # Pulse shape data files
```

## Usage

### Pulse Shape Preparation

Create normalized pulse shapes for simulation:
```bash
python scripts/make_spline.py
```

This automatically:
- Reads pulse data from feather files
- Normalizes to match existing pulse format
- Generates comparison plots
- Outputs ROOT TSpline3 files ready for simulation

### Local Execution

Basic simulation:
```bash
python scripts/run_simulation.py --config configs/default.yaml
```

Override number of events:
```bash
python scripts/run_simulation.py --config configs/default.yaml --events 100
```

Check configuration without running:
```bash
python scripts/run_simulation.py --config configs/default.yaml --dry-run
```

### SLURM Cluster Execution

For running on SLURM clusters:

1. **Single position simulation**:
```bash
sbatch jobs/sipm_sim.sh
```

2. **Position scanning** (automated multi-position runs):
```bash
# Default scan: x=[-6.65 to -1.85], y=[1.12 to 5.92], step=0.4 cm
./jobs/position_scan.sh

# Custom scan parameters: x_start x_end y_start y_end step
# Example: 2x2 grid with regions spaced 0.4 cm apart
./jobs/position_scan.sh -6.65 -6.25 1.12 1.52 0.4
```

**Parameter Logic:**
- `x_start`, `x_end`: Lower bounds of first and last scan regions in X
- `y_start`, `y_end`: Lower bounds of first and last scan regions in Y  
- `step`: Spacing between region lower bounds
- Each region is fixed at 0.3×0.3 cm (SiPM active area)

**Example:** `-6.65 -6.25 1.12 1.52 0.4` creates 4 regions:
- Region 1: x[-6.65, -6.35], y[1.12, 1.42]
- Region 2: x[-6.25, -5.95], y[1.12, 1.42]  
- Region 3: x[-6.65, -6.35], y[1.52, 1.82]
- Region 4: x[-6.25, -5.95], y[1.52, 1.82]

The position scan automatically:
- Creates scan regions based on lower bound ranges and step spacing
- **Submits individual SLURM jobs for each 0.3×0.3 cm region** (parallel execution)
- Each job uses 4GB memory and 2-hour time limit for efficient resource usage
- Generates temporary config files and runs simulations independently
- Saves results to separate timestamped directories

**Job Management:**
- Monitor all jobs: `squeue -u $USER`
- View logs: `tail -f ../logs/output_sipm_scan_*`
- Each job is named `sipm_scan_XXX` with unique output files

### Merging Scan Results

After position scanning completes, merge all results into a single HDF5 file:

```bash
# Basic merge with default HDF5 format
python scripts/merge_scan_results.py

# Size optimization options:
python scripts/merge_scan_results.py --format npz           # NPZ format (50-70% smaller, faster loading)
python scripts/merge_scan_results.py --waveforms-only      # Waveforms+time+photon_times (60% size reduction)
python scripts/merge_scan_results.py --format split        # Split into manageable chunks (10k events/file)

# Advanced filtering and chunking:
python scripts/merge_scan_results.py --min-photons 5 --format npz --waveforms-only
python scripts/merge_scan_results.py --format split --chunk-size 5000  # Custom chunk size
```

**Features & Size Optimization:**
- **Multiple formats**: HDF5 (full compatibility), NPZ (50-70% smaller), Split (chunked files)
- **Waveforms-only mode**: Save essential data (waveforms + time_axis + detected photon times) for 60% size reduction
- **Smart filtering**: Automatically removes zero-photon events by default
- **Position tracking**: Preserves x,y coordinate ranges for each event (unless waveforms-only)
- **Chunked output**: Split large datasets into manageable file sizes
- **Compression**: Uses gzip compression for all formats

### Inspecting Merged Data

Check the contents and structure of merged files:

```bash
# Inspect default merged file
python scripts/inspect_merged_data.py

# Inspect specific file
python scripts/inspect_merged_data.py ../output/my_data.npz
```

Shows file size, data arrays, dimensions, and basic statistics.

### Creating ML Training Dataset

Generate a clean, filtered dataset optimized for machine learning:

```bash
# Create filtered ML dataset from merged results
python scripts/create_ml_dataset.py

# Custom photon count threshold (default: <95 photons per event)
python scripts/create_ml_dataset.py --max-photons 80
```

**Dataset Preparation Workflow:**
1. **Position Scanning**: Run parallel SLURM jobs → Individual region results
2. **Data Merging**: Combine all regions → Single consolidated file  
3. **ML Filtering**: Remove outliers and out-of-window photons → Clean training dataset

**Filtering Steps:**
- Remove photons outside waveform time window ([-1, 50] ns)
- Filter events with excessive photon counts (≥95 photons) to remove anomalies
- Preserve only essential data: `waveforms`, `time_axis`, `photon_times_detected_jittered`
- Generate clean, ready-to-use ML training data

## Configuration

Edit `configs/default.yaml` to adjust:
- Number of events to process
- Quantum efficiency (default: 40%)
- Timing jitter (default: 0.2 ns)
- SiPM active area (default: x[-4.25,-3.95], y[4.32,4.62])
- Baseline range for noise estimation (default: [-15,-3] ns)
- Output directory and format

## Physics

The simulation implements realistic SiPM response modeling:

1. **Photon Filtering**: Selects photons by position (SiPM active area) and type (Cherenkov core)
2. **Quantum Efficiency**: Applies detection probability based on configured QE
3. **Timing Jitter**: Adds Gaussian timing uncertainty to photon arrival times
4. **Pulse Convolution**: Convolves detected photons with SiPM pulse shape
5. **Noise Addition**: Adds realistic Gaussian background noise
6. **Output Generation**: Produces time-domain waveforms with metadata

Results are saved in descriptive directories:
- **Position scans**: `../output/run_YYYYMMDD_xNaNbtoNcNd_yNaNbtoNcNd/` (e.g., `run_20250912_xn6p65ton1p85_y1p12to5p92`)
- **Regular runs**: `../output/run_YYYYMMDD_HHMMSS/` 

Each directory contains waveforms, plots, and metadata. The position scan naming convention uses 'n' for negative signs and 'p' for decimal points to ensure filesystem compatibility.

## Output Data Format

### SiPM Waveforms File (`sipm_waveforms.h5`)

The main simulation output is an HDF5 file containing:

**Primary Data:**
- **`waveforms`**: SiPM response waveforms (n_events × n_time_points) with realistic noise
- **`time_axis`**: Corresponding time axis in nanoseconds

**Photon Timing Data** (all arrays: n_events × 100 with NaN padding):
- **`photon_times_original`**: Raw photon arrival times from ROOT files
- **`photon_times_jittered`**: Original times with timing jitter applied
- **`photon_times_detected`**: Times after quantum efficiency filtering  
- **`photon_times_detected_jittered`**: Final detected times used for waveform generation

**Simulation Metadata:**
- Run parameters (quantum efficiency, timing jitter, time step)
- Event statistics (total events, total photons processed)
- Configuration settings (baseline range, etc.)

**Usage Example:**
```python
import h5py
import numpy as np

# Load simulation results
with h5py.File('sipm_waveforms.h5', 'r') as f:
    waveforms = f['waveforms'][:]  # Shape: (n_events, n_time_points)
    time_axis = f['time_axis'][:]   # Time in ns
    n_events = f['metadata'].attrs['n_events']
    quantum_eff = f['metadata'].attrs['quantum_efficiency']
```
