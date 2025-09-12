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

1. **Submit job**:
```bash
sbatch jobs/sipm_sim.sh
```

## Configuration

Edit `configs/default.yaml` to adjust:
- Number of events to process
- Quantum efficiency (default: 30%)
- Timing jitter (default: 0.2 ns)
- SiPM active area (default: x[-4.25,-3.95], y[4.32,4.62])
- Output directory and format

## Physics

The simulation implements realistic SiPM response modeling:

1. **Photon Filtering**: Selects photons by position (SiPM active area) and type (Cherenkov core)
2. **Quantum Efficiency**: Applies detection probability based on configured QE
3. **Timing Jitter**: Adds Gaussian timing uncertainty to photon arrival times
4. **Pulse Convolution**: Convolves detected photons with SiPM pulse shape
5. **Noise Addition**: Adds realistic Gaussian background noise
6. **Output Generation**: Produces time-domain waveforms with metadata

Results are saved in timestamped directories under `../output/run_YYYYMMDD_HHMMSS/` containing waveforms, plots, and metadata.
