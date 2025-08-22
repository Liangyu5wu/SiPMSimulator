# SiPM Simulator

SiPM simulation for particle physics experiments. Processes ROOT files containing optical photon data and simulates Silicon Photomultiplier response.

## Performance

This simulator has been optimized for high performance:
- **30x faster** array conversion from ROOT to numpy
- **Branch activation** for efficient ROOT I/O  
- **20-30x overall speedup** compared to original implementation
- Typical processing time: 2-3 seconds for 10 events (vs 60+ seconds originally)

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

## Usage

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

## Output

Results are saved in timestamped directories:
```
output/
└── run_YYYYMMDD_HHMMSS/
    ├── waveforms/          # SiPM waveform data
    ├── plots/              # 2D photon distributions
    └── metadata/           # Configuration and statistics
```

## Physics

1. Filters photons by position and type (Cherenkov core)
2. Applies quantum efficiency and timing jitter
3. Convolves with SiPM pulse shape
4. Adds Gaussian background noise
5. Outputs time-domain waveforms

## Optimization Details

### Key Performance Improvements

1. **ROOT Branch Activation**: Only loads needed data branches, reducing I/O by 50-70%
2. **Efficient Array Conversion**: Uses `np.array(ROOT_vector)` instead of element-by-element access
3. **Vectorized Operations**: Numpy operations replace Python loops where possible

### Performance Comparison

| Operation | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| Array conversion | 25.97s | 0.87s | 30x |
| Total processing | ~60s | ~2s | 30x |

### Resource Requirements

- **Memory**: 8GB sufficient for typical runs (10-1000 events)
- **Time**: 2-3 seconds per 10 events locally
- **Storage**: ~10-50MB per simulation run
