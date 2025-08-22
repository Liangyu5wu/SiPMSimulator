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

## Usage

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

## Configuration

Edit `configs/default.yaml` to adjust:
- Number of events to process
- Quantum efficiency (default: 30%)
- Timing jitter (default: 0.2 ns)
- SiPM active area (default: x[-4.25,3.95], y[4.32,4.62])
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
