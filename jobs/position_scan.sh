#!/bin/bash
#
# SiPM Position Scanning Script - Unified submission and execution
#
# Usage examples:
#   ./jobs/position_scan.sh                              # Default scan
#   ./jobs/position_scan.sh -6.65 -1.85 1.12 5.92 0.4   # Custom scan
#
# Parameters: x_start x_end y_start y_end step_size

# Set scan parameters
X_START=${1:-"-6.65"}
X_END=${2:-"-1.85"}
Y_START=${3:-"1.12"}
Y_END=${4:-"5.92"}
STEP=${5:-"0.4"}
SIPM_HALF_SIZE=0.15

if [[ -z "$SLURM_JOB_ID" ]]; then
    #=== SUBMISSION MODE ===
    echo "Submitting SiPM position scan job..."
    echo "Parameters: X[${X_START}, ${X_END}], Y[${Y_START}, ${Y_END}], step=${STEP}"
    
    # Show expected grid
    python3 -c "
import numpy as np
x_pos = np.arange($X_START, $X_END + $STEP/2, $STEP)
y_pos = np.arange($Y_START, $Y_END + $STEP/2, $STEP)
print(f'Expected positions: {len(x_pos)} × {len(y_pos)} = {len(x_pos)*len(y_pos)}')
"
    
    # Create and submit SLURM job
    TEMP_SCRIPT=$(mktemp)
    cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --account=atlas:default
#SBATCH --partition=roma
#SBATCH --job-name=sipm_scan
#SBATCH --output=../logs/output_sipm_scan-%j.txt
#SBATCH --error=../logs/error_sipm_scan-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=08:00:00

exec "$PWD/jobs/position_scan.sh" "$X_START" "$X_END" "$Y_START" "$Y_END" "$STEP"
EOF
    
    sbatch "$TEMP_SCRIPT"
    rm -f "$TEMP_SCRIPT"
    echo "Job submitted! Monitor: squeue -u \$USER"

else
    #=== EXECUTION MODE ===
    echo "=== SiPM Position Scan ==="
    echo "X[${X_START}, ${X_END}], Y[${Y_START}, ${Y_END}], step=${STEP}"
    
    cd /fs/ddn/sdf/group/atlas/d/liangyu/dSiPM/fasttiming/SiPMSimulator
    source setup.sh
    
    # Generate and process positions
    python3 -c "
import numpy as np
import yaml
import os

x_positions = np.arange($X_START, $X_END + $STEP/2, $STEP)
y_positions = np.arange($Y_START, $Y_END + $STEP/2, $STEP)
total = len(x_positions) * len(y_positions)

print(f'Processing {total} positions...')

for i, x in enumerate(x_positions):
    for j, y in enumerate(y_positions):
        pos_num = i * len(y_positions) + j + 1
        print(f'Position {pos_num}/{total}: center=({x:.2f}, {y:.2f})')
        
        # Calculate bounds and convert to native Python float
        x_min, x_max = float(x - $SIPM_HALF_SIZE), float(x + $SIPM_HALF_SIZE)
        y_min, y_max = float(y - $SIPM_HALF_SIZE), float(y + $SIPM_HALF_SIZE)
        
        # Create temp config
        config_name = f'configs/temp_scan_x{x:.2f}_y{y:.2f}.yaml'
        with open('configs/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['photon_filter']['x_range'] = [x_min, x_max]
        config['photon_filter']['y_range'] = [y_min, y_max]
        with open(config_name, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Run simulation
        os.system(f'python scripts/run_simulation.py --config {config_name}')
        os.remove(config_name)

print(f'Scan complete: {total} positions processed')
"
fi