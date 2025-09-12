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
x_bounds = np.arange($X_START, $X_END + $STEP/2, $STEP)
y_bounds = np.arange($Y_START, $Y_END + $STEP/2, $STEP)
total = len(x_bounds) * len(y_bounds)
print(f'Expected scan regions: {len(x_bounds)} × {len(y_bounds)} = {total}')
print(f'Each region size: 0.3 × 0.3 cm')
print(f'First region: x[{x_bounds[0]:.2f}, {x_bounds[0]+0.3:.2f}], y[{y_bounds[0]:.2f}, {y_bounds[0]+0.3:.2f}]')
if total > 1:
    print(f'Last region: x[{x_bounds[-1]:.2f}, {x_bounds[-1]+0.3:.2f}], y[{y_bounds[-1]:.2f}, {y_bounds[-1]+0.3:.2f}]')
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
    
    # Generate and process scan regions
    python3 -c "
import numpy as np
import yaml
import os

# Generate x and y lower bounds for each scan region
x_lower_bounds = np.arange(float('$X_START'), float('$X_END') + float('$STEP')/2, float('$STEP'))
y_lower_bounds = np.arange(float('$Y_START'), float('$Y_END') + float('$STEP')/2, float('$STEP'))
total = len(x_lower_bounds) * len(y_lower_bounds)

print(f'Processing {total} scan regions...')

for i, x_min in enumerate(x_lower_bounds):
    for j, y_min in enumerate(y_lower_bounds):
        pos_num = i * len(y_lower_bounds) + j + 1
        
        # Calculate region bounds: each region is 0.3 x 0.3 cm
        x_max = float(x_min + 2 * $SIPM_HALF_SIZE)
        y_max = float(y_min + 2 * $SIPM_HALF_SIZE)
        
        print(f'Region {pos_num}/{total}: x[{x_min:.2f}, {x_max:.2f}], y[{y_min:.2f}, {y_max:.2f}]')
        
        # Create temp config
        config_name = f'configs/temp_scan_x{x_min:.2f}_y{y_min:.2f}.yaml'
        with open('configs/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['photon_filter']['x_range'] = [float(x_min), float(x_max)]
        config['photon_filter']['y_range'] = [float(y_min), float(y_max)]
        with open(config_name, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Run simulation
        os.system(f'python scripts/run_simulation.py --config {config_name}')
        os.remove(config_name)

print(f'Scan complete: {total} regions processed')
"
fi