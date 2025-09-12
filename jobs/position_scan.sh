#!/bin/bash
#
# SiPM Position Scanning Script - Individual job submission per region
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

# Check if this is a single region execution (has SCAN_X_MIN environment variable)
if [[ -n "$SCAN_X_MIN" ]]; then
    #=== SINGLE REGION EXECUTION MODE ===
    echo "=== SiPM Single Region Scan ==="
    echo "Region: x[${SCAN_X_MIN}, ${SCAN_X_MAX}], y[${SCAN_Y_MIN}, ${SCAN_Y_MAX}]"
    
    cd /fs/ddn/sdf/group/atlas/d/liangyu/dSiPM/fasttiming/SiPMSimulator
    source setup.sh
    
    # Create temporary config for this specific region
    config_name="configs/temp_scan_x${SCAN_X_MIN}_y${SCAN_Y_MIN}.yaml"
    python3 -c "
import yaml
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['photon_filter']['x_range'] = [float('$SCAN_X_MIN'), float('$SCAN_X_MAX')]
config['photon_filter']['y_range'] = [float('$SCAN_Y_MIN'), float('$SCAN_Y_MAX')]
with open('$config_name', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)
"
    
    # Run simulation for this region
    python scripts/run_simulation.py --config "$config_name"
    rm -f "$config_name"
    
    echo "Region scan complete: x[${SCAN_X_MIN}, ${SCAN_X_MAX}], y[${SCAN_Y_MIN}, ${SCAN_Y_MAX}]"

else
    #=== JOB SUBMISSION MODE ===
    echo "Submitting SiPM position scan jobs..."
    echo "Parameters: X[${X_START}, ${X_END}], Y[${Y_START}, ${Y_END}], step=${STEP}"
    
    # Calculate and display expected grid
    python3 -c "
import numpy as np
x_bounds = np.arange($X_START, $X_END + $STEP/2, $STEP)
y_bounds = np.arange($Y_START, $Y_END + $STEP/2, $STEP)
total = len(x_bounds) * len(y_bounds)
print(f'Will submit {total} individual jobs ({len(x_bounds)} × {len(y_bounds)})')
print(f'Each region size: 0.3 × 0.3 cm')
print(f'First region: x[{x_bounds[0]:.2f}, {x_bounds[0]+0.3:.2f}], y[{y_bounds[0]:.2f}, {y_bounds[0]+0.3:.2f}]')
if total > 1:
    print(f'Last region: x[{x_bounds[-1]:.2f}, {x_bounds[-1]+0.3:.2f}], y[{y_bounds[-1]:.2f}, {y_bounds[-1]+0.3:.2f}]')
"
    
    echo ""
    echo "Submitting individual jobs..."
    
    # Submit individual jobs for each scan region
    job_count=0
    python3 -c "
import numpy as np
import os
import subprocess

x_bounds = np.arange($X_START, $X_END + $STEP/2, $STEP)
y_bounds = np.arange($Y_START, $Y_END + $STEP/2, $STEP)
total = len(x_bounds) * len(y_bounds)

for i, x_min in enumerate(x_bounds):
    for j, y_min in enumerate(y_bounds):
        job_num = i * len(y_bounds) + j + 1
        
        # Calculate region bounds
        x_max = x_min + 2 * $SIPM_HALF_SIZE
        y_max = y_min + 2 * $SIPM_HALF_SIZE
        
        # Create individual SLURM job script
        job_script = f'''#!/bin/bash
#SBATCH --account=atlas:default
#SBATCH --partition=roma
#SBATCH --job-name=sipm_scan_{job_num:03d}
#SBATCH --output=../logs/output_sipm_scan_{job_num:03d}-%j.txt
#SBATCH --error=../logs/error_sipm_scan_{job_num:03d}-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=02:00:00

export SCAN_X_MIN={x_min:.3f}
export SCAN_X_MAX={x_max:.3f}
export SCAN_Y_MIN={y_min:.3f}
export SCAN_Y_MAX={y_max:.3f}

exec \"$PWD/jobs/position_scan.sh\"
'''
        
        # Write temporary job script and submit
        with open(f'/tmp/sipm_job_{job_num:03d}.sh', 'w') as f:
            f.write(job_script)
        
        # Submit the job
        result = subprocess.run(['sbatch', f'/tmp/sipm_job_{job_num:03d}.sh'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f'Job {job_num}/{total}: Region x[{x_min:.2f}, {x_max:.2f}], y[{y_min:.2f}, {y_max:.2f}] → Job ID {job_id}')
        else:
            print(f'Failed to submit job {job_num}: {result.stderr}')
        
        # Clean up temporary script
        os.remove(f'/tmp/sipm_job_{job_num:03d}.sh')
"
    
    echo ""
    echo "All jobs submitted! Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f ../logs/output_sipm_scan_*"
fi