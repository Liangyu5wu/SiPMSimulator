#!/bin/bash
#
# Submit SiPM Position Scan Jobs
#
# Usage examples:
#   ./jobs/submit_position_scan.sh                    # Use default parameters
#   ./jobs/submit_position_scan.sh -5.0 -2.0 3.0 6.0 0.4    # Custom: x_start x_end y_start y_end step
#

# Set default parameters
X_START=${1:-"-5.0"}
X_END=${2:-"-2.0"}
Y_START=${3:-"3.0"}
Y_END=${4:-"6.0"}
STEP=${5:-"0.4"}

echo "Submitting SiPM position scan job..."
echo "Parameters:"
echo "  X range: ${X_START} to ${X_END} cm"
echo "  Y range: ${Y_START} to ${Y_END} cm"
echo "  Step size: ${STEP} cm"

# Calculate expected number of positions
python3 -c "
import numpy as np
x_positions = np.arange($X_START, $X_END + $STEP/2, $STEP)
y_positions = np.arange($Y_START, $Y_END + $STEP/2, $STEP)
total = len(x_positions) * len(y_positions)
print(f'Expected positions: {total}')
print(f'X grid: {len(x_positions)} points')
print(f'Y grid: {len(y_positions)} points')
"

echo ""
echo "Submitting to SLURM..."

sbatch --export=X_START=$X_START,X_END=$X_END,Y_START=$Y_START,Y_END=$Y_END,STEP=$STEP jobs/sipm_position_scan.sh

echo "Job submitted! Check status with: squeue -u \$USER"
echo "Monitor output: tail -f ../logs/output_sipm_scan-*.txt"