#!/bin/bash
#
#SBATCH --account=atlas:default
#SBATCH --partition=roma
#SBATCH --job-name=sipm_scan
#SBATCH --output=../logs/output_sipm_scan-%j.txt
#SBATCH --error=../logs/error_sipm_scan-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=08:00:00

# =============================================================================
# SiPM Position Scanning Script
# 
# Usage:
#   sbatch --export=X_START=-5.0,X_END=-2.0,Y_START=3.0,Y_END=6.0,STEP=0.4 jobs/sipm_position_scan.sh
#   
# Default scan parameters if not specified:
#   X_START=-5.0, X_END=-2.0, Y_START=3.0, Y_END=6.0, STEP=0.4
#
# SiPM active area size: 0.3 cm (fixed)
# =============================================================================

# Set default scan parameters if not provided
X_START=${X_START:-"-5.0"}
X_END=${X_END:-"-2.0"}
Y_START=${Y_START:-"3.0"}
Y_END=${Y_END:-"6.0"}
STEP=${STEP:-"0.4"}

# SiPM active area size (half-width)
SIPM_HALF_SIZE=0.15

echo "=== SiPM Position Scan Parameters ==="
echo "X range: ${X_START} to ${X_END} cm"
echo "Y range: ${Y_START} to ${Y_END} cm" 
echo "Step size: ${STEP} cm"
echo "SiPM size: ±${SIPM_HALF_SIZE} cm"
echo "====================================="

# Change to SiPM simulator directory
cd /fs/ddn/sdf/group/atlas/d/liangyu/dSiPM/fasttiming/SiPMSimulator

# Setup environment
source setup.sh

# Function to run simulation at specific position
run_position() {
    local x_center=$1
    local y_center=$2
    
    # Calculate x and y ranges
    local x_min=$(python3 -c "print(f'{$x_center - $SIPM_HALF_SIZE:.2f}')")
    local x_max=$(python3 -c "print(f'{$x_center + $SIPM_HALF_SIZE:.2f}')")
    local y_min=$(python3 -c "print(f'{$y_center - $SIPM_HALF_SIZE:.2f}')")
    local y_max=$(python3 -c "print(f'{$y_center + $SIPM_HALF_SIZE:.2f}')")
    
    echo "Processing position: x=[${x_min}, ${x_max}], y=[${y_min}, ${y_max}]"
    
    # Create temporary config file
    local temp_config="configs/temp_scan_x${x_center}_y${y_center}.yaml"
    cp configs/default.yaml $temp_config
    
    # Update x_range and y_range in temporary config
    python3 -c "
import yaml
with open('$temp_config', 'r') as f:
    config = yaml.safe_load(f)
config['photon_filter']['x_range'] = [$x_min, $x_max]
config['photon_filter']['y_range'] = [$y_min, $y_max]
with open('$temp_config', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)
"
    
    # Run simulation
    python scripts/run_simulation.py --config $temp_config
    
    # Clean up temporary config
    rm $temp_config
    
    echo "Completed position: x=[${x_min}, ${x_max}], y=[${y_min}, ${y_max}]"
}

# Main scanning loop
echo "Starting position scan..."

# Generate scan positions using Python
python3 -c "
import numpy as np
x_positions = np.arange($X_START, $X_END + $STEP/2, $STEP)
y_positions = np.arange($Y_START, $Y_END + $STEP/2, $STEP)

total_positions = len(x_positions) * len(y_positions)
print(f'Total scan positions: {total_positions}')
print('Scan grid:')
print(f'X positions: {x_positions.tolist()}')  
print(f'Y positions: {y_positions.tolist()}')

# Write positions to file for bash to read
with open('/tmp/scan_positions.txt', 'w') as f:
    for x in x_positions:
        for y in y_positions:
            f.write(f'{x:.2f} {y:.2f}\n')
"

# Read positions and run simulations
position_count=0
total_positions=$(wc -l < /tmp/scan_positions.txt)

while IFS=' ' read -r x_pos y_pos; do
    position_count=$((position_count + 1))
    echo ""
    echo "=== Position ${position_count}/${total_positions} ==="
    run_position $x_pos $y_pos
done < /tmp/scan_positions.txt

# Clean up
rm -f /tmp/scan_positions.txt

echo ""
echo "=== Scan Complete ==="
echo "Processed ${total_positions} positions"
echo "Results saved in ../output/ with timestamps"