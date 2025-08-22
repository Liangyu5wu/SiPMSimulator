#!/bin/bash
#
#SBATCH --account=atlas:default
#SBATCH --partition=roma
#SBATCH --job-name=sipm_sim
#SBATCH --output=../logs/output_sipm_sim-%j.txt
#SBATCH --error=../logs/error_sipm_sim-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=08:00:00

# Change to SiPM simulator directory
cd /fs/ddn/sdf/group/atlas/d/liangyu/dSiPM/fasttiming/SiPMSimulator

# Setup environment
source setup.sh

python scripts/run_simulation.py --config configs/default.yaml
