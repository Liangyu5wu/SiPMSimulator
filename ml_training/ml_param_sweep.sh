#!/bin/bash
#
#SBATCH --account=atlas:default
#SBATCH --partition=roma
#SBATCH --job-name=ml_param_sweep
#SBATCH --output=../logs/output_ml_param_sweep-%A_%a.txt
#SBATCH --error=../logs/error_ml_param_sweep-%A_%a.txt
#SBATCH --array=0-29
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=08:00:00

# Change to SiPM simulator directory
cd /fs/ddn/sdf/group/atlas/d/liangyu/dSiPM/fasttiming/SiPMSimulator

# Setup environment
source setup.sh

# Run parameter sweep for this job array task
echo "Starting job array task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_MAX}"
python ml_training/param_sweep.py

echo "Job ${SLURM_ARRAY_TASK_ID} completed"