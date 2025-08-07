#!/bin/bash

#SBATCH -p a100                # Use the A100 partition
#SBATCH --gres=gpu:2          # Request 2 GPUs
#SBATCH --nodes=1             # Use 1 compute node
#SBATCH --time=60:00:00       # Set maximum runtime to 60 hours
#SBATCH --job-name=dp_cifar_train  # Job name for Slurm
#SBATCH --output=dp_job_output.log # Output log file
#SBATCH --mem=80G             # Request 80 GB of memory

set -e  # Exit immediately on error

# Activate Python environment that you generated before submitting the job
source ~/.bashrc
conda activate opacus

# Change to working directory
cd /home/sz1c24/opacus/examples

# Run the training script
python resiclip.py
