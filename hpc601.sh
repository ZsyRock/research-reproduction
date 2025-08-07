#!/bin/bash

#SBATCH -p a100                # Use the A100 partition
#SBATCH --gres=gpu:2          # Request 2 GPUs
#SBATCH --nodes=1             # Use 1 compute node
#SBATCH --time=60:00:00       # Set maximum runtime to 60 hours
#SBATCH --job-name=dp_cifar_train  # Job name for Slurm
#SBATCH --output=dp_job_output.log # Output log file
#SBATCH --mem=80G             # Request 80 GB of memory

set -e  # Exit immediately if a command exits with a non-zero status

# Activate environment
source ~/.bashrc              # Load user environment
conda activate opacus         # Activate the 'opacus' Conda environment

# Change to working directory
cd /home/sz1c24/opacus/examples  # Navigate to the working directory

# General training parameters
BATCH=256
EPOCHS=90
MAX_NORM=8.0
TARGET_EPS=4.0
DELTA=1e-5
DATA_ROOT=./cifar10
DEVICE=cuda
PRINTFREQ=10
LOG_DIR="./logs"
CKPT_DIR="./checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"  # Create log and checkpoint directories if they don't exist

# Auto-incrementing run ID function
get_next_id() {
    METHOD="$1"
    LAST_ID=$(find "$LOG_DIR" -maxdepth 1 -type f -name "${METHOD}_run_*.log" | sed -E "s/.*${METHOD}_run_([0-9]+)\.log/\1/" | sort -n | tail -n 1)
    if [[ -z "$LAST_ID" ]]; then
        echo "001"  # Return 001 if no previous runs
    else
        printf "%03d" $((10#$LAST_ID + 1))  # Increment the last ID and format as 3-digit number
    fi
}

# Free GPU memory function
free_gpu() {
    echo "=== Freeing GPU memory ==="
    kill -9 $(nvidia-smi | awk '$5=="C"{print $3}') 2>/dev/null || echo "No lingering GPU processes."  # Kill compute processes on GPU
    sleep 3
    nvidia-smi  # Show updated GPU status
}

# Run AdaClip training 5 times
for i in {1..5}; do
    RUN_ID=$(get_next_id "adaclip")
    echo "=== [AdaClip] Run #$RUN_ID ==="

    python cifar10.py \
      --batch-size $BATCH \
      --epochs $EPOCHS \
      --target-epsilon $TARGET_EPS \
      --delta $DELTA \
      --max-per-sample-grad_norm $MAX_NORM \
      --clipping adaptive \
      --checkpoint-file ${CKPT_DIR}/adaclip_ckpt_${RUN_ID}.tar \
      --log-dir "$LOG_DIR" \
      --data-root "$DATA_ROOT" \
      --device $DEVICE \
      --print-freq $PRINTFREQ \
      --debug 0 \
      --grad_sample_mode hooks \
      --workers 0 | tee "$LOG_DIR/adaclip_run_${RUN_ID}.log"  # Save both stdout and log file

    free_gpu  # Release GPU after each run
done

# Run ResiClip training 5 times
for i in {1..5}; do
    RUN_ID=$(get_next_id "resiclip")
    echo "=== [ResiClip] Run #$RUN_ID ==="

    python resiclip_cifar10.py \
      --batch-size $BATCH \
      --epochs $EPOCHS \
      --target-epsilon $TARGET_EPS \
      --delta $DELTA \
      --max-per-sample-grad_norm $MAX_NORM \
      --clipping resiclip \
      --checkpoint-file ${CKPT_DIR}/resiclip_ckpt_${RUN_ID}.tar \
      --log-dir "$LOG_DIR" \
      --data-root "$DATA_ROOT" \
      --device $DEVICE \
      --print-freq $PRINTFREQ \
      --debug 0 \
      --grad_sample_mode hooks \
      --workers 0 | tee "$LOG_DIR/resiclip_run_${RUN_ID}.log"  # Save both stdout and log file

    free_gpu  # Release GPU after each run
done

echo "All training tasks completed!"
