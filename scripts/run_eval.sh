#!/bin/bash

# Script to generate and run evaluation command for Chimera model
# Usage: ./run_eval.sh <input_parquet_file> [num_workers] [batch_size]

# Default values
DEFAULT_NUM_WORKERS=30
DEFAULT_BATCH_SIZE=24

# Check if input file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_parquet_file> [num_workers] [batch_size]"
    echo "Example: $0 data/short_read/mda/SRR11563612/SRR11563612_1.parquet 16 32"
    echo "Default num_workers: $DEFAULT_NUM_WORKERS"
    echo "Default batch_size: $DEFAULT_BATCH_SIZE"
    exit 1
fi

# Get the input file path
INPUT_FILE="$1"

# Get optional parameters or use defaults
NUM_WORKERS="${2:-$DEFAULT_NUM_WORKERS}"
BATCH_SIZE="${3:-$DEFAULT_BATCH_SIZE}"

# Extract filename without extension for output directory naming
FILENAME=$(basename "$INPUT_FILE" .parquet)

# Set variables
CHECKPOINT="/projects/b1171/ylk4626/project/Chimera/logs/train/multiruns/2025-02-08_21-24-35/0/checkpoints/epoch_010_f1_0.9347.ckpt"
OUTPUT_DIR="logs/eval/runs/mamba_${FILENAME}"
LOG_DIR="logs/eval/runs/mamba_${FILENAME}"

# Construct and run the command
CMD="uv run eval.py ckpt_path=${CHECKPOINT} trainer=gpu +trainer.precision=bf16-mixed data.num_workers=${NUM_WORKERS} data.batch_size=${BATCH_SIZE} +data.predict_data_path=${INPUT_FILE} paths.output_dir=${OUTPUT_DIR} paths.log_dir=${LOG_DIR}"

echo "Running evaluation with command:"
echo "$CMD"
echo "----------------------------------------"
echo "Input file: $INPUT_FILE"
echo "Num workers: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "----------------------------------------"

# Execute the command
eval "$CMD"
