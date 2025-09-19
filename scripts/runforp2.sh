#!/bin/bash

# Configuration variables

# CNN
# /gpfs/projects/b1171/ylk4626/project/Chimera/logs/train/runs/2025-07-11_13-29-37/checkpoints/epoch_014_f1_0.8763.ckpt

# Hyena
# /gpfs/projects/b1171/ylk4626/project/Chimera/logs/train/runs/2025-07-11_20-45-24/checkpoints/epoch_014_f1_0.8708.ckpt

# Transformer
# /gpfs/projects/b1171/ylk4626/project/Chimera/logs/train/runs/2025-07-14_10-29-45/checkpoints/epoch_011_f1_0.8705.ckpt

CKPT_PATH=/gpfs/projects/b1171/ylk4626/project/Chimera/logs/train/runs/2025-07-14_10-29-45/checkpoints/epoch_011_f1_0.8705.ckpt
MODEL=transformer
EXPERIMENT_PREFIX=chtransformer_p2_586360_p2

TRAINER=gpu
NUM_WORKERS=30
BATCH_SIZE=24
BASE_DATA_PATH=data/raw/PC3_10_cells_MDA_P2_dirty.chimeric.fq_chunks
BASE_OUTPUT_DIR=logs/eval/runs

# Array of chunk numbers to process
CHUNKS=(1 2 3 4 5 6 7 8 9 10 11 12 13)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Function to run evaluation for a specific chunk
run_eval() {
    local chunk_num=$1
    local data_file="${BASE_DATA_PATH}/PC3_10_cells_MDA_P2_dirty.chimeric_${chunk_num}.parquet"
    local output_dir="${BASE_OUTPUT_DIR}/${EXPERIMENT_PREFIX}_${chunk_num}"
    
    log "Starting evaluation for chunk ${chunk_num}..."
    log "Data file: ${data_file}"
    log "Output directory: ${output_dir}"
    
    # Check if data file exists (optional)
    if [[ ! -f "$data_file" ]]; then
        warn "Data file does not exist: $data_file"
    fi
    
    # Run the evaluation
    uv run eval.py \
        ckpt_path=$CKPT_PATH \
        model=$MODEL \
        trainer=$TRAINER \
        data.num_workers=$NUM_WORKERS \
        data.batch_size=$BATCH_SIZE \
        +data.predict_data_path=$data_file \
        paths.output_dir=$output_dir \
        paths.log_dir=$output_dir
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        log "Successfully completed evaluation for chunk ${chunk_num}"
    else
        error "Evaluation failed for chunk ${chunk_num} with exit code ${exit_code}"
        return $exit_code
    fi
}

# Main execution
main() {
    log "Starting batch evaluation script"
    log "Processing ${#CHUNKS[@]} chunks: ${CHUNKS[*]}"
    
    local failed_chunks=()
    local success_count=0
    
    for chunk in "${CHUNKS[@]}"; do
        if run_eval "$chunk"; then
            ((success_count++))
        else
            failed_chunks+=("$chunk")
        fi
        echo "----------------------------------------"
    done
    
    # Summary
    log "Batch evaluation completed"
    log "Successful runs: ${success_count}/${#CHUNKS[@]}"
    
    if [[ ${#failed_chunks[@]} -gt 0 ]]; then
        error "Failed chunks: ${failed_chunks[*]}"
        exit 1
    else
        log "All evaluations completed successfully!"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --dry-run    Show commands without executing"
            echo "  --parallel   Run evaluations in parallel (experimental)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Dry run mode
if [[ "$DRY_RUN" == "true" ]]; then
    log "DRY RUN MODE - Commands that would be executed:"
    for chunk in "${CHUNKS[@]}"; do
        echo "Chunk $chunk:"
        echo "  uv run eval.py ckpt_path=\"$CKPT_PATH\" model=\"$MODEL\" trainer=\"$TRAINER\" data.num_workers=\"$NUM_WORKERS\" data.batch_size=\"$BATCH_SIZE\" +data.predict_data_path=\"${BASE_DATA_PATH}/PC3_10_cells_MDA_P2_dirty.chimeric_${chunk}.parquet\" paths.output_dir=\"${BASE_OUTPUT_DIR}/${EXPERIMENT_PREFIX}_${chunk}\" paths.log_dir=\"${BASE_OUTPUT_DIR}/${EXPERIMENT_PREFIX}_${chunk}\""
        echo ""
    done
    exit 0
fi

# Run main function
main
