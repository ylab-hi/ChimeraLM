#!/bin/bash

# Script to run evaluation results extraction for multiple prediction blocks
# Usage: ./batch_eval_results.sh

set -e  # Exit on any error

# Configuration
prefix="chtransformer_p2_586360_p2"
block=(1 2 3 4 5 6 7 8 9 10 11 12 13)

# Define script path
SCRIPT_PATH="scripts/get_result_from_predictions.py"

# Check if the Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi

echo "Starting batch evaluation for prefix: $prefix"
echo "Processing blocks: ${block[*]}"
echo "----------------------------------------"

# Loop through each block
for b in "${block[@]}"; do
    # Construct folder and output paths
    folder_1="${prefix}_${b}"
    prefix_1="${prefix}_${b}"

    predictions_dir="logs/eval/runs/${folder_1}/predicts/0"
    output_file="logs/eval/runs/${prefix_1}/predicts.txt"

    echo "Processing block $b..."
    echo "  Folder: $folder_1"
    echo "  Predictions dir: $predictions_dir"
    echo "  Output file: $output_file"

    # Check if predictions directory exists
    if [ ! -d "$predictions_dir" ]; then
        echo "  Warning: Predictions directory not found, skipping block $b"
        echo "  Missing: $predictions_dir"
        continue
    fi

    # Create output directory if it doesn't exist
    output_dir=$(dirname "$output_file")
    mkdir -p "$output_dir"

    # Run the command
    echo "  Running evaluation..."
    if uv run python "$SCRIPT_PATH" "$predictions_dir" "$output_file"; then
        echo "  ✓ Block $b completed successfully"
    else
        echo "  ✗ Block $b failed"
        exit 1
    fi

    echo "  Results saved to: $output_file"
    echo ""
done

echo "----------------------------------------"
echo "Batch evaluation completed successfully!"
echo "Processed ${#block[@]} blocks for prefix: $prefix"
