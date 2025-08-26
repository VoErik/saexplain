#!/bin/bash

set -e

PYTHON_ARGS=""
if [[ "$1" == "--config" ]]; then
    PYTHON_ARGS="$1 $2"
fi

## --- Configuration (Inside the Container) ---
DATA_DIR="/app/data"
RESULTS_DEST_DIR="/app/results"
OUTPUT_DIR_TO_ARCHIVE="out"

## --- Cleanup Function ---
cleanup() {
    echo "🧹 Executing cleanup inside container..."
    echo "🗑️ Removing temporary output directory: $OUTPUT_DIR_TO_ARCHIVE"
    rm -rf "$OUTPUT_DIR_TO_ARCHIVE"
}
trap cleanup EXIT

## --- Main Script ---
echo "🚀 Data is pre-mounted. Running the Python training script..."
python train_mae.py $PYTHON_ARGS
echo "✅ Training script finished!"

echo "🏁 Starting post-training tasks..."
if [ -d "$OUTPUT_DIR_TO_ARCHIVE" ]; then
    output_archive_name="results_$(date +%Y-%m-%d_%H-%M-%S).tgz"
    echo "📦 Archiving output folder '$OUTPUT_DIR_TO_ARCHIVE' to '$output_archive_name'..."
    tar -czf "$output_archive_name" "$OUTPUT_DIR_TO_ARCHIVE"
    echo "Ensuring results destination exists: $RESULTS_DEST_DIR"
    mkdir -p "$RESULTS_DEST_DIR"
    echo "🚚 Moving '$output_archive_name' to '$RESULTS_DEST_DIR/'..."
    mv "$output_archive_name" "$RESULTS_DEST_DIR/"
else
    echo "⚠️ Output directory '$OUTPUT_DIR_TO_ARCHIVE' not found, skipping archiving."
fi

echo "✨ All tasks complete!"
