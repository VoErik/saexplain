#!/bin/bash

set -e

## --- Configuration ---
DATA_PREFIX="../../data/"
DATA_SUFFIX=".tgz"
TARGET_DIR="data"
OUTPUT_DIR_TO_ARCHIVE="out"
RESULTS_DEST_DIR="${DATA_PREFIX}results"

# --- Internal Variables ---
COPIED_ARCHIVES=()
SOURCE_FILENAMES=("$@")

export CUDA_VISIBLE_DEVICES=1

## --- Cleanup Function ---
cleanup() {
    echo "🧹 Executing cleanup..."
    echo "🗑️ Removing training data directory: $TARGET_DIR"
    rm -rf "$TARGET_DIR"

    echo "🗑️ Removing output directory: $OUTPUT_DIR_TO_ARCHIVE"
    rm -rf "$OUTPUT_DIR_TO_ARCHIVE"

    for archive in "${COPIED_ARCHIVES[@]}"; do
        echo "🗑️ Removing temporary archive: $archive"
        rm -f "$archive"
    done
}

trap cleanup EXIT

## --- Main Script ---
if [ "$#" -eq 0 ]; then
    echo "❌ Error: No source filenames provided."
    echo "Usage: $0 filename1 [filename2 ...]"
    echo "Example: $0 fitzpatrick17k another_dataset"
    exit 1
fi

echo "➡️ Copying archives into the current directory..."
for filename in "${SOURCE_FILENAMES[@]}"; do
    source_file="${DATA_PREFIX}${filename}${DATA_SUFFIX}"
    archive_name="${filename}${DATA_SUFFIX}"
    if [ -f "$source_file" ]; then
        cp "$source_file" .
        COPIED_ARCHIVES+=("$archive_name")
    else
        echo "⚠️ Warning: Source file not found: $source_file"
    fi
done
if [ ${#COPIED_ARCHIVES[@]} -eq 0 ]; then
    echo "❌ No archives found to process. Exiting."
    exit 1
fi
echo "📦 Creating target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"
for archive in "${COPIED_ARCHIVES[@]}"; do
    echo " unpacking $archive into $TARGET_DIR..."
    tar --warning=no-unknown-keyword -xf "$archive" -C "$TARGET_DIR"
done

echo "🚀 Running the Python training script..."
python train_sae.py
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