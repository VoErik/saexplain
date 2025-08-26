#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

## --- Configuration ---
# Path where source data is stored.
DATA_PREFIX="../../data/"
DATA_SUFFIX=".tgz"
# Directory in the current folder for extracting training data.
TARGET_DIR="data"
# Directory in the current folder where Python script saves output.
OUTPUT_DIR_TO_ARCHIVE="out"
# Final destination for the results archive.
RESULTS_DEST_DIR="${DATA_PREFIX}results"

# --- Internal Variables ---
COPIED_ARCHIVES=()
SOURCE_FILENAMES=("$@")

## --- Cleanup Function ---
# This function runs on any script exit to ensure temporary files are removed.
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

# The 'trap' command registers the 'cleanup' function to run on EXIT.
trap cleanup EXIT

## --- Main Script ---

# 1. Validate that at least one filename has been provided.
if [ "$#" -eq 0 ]; then
    echo "❌ Error: No source filenames provided."
    echo "Usage: $0 filename1 [filename2 ...]"
    echo "Example: $0 fitzpatrick17k another_dataset"
    exit 1
fi

# 2. Prepare data.
# (The data preparation logic remains the same)
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

# 3. Run the training script.
echo "🚀 Running the Python training script..."
python training_scripts/finetune_custom_mae.py
echo "✅ Training script finished!"


# 4. Archive and move results.
echo "🏁 Starting post-training tasks..."
if [ -d "$OUTPUT_DIR_TO_ARCHIVE" ]; then
    # Create a unique filename for the archive with a timestamp.
    output_archive_name="results_$(date +%Y-%m-%d_%H-%M-%S).tgz"

    echo "📦 Archiving output folder '$OUTPUT_DIR_TO_ARCHIVE' to '$output_archive_name'..."
    # -c: create, -z: gzip, -f: file
    tar -czf "$output_archive_name" "$OUTPUT_DIR_TO_ARCHIVE"

    echo "Ensuring results destination exists: $RESULTS_DEST_DIR"
    # -p creates parent directories as needed and doesn't error if it exists.
    mkdir -p "$RESULTS_DEST_DIR"

    echo "🚚 Moving '$output_archive_name' to '$RESULTS_DEST_DIR/'..."
    mv "$output_archive_name" "$RESULTS_DEST_DIR/"
else
    echo "⚠️ Output directory '$OUTPUT_DIR_TO_ARCHIVE' not found, skipping archiving."
fi

echo "✨ All tasks complete!"
# The 'trap' will now automatically call the cleanup function.
