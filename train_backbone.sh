#!/bin/bash

set -e

## --- Configuration ---
DATA_PREFIX="../../data/"
DATA_SUFFIX=".tgz"
TARGET_DIR="data"
OUTPUT_DIR_TO_ARCHIVE="out"
RESULTS_DEST_DIR="${DATA_PREFIX}results"

declare -A PYTHON_SCRIPTS
PYTHON_SCRIPTS["dino"]="train_dino.py"
PYTHON_SCRIPTS["mae"]="train_mae.py"
PYTHON_SCRIPTS["clip"]="train_clip.py"
PYTHON_SCRIPTS["resnet"]="train_resnet.py"

# --- Internal Variables ---
COPIED_ARCHIVES=()
TASK_NAME=""
PROCESS_DATASET=true
DATASET_FILENAMES=()

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

## --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --task)
        TASK_NAME="$2"
        shift
        shift
        ;;
        --no-dataset)
        PROCESS_DATASET=false
        shift
        ;;
        *)
        DATASET_FILENAMES+=("$1")
        shift
        ;;
    esac
done


## --- Main Script ---
if [ -z "$TASK_NAME" ]; then
    echo "❌ Error: --task argument not provided."
    echo "Available tasks: ${!PYTHON_SCRIPTS[@]}"
    exit 1
fi
if [[ ! -v "PYTHON_SCRIPTS[$TASK_NAME]" ]]; then
    echo "❌ Error: Invalid task name '$TASK_NAME'."
    echo "Available tasks: ${!PYTHON_SCRIPTS[@]}"
    exit 1
fi

if [ "$PROCESS_DATASET" = true ]; then
    echo "ℹ️ Data processing is enabled."
    if [ ${#DATASET_FILENAMES[@]} -eq 0 ]; then
        echo "❌ Error: No source dataset filenames provided."
        echo "Usage: $0 --task <task_name> dataset1 [dataset2 ...]"
        echo "Example: $0 --task dino fitzpatrick17k"
        echo "Or use --no-dataset if the script doesn't need data."
        exit 1
    fi

    echo "➡️ Copying archives into the current directory..."
    for filename in "${DATASET_FILENAMES[@]}"; do
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
else
    echo "ℹ️ --no-dataset flag is set, skipping data preparation."
fi

SCRIPT_TO_RUN="${PYTHON_SCRIPTS[$TASK_NAME]}"
echo "🚀 Running the Python script for task '$TASK_NAME': $SCRIPT_TO_RUN..."
python "$SCRIPT_TO_RUN"
echo "✅ Python script finished!"

echo "🏁 Starting post-training tasks..."
if [ -d "$OUTPUT_DIR_TO_ARCHIVE" ]; then
    output_archive_name="results_${TASK_NAME}_$(date +%Y-%m-%d_%H-%M-%S).tgz"

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