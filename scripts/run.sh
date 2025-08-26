#!/bin/bash

set -e

## --- Default Configuration ---
DATA_PREFIX="../../data/"
DATA_SUFFIX=".tgz"
TARGET_DIR="data"
OUTPUT_DIR_TO_ARCHIVE="out"
RESULTS_DEST_DIR="${DATA_PREFIX}results"

# --- Internal Variables ---
COPIED_ARCHIVES=()
SOURCE_FILENAMES=()
GPU_DEVICE=0
TASK_SCRIPT=""
CONFIG_FILE=""
EXTRA_ARGS=()
NO_CLEANUP=false

## --- Cleanup Function ---
cleanup() {
    if [ "$NO_CLEANUP" = true ]; then
        echo "🧹 Skipping cleanup (--no-cleanup enabled)"
        return
    fi

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
    case "$1" in
        --gpu)
            GPU_DEVICE="$2"
            shift 2
            ;;
        --datasets)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                SOURCE_FILENAMES+=("$1")
                shift
            done
            ;;
        --task)
            TASK_SCRIPT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DEST_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR_TO_ARCHIVE="$2"
            shift 2
            ;;
        --data-prefix)
            DATA_PREFIX="$2"
            shift 2
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            shift
            ;;
        --extra-args)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                EXTRA_ARGS+=("$1")
                shift
            done
            ;;
        *)
            echo "❌ Unknown argument: $1"
            exit 1
            ;;
    esac
done

## --- Validation ---
if [ -z "$TASK_SCRIPT" ]; then
    echo "❌ Error: No task script provided. Use --task path/to/script.py"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"

## --- Dataset Preparation (Optional) ---
if [ ${#SOURCE_FILENAMES[@]} -gt 0 ]; then
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

    if [ ${#COPIED_ARCHIVES[@]} -gt 0 ]; then
        echo "📦 Creating target directory: $TARGET_DIR"
        mkdir -p "$TARGET_DIR"
        for archive in "${COPIED_ARCHIVES[@]}"; do
            echo " unpacking $archive into $TARGET_DIR..."
            tar --warning=no-unknown-keyword -xf "$archive" -C "$TARGET_DIR"
        done
    else
        echo "⚠️ No dataset archives were copied. Continuing without datasets."
    fi
else
    echo "ℹ️ No datasets specified, skipping dataset preparation."
fi

## --- Run Task ---
echo "🚀 Running the Python task script..."
CMD=(python "$TASK_SCRIPT")
if [ -n "$CONFIG_FILE" ]; then
    CMD+=("--config" "$CONFIG_FILE")
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "🔧 Command: ${CMD[@]}"
"${CMD[@]}"
echo "✅ Task script finished!"

## --- Post-task Archiving ---
echo "🏁 Starting post-task archiving..."
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
