#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
IMAGE_NAME="thesis-training"
DATA_DIR="$(pwd)/../data"
RESULTS_DIR="$(pwd)/../results"

# --- Default Values ---
GPU_DEVICE_ID=1
DATASET_ARGS=()
CONFIG_ARG=""

# --- Argument Parsing ---
while (( "$#" )); do
  case "$1" in
    -g|--gpu)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        GPU_DEVICE_ID=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -c|--config)
          if [ -n "$2" ] && [ -f "$2" ]; then
            CONFIG_FILE_PATH=$(realpath "$2")
            CONFIG_ARG="--config /app/configs/config.yaml"
            shift 2
          else
            echo "Error: --config requires a valid file path." >&2
            exit 1
          fi
          ;;
    -*|--*=)
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *)
      DATASET_ARGS+=("$1")
      shift
      ;;
  esac
done

# --- Main Script ---
if [ -z "$WANDB_API_KEY" ]; then
    echo "❌ Error: WANDB_API_KEY environment variable is not set."
    echo "Please set it before running the script:"
    echo "export WANDB_API_KEY=\"your_wandb_api_key_here\""
    exit 1
fi
if [ ${#DATASET_ARGS[@]} -eq 0 ]; then
    echo "❌ Error: No source filenames provided."
    echo "Usage: $0 [-g GPU_ID] filename1 [filename2 ...]"
    echo "Example: $0 -g 0 fitzpatrick17k"
    exit 1
fi
echo "🔎 Checking for host directories at '$DATA_DIR' and '$RESULTS_DIR'..."
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found at '$DATA_DIR'"
    echo "Please ensure you are running this script from the 'thesis' directory and the 'data' directory exists alongside it."
    exit 1
fi
mkdir -p "$RESULTS_DIR"
echo "👍 Host directories are ready."

echo -e "\n🛠️  Building Docker image '$IMAGE_NAME' from the current directory..."
docker build -t "$IMAGE_NAME" .
echo "✅ Image built successfully."

echo -e "\n🚀 Starting the training container on GPU ${GPU_DEVICE_ID}..."

docker run \
  --rm \
  --gpus "\"device=${GPU_DEVICE_ID}\"" \
  --cpus=16 \
  --memory=32g \
  --shm-size=4g \
  -v "$DATA_DIR:/app/source_data:ro" \
  -v "$RESULTS_DIR:/app/results" \
  -v "$CONFIG_FILE_PATH:/app/config.yaml:ro" \
  -e WANDB_API_KEY \
  "$IMAGE_NAME" "$CONFIG_ARG" "${DATASET_ARGS[@]}"

echo -e "\n✨ Docker container has finished execution."
echo "🎉 Check for your output in the '$RESULTS_DIR' directory."
