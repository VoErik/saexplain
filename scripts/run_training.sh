#!/bin/bash

set -e

# --- Configuration ---
IMAGE_NAME="thesis-training"
DATA_DIR="$(pwd)/../data"
RESULTS_DIR="$(pwd)/../results"
PERMANENT_UNPACK_DIR="$(pwd)/../unpacked_data"

# --- Default Values ---
GPU_DEVICE_ID=1
DATASET_ARGS=()
CONFIG_ARG=""
CONFIG_FILE_PATH=""
PERFORM_CLEANUP=false

cleanup() {
    if [ "$PERFORM_CLEANUP" = true ]; then
        if [ -d "$STAGING_DATA_DIR" ]; then
            echo "🧹 --cleanup flag detected. Removing staging directory: $STAGING_DATA_DIR"
            rm -rf "$STAGING_DATA_DIR"
        fi
    else
        if [ -d "$STAGING_DATA_DIR" ]; then
            echo "ℹ️  Keeping staging directory for future runs: $STAGING_DATA_DIR"
        fi
    fi
}
trap cleanup EXIT


# --- Argument Parsing ---
while (( "$#" )); do
  case "$1" in
    --cleanup)
      PERFORM_CLEANUP=true
      shift
      ;;
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
            CONFIG_ARG="--config /app/config.yaml"
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
if [ -z "$WANDB_API_KEY" ]; then echo "❌ Error: WANDB_API_KEY not set."; exit 1; fi
if [ ${#DATASET_ARGS[@]} -eq 0 ]; then echo "❌ Error: No dataset names provided."; exit 1; fi
mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$PERMANENT_UNPACK_DIR"

echo -e "\n📦 Preparing data on the host..."

sorted_datasets=$(printf "%s\n" "${DATASET_ARGS[@]}" | sort | tr '\n' ' ' | sed 's/ *$//')
data_hash=$(echo -n "$sorted_datasets" | md5sum | cut -d' ' -f1)
STAGING_DATA_DIR="${PERMANENT_UNPACK_DIR}/${data_hash}"

if [ -d "$STAGING_DATA_DIR" ] && [ -n "$(ls -A "$STAGING_DATA_DIR")" ]; then
    echo "✅ Found cached, unpacked data. Reusing..."
    echo "   -> Location: $STAGING_DATA_DIR"
else
    echo "ℹ️  No cached data found for this dataset combination. Unpacking now..."
    echo "   -> Creating new staging directory: $STAGING_DATA_DIR"
    mkdir -p "$STAGING_DATA_DIR"
    for dataset_name in "${DATASET_ARGS[@]}"; do
        source_tgz="${DATA_DIR}/${dataset_name}.tgz"
        if [ -f "$source_tgz" ]; then
            echo "   -> Unpacking '${dataset_name}.tgz'..."
            tar --warning=no-unknown-keyword -xf "$source_tgz" -C "$STAGING_DATA_DIR"
        else
            echo "⚠️ Warning: Source archive not found at '$source_tgz'"
        fi
    done
    echo "✅ Data preparation complete."
fi

echo -e "\n🛠️  Building Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" .
echo "✅ Image built successfully."

echo -e "\n🚀 Starting the training container on GPU ${GPU_DEVICE_ID}..."
VOLUME_ARGS=(
  "--rm" "--gpus" "\"device=${GPU_DEVICE_ID}\"" "--cpus=16" "--memory=32g" "--shm-size=4g"
  "-v" "$STAGING_DATA_DIR:/app/data:ro"
  "-v" "$RESULTS_DIR:/app/results"
)
if [ -n "$CONFIG_FILE_PATH" ]; then
  VOLUME_ARGS+=("-v" "$CONFIG_FILE_PATH:/app/config.yaml:ro")
fi

docker run \
  "${VOLUME_ARGS[@]}" \
  -e WANDB_API_KEY \
  "$IMAGE_NAME" "$CONFIG_ARG"

echo -e "\n✨ Docker container has finished execution."