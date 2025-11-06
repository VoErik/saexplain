#!/bin/bash

CONFIG_DIR="assets/configs"
ARCHITECTURES=("relu" "topk" "batchtopk" "jumprelu" "matryoshka")
BASE_CONFIG="${CONFIG_DIR}/sae.yaml"

echo "Starting all sweeps..."

for arch in "${ARCHITECTURES[@]}"; do
  SWEEP_FILE="${CONFIG_DIR}/sweep_${arch}.yaml"
  
  echo "-----------------------------------"
  echo "     Starting sweep for: $arch"
  echo "     Using sweep file: $SWEEP_FILE"
  echo "-----------------------------------"
  
  # Run the command
  uv run main.py \
    --mode="train" \
    --config="$BASE_CONFIG" \
    --sweep="$SWEEP_FILE" \
    --do_sweep
    
  echo "Sweep for $arch started."
done

echo "All sweeps have been started."