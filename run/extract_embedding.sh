#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get the directory of this script
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go one level up to get the project root
PROJECT_ROOT="$(dirname "$THIS_DIR")"

# Now define the src directory
SCRIPT_DIR="$PROJECT_ROOT/src"

echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"


# module load anaconda3/latest
# source /hpc/packages/minerva-centos7/anaconda3/2023.09/etc/profile.d/conda.sh
# conda activate drug_discovery_env

# Default configuration
DATASET_DIR="mutadescribe_data"
DATA_FN="${DATASET_DIR}/structural_split/train.csv"
OUTPUT_DIR="output/data"
OUTPUT_FN="${OUTPUT_DIR}/structural_split_train_with_embeddings.csv"
MODEL_NAME="facebook/esm2_t6_8M_UR50D"
N=2000
LOG_DIR="logs"
LOG_LEVEL="INFO"
SEED=42


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --output_fn) OUTPUT_FN="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting training with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data fn: ${DATA_FN}" | tee -a "$LOG_FILE"
echo "  Random seed: ${SEED}" | tee -a "$LOG_FILE"
echo "  Model name: ${MODEL_NAME}" | tee -a "$LOG_FILE"
echo "  Output fn: ${OUTPUT_FN}" | tee -a "$LOG_FILE"
echo "  N: ${N}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"
echo "  Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"

python "$SCRIPT_DIR/extract_embeddings.py" \
    --data_fn "$DATA_FN" \
    --output_fn "$OUTPUT_FN" \
    --model_name "$MODEL_NAME" \
    --n "$N" \
    --log_dir "$LOG_DIR" \
    --log_level "$LOG_LEVEL" \
    --seed "$SEED"