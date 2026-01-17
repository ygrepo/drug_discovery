#!/bin/bash
#   extract_OMIESI_embeddings.sh    — submit extract_OMIESI_embeddings jobs to LSF GPU queue


#BSUB -J OMIESI_embeddings
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=32G]"
#BSUB -W 2:00
#BSUB -o logs/OMIESI_embeddings.%J.out
#BSUB -e logs/OMIESI_embeddings.%J.err

set -euo pipefail

# --- Clean environment to avoid ~/.local issues ---
module purge
module load cuda/11.8 cudnn
module load anaconda3/latest
source $(conda info --base)/etc/profile.d/conda.sh

export PROJ=/sc/arion/projects/DiseaseGeneCell/Huang_lab_data
export CONDARC="$PROJ/conda/condarc"
conda activate /sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env

ml proxies/1 || true

export HF_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.cache/huggingface"
mkdir -p "$HF_HOME"
export TORCH_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.torch_hub"
mkdir -p "$TORCH_HOME"

# Default configuration
DATA_FN="data/OMIESI/mut_classify/train.csv"
OUTPUT_DIR="output/data"
OUTPUT_FN="${OUTPUT_DIR}/OMIESI_train_embeddings.csv"

NROWS=200
LOG_DIR="logs"
LOG_LEVEL="INFO"
GENERATE_FINGERPRINTS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --output_fn) OUTPUT_FN="$2"; shift 2 ;;
    --n) NROWS="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    --generate_fingerprints) GENERATE_FINGERPRINTS="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"


# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/extract_OMIESI_embeddings_${TIMESTAMP}.log"

echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data fn: ${DATA_FN}" | tee -a "$LOG_FILE"
echo "  Output fn: ${OUTPUT_FN}" | tee -a "$LOG_FILE"
echo "  N: ${NROWS}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"
echo "  Generate fingerprints: ${GENERATE_FINGERPRINTS}" | tee -a "$LOG_FILE"
echo "  Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"

export CUDA_LAUNCH_BLOCKING=1

PYTHON="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env/bin/python"

# Build command with conditional fingerprint flag
PYTHON_CMD="$PYTHON src/extract_OMIESI_embeddings.py --data_fn \"$DATA_FN\" --output_fn \"$OUTPUT_FN\" --log_fn \"$LOG_FILE\" --log_level \"$LOG_LEVEL\" --n \"$NROWS\""

if [ "$GENERATE_FINGERPRINTS" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --generate_fingerprints"
fi

eval "$PYTHON_CMD"

# Check the exit status of the Python script
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi