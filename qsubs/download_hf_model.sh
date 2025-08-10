#!/bin/bash


#BSUB -J download_hf_model
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 0:30
#BSUB -o logs/download_hf_model.%J.out
#BSUB -e logs/download_hf_model.%J.err

set -euo pipefail

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

# Must export for Python to see it
#export MODEL_NAME="facebook/esm1v_t33_650M_UR90S_5"
export MODEL_NAME="facebook/facebook/esm2_t33_650M_UR50D"

# ---------------- Configuration ----------------
export MODELS_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models"
export MODEL_DIR=${MODELS_DIR}/${MODEL_NAME}
export SAFE_DIR=${MODELS_DIR}/${MODEL_NAME}_safe

LOG_DIR="logs"
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models_dir) MODELS_DIR="$2"; shift 2 ;;
    --hf_home) HF_HOME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --model_dir) MODEL_DIR="$2"; shift 2 ;;
    --safe_dir) SAFE_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    --log_fn) LOG_FN="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/download_hf_model_${TIMESTAMP}.log"

echo "Starting download_hf_model at $(date)" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting with the following configuration:" | tee -a "$LOG_FILE"
echo "  Model name: ${MODEL_NAME}" | tee -a "$LOG_FILE"
echo "  Model dir: ${MODEL_DIR}" | tee -a "$LOG_FILE"
echo "  Safe dir: ${SAFE_DIR}" | tee -a "$LOG_FILE"

echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"
echo "  Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"

python \
    "src/download_hf_model.py" \
    --model_name "$MODEL_NAME" \
    --model_dir "$MODEL_DIR" \
    --safe_dir "$SAFE_DIR" \
    --log_level "$LOG_LEVEL" \
    --log_fn "$LOG_FILE" \
    2>&1 | tee -a "$LOG_FILE"

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