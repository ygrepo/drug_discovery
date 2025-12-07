#!/bin/bash
#   load_model.sh    — submit load_model jobs to LSF GPU queue


#export HF_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.cache/huggingface"
# export HF_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models"
# mkdir -p "$HF_HOME"

# export TORCH_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.torch_hub"
# mkdir -p "$TORCH_HOME"

# export HF_TOKEN_PATH="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/tokens/hf_token.csv"

LOG_DIR="logs"
LOG_LEVEL="INFO"
#MODEL_TYPE="ESM2"
MODEL_TYPE="ESMv1"
#MODEL_TYPE="MUTAPLM"
#MODEL_TYPE="ProteinCLIP"
#MODEL_TYPE="LLAMA"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    --model_type) MODEL_TYPE="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"


# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/load_model_${TIMESTAMP}.log"

echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting with the following configuration:" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"
echo "  Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "  Model type: ${MODEL_TYPE}" | tee -a "$LOG_FILE"

set +e
python \
    "src/load_model.py" \
    --log_fn "$LOG_FILE" \
    --log_level "$LOG_LEVEL" \
    --model_type "$MODEL_TYPE"

exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "OK: finished at $(date)" | tee -a "$LOG_FILE"
else
  echo "ERROR: failed with exit code ${exit_code} at $(date)" | tee -a "$LOG_FILE"
  exit ${exit_code}
fi