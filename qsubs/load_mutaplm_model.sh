#!/bin/bash
#   load_mutaplm_model.sh    — submit create_mutaplm_model jobs to LSF GPU queue


#BSUB -J load_mutaplm_model
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 2:00
#BSUB -o logs/load_mutaplm_model.%J.out
#BSUB -e logs/load_mutaplm_model.%J.err

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

LOG_DIR="logs"
LOG_LEVEL="INFO"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"


# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/load_mutaplm_model_${TIMESTAMP}.log"

echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting with the following configuration:" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"
echo "  Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"

PYTHON="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env/bin/python"

$PYTHON \
    "src/load_mutaplm_model.py" \
    --log_fn "$LOG_DIR" \
    --log_level "$LOG_LEVEL" \
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