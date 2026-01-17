#!/bin/bash
# OMIESI_classifier.sh — submit OMIESI_classifier jobs to LSF GPU queue

#BSUB -J OMIESI_classifier
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=32G]"
#BSUB -W 2:00
#BSUB -o logs/OMIESI_classifier.%J.out
#BSUB -e logs/OMIESI_classifier.%J.err

set -euo pipefail

# --- Clean environment to avoid ~/.local issues ---
module purge
module load cuda/11.8 cudnn
module load anaconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"

export PROJ=/sc/arion/projects/DiseaseGeneCell/Huang_lab_data
export CONDARC="$PROJ/conda/condarc"
conda activate /sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env

ml proxies/1 || true

export HF_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.cache/huggingface"
mkdir -p "$HF_HOME"
export TORCH_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.torch_hub"
mkdir -p "$TORCH_HOME"

# Default configuration
BASE_DATA_DIR="output/data/OMIESI"
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

mkdir -p "$LOG_DIR"

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/OMIESI_classifier_${TIMESTAMP}.log"

echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

echo "Starting with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data dir: ${BASE_DATA_DIR}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"
echo "  Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"

export CUDA_LAUNCH_BLOCKING=1

PYTHON="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env/bin/python"
MAIN="src/ML_OMIESI_benchmark.py"

set +e
"${PYTHON}" "${MAIN}" \
  --log_fn "${LOG_FILE}" \
  --log_level "${LOG_LEVEL}" \
  --data_dir "${BASE_DATA_DIR}" \
  --use_fingerprints \
  --scale_for_linear \
  --mutation_encoding both \
  --use_pos_norm
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "OK: Script finished at $(date)" | tee -a "${LOG_FILE}"
else
  echo "ERROR: Script failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
  exit ${exit_code}
fi
