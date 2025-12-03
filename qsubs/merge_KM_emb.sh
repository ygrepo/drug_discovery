#!/bin/bash
#   merge_KM_embeddings.sh    — submit merge_KM_embeddings jobs to LSF GPU queue


#BSUB -J merge_KM_embeddings
#BSUB -P acc_DiseaseGeneCell
#BSUB -q premium
#BSUB -n 4
#BSUB -R "rusage[mem=64G]"
#BSUB -W 100:00
#BSUB -o logs/merge_KM_embeddings.%J.out
#BSUB -e logs/merge_KM_embeddings.%J.err

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
DATA_FN="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/EnzymaticReactionPrediction/Regression_Data/exp_of_catpred_MPEK_EITLEM_inhouse_dataset/experiments/dataset_MPEK_km/A01_dataset/data_km_with_features.joblib"
EMBEDDING_FN="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/drug_discovery/output/data/20251203_data_km_embeddings.pt"
OUTPUT_DIR="output/data"

OUTPUT_FN="data_km_with_features_and_embeddings"
N_SAMPLES=0
NROWS=0
LOG_DIR="logs"
LOG_LEVEL="DEBUG"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --embedding_fn) EMBEDDING_FN="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --output_fn) OUTPUT_FN="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"


# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/merge_KM_embeddings_${TIMESTAMP}.log"

echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data fn: ${DATA_FN}" | tee -a "$LOG_FILE"
echo "  Embedding fn: ${EMBEDDING_FN}" | tee -a "$LOG_FILE"
echo "  Output fn: ${OUTPUT_FN}" | tee -a "$LOG_FILE"
echo "  N_SAMPLES: ${N_SAMPLES}" | tee -a "$LOG_FILE"
echo "  NROWS: ${NROWS}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"
echo "  Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"

export CUDA_LAUNCH_BLOCKING=1

PYTHON="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env/bin/python"
MAIN="src/merge_KM_embeddings.py"

set +e
"${PYTHON}" "${MAIN}" \
    --data_fn "$DATA_FN" \
    --embedding_fn "$EMBEDDING_FN" \
    --output_dir "$OUTPUT_DIR" \
    --output_fn "$OUTPUT_FN" \
    --log_fn "$LOG_FILE" \
    --log_level "$LOG_LEVEL" \
    --n_samples "$N_SAMPLES" \
    --nrows "$NROWS"
exit_code=$?
set -e


# Check the exit status of the Python script
EXIT_CODE=$exit_code

if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi