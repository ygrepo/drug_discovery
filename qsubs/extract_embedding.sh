#!/bin/bash
#   extract_embedding.sh    — submit extract_embedding jobs to LSF GPU queue


#BSUB -J embeddings
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 2:00
#BSUB -o logs/embeddings.%J.out
#BSUB -e logs/embeddings.%J.err

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

# Must export for Python to see it
#MODEL_TYPE="ESM2"
#MODEL_TYPE="ESMv1"
MODEL_TYPE="MUTAPLM"

# Default configuration
DATA_FN="mutadescribe_data/structural_split/train.csv"
OUTPUT_DIR="output/data"
#OUTPUT_FN="${OUTPUT_DIR}/esm1v_structural_split_train_with_embeddings.csv"
#OUTPUT_FN="${OUTPUT_DIR}/esmv1_structural_split_train_with_embeddings.csv"
#OUTPUT_FN="${OUTPUT_DIR}/esm2_t33_650M_UR50D_structural_split_train_with_embeddings.csv"
OUTPUT_FN="${OUTPUT_DIR}/mutaplm_structural_split_train_with_embeddings.csv"
N=2000
LOG_DIR="logs"
LOG_LEVEL="INFO"
SEED=42


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --output_fn) OUTPUT_FN="$2"; shift 2 ;;
    --model_type) MODEL_TYPE="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"


# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/extract_embeddings_${TIMESTAMP}.log"

echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data fn: ${DATA_FN}" | tee -a "$LOG_FILE"
echo "  Random seed: ${SEED}" | tee -a "$LOG_FILE"
echo "  Model type: ${MODEL_TYPE}" | tee -a "$LOG_FILE"
echo "  Output fn: ${OUTPUT_FN}" | tee -a "$LOG_FILE"
echo "  N: ${N}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"
echo "  Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"

export CUDA_LAUNCH_BLOCKING=1

PYTHON="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env/bin/python"

$PYTHON \
    "src/extract_embeddings.py" \
    --data_fn "$DATA_FN" \
    --output_fn "$OUTPUT_FN" \
    --model_type "$MODEL_TYPE" \
    --log_fn "$LOG_FILE" \
    --log_level "$LOG_LEVEL" \
    --seed "$SEED" \
    --n "$N" \
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