#!/bin/bash
#   extract_embedding.sh    — submit extract_embedding jobs to LSF GPU queue


#BSUB -J embeddings
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 0:30
#BSUB -o logs/embeddings.%J.out
#BSUB -e logs/embeddings.%J.err

set -euo pipefail

# --- Clean environment to avoid ~/.local issues ---
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONUSERBASE

module purge
module load cuda/11.8 cudnn
module load anaconda3/latest
source $(conda info --base)/etc/profile.d/conda.sh
conda activate drug_discovery_env

ml proxies/1 || true

# Get the directory of this script
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Now define the src directory
SCRIPT_DIR="$THIS_DIR/src"

echo "Project root: $THIS_DIR"
cd "$THIS_DIR"

# --- Verify environment ---
echo "Python: $(which python)"
/hpc/users/greaty01/.conda/envs/drug_discovery_env/bin/python -c \
    "import sys, torch, transformers; print('Python', sys.version); print('Torch', torch.__version__); print('Transformers', transformers.__version__)"


export HF_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.cache/huggingface"
MODEL_NAME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models/esm1v_t33_650M_UR90S_5_safe"


# Default configuration
DATASET_DIR="mutadescribe_data"
DATA_FN="${DATASET_DIR}/structural_split/train.csv"
OUTPUT_DIR="output/data"
OUTPUT_FN="${OUTPUT_DIR}/esm1v_structural_split_train_with_embeddings.csv"
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
echo "Project root: $THIS_DIR" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data fn: ${DATA_FN}" | tee -a "$LOG_FILE"
echo "  Random seed: ${SEED}" | tee -a "$LOG_FILE"
echo "  Model name: ${MODEL_NAME}" | tee -a "$LOG_FILE"
echo "  Output fn: ${OUTPUT_FN}" | tee -a "$LOG_FILE"
echo "  N: ${N}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"
echo "  Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"

#/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/mutaplm_env/bin/python \
/hpc/users/greaty01/.conda/envs/drug_discovery_env/bin/python \
    "$SCRIPT_DIR/extract_embeddings.py" \
    --data_fn "$DATA_FN" \
    --output_fn "$OUTPUT_FN" \
    --model_name "$MODEL_NAME" \
    --n "$N" \
    --log_dir "$LOG_DIR" \
    --log_level "$LOG_LEVEL" \
    --seed "$SEED" \
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