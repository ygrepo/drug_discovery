#!/bin/bash

#   run_experiment_embeddings.sh    — submit run_experiment_embeddings jobs to LSF GPU queue

#BSUB -J run_experiment_embeddings
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -R h100nvl
#BSUB -n 4
#BSUB -R "rusage[mem=512G]"
#BSUB -W 100:00
#BSUB -o logs/run_experiment_embeddings.%J.out
#BSUB -e logs/run_experiment_embeddings.%J.err

set -euo pipefail

# ------------------------
# Parse command line args
# ------------------------
DATA_FN=""
OUTPUT_FN=""
LOG_FN="logs/extract_embeddings_default.log"
LOG_LEVEL="DEBUG"
N_SAMPLES=0
NROWS=10
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn)
      DATA_FN="$2"
      shift 2
      ;;
    --output_fn)
      OUTPUT_FN="$2"
      shift 2
      ;;
    --log_fn)
      LOG_FN="$2"
      shift 2
      ;;
    --log_level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --n_samples)
      N_SAMPLES="$2"
      shift 2
      ;;
    --nrows)
      NROWS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$DATA_FN" ]]; then
    echo "ERROR: --data_fn is required"
    exit 1
fi

if [[ -z "$OUTPUT_FN" ]]; then
    echo "ERROR: --output_fn is required"
    exit 1
fi

LOG_DIR="$(dirname "$LOG_FN")"
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_FN")"

# ------------------------
# Environment setup
# ------------------------
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

export CUDA_LAUNCH_BLOCKING=1

PYTHON="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env/bin/python"
MAIN="src/extract_all_model_embeddings.py"

# ------------------------
# Run Python script
# ------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting embedding extraction at $(date)"
echo "  Data fn:      ${DATA_FN}"
echo "  Output fn:    ${OUTPUT_FN}"
echo "  Log fn:       ${LOG_FN}"
echo "  Log level:    ${LOG_LEVEL}"
echo "  N_SAMPLES:    ${N_SAMPLES}"
echo "  NROWS:        ${NROWS}"
echo "  SEED:         ${SEED}"

set +e
"${PYTHON}" "${MAIN}" \
    --data_fn "$DATA_FN" \
    --output_fn "$OUTPUT_FN" \
    --log_fn "$LOG_FN" \
    --log_level "$LOG_LEVEL" \
    --seed "$SEED" \
    --n_samples "$N_SAMPLES" \
    --nrows "$NROWS"
exit_code=$?
set -e

if [[ $exit_code -eq 0 ]]; then
    echo "Job completed successfully at $(date)"
    exit 0
else
    echo "ERROR: Python script failed with exit code ${exit_code}"
    echo "Check log file: ${LOG_FN}"
    exit "$exit_code"
fi
