#!/bin/bash
#   combine_ML_benchmark_predictions.sh    — submit combine_ML_benchmark_predictions jobs to LSF GPU queue


#BSUB -J combine_ML_benchmark_predictions
#BSUB -P acc_DiseaseGeneCell
#BSUB -q premium
#BSUB -n 8
#BSUB -R "rusage[mem=512G]"
#BSUB -W 6:00
#BSUB -o logs/combine_ML_benchmark_predictions.%J.out
#BSUB -e logs/combine_ML_benchmark_predictions.%J.err

set -euo pipefail

# --- Modules / env ---
module purge
module load anaconda3/latest
module load cuda/12.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/dti"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"
export PIP_CACHE_DIR CONDA_PKGS_DIRS PYTHONNOUSERSITE=1
unset PYTHONPATH || true

conda activate "${ENV_PREFIX}"
PYTHON="${ENV_PREFIX}/bin/python"

ml proxies/1 || true
export PYTHONUNBUFFERED=1 TERM=xterm
export RAYON_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Project paths ---
LOG_DIR="logs"; mkdir -p "${LOG_DIR}"
PREDICTION_DIR="output/metrics"; mkdir -p "${PREDICTION_DIR}"
OUTPUT_DIR="output/data"; mkdir -p "${OUTPUT_DIR}"
LOG_LEVEL="INFO"

BASE_DATA_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/BindDBdata/All_BindingDB"
MAIN="src/combine_ML_benchmark_predictions.py"
DATASET="BindingDB"
DATE_PATTERN="20251013"
ts=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${ts}_combine_ML_benchmark_${DATASET}.log"


echo "JOBID=${LSB_JOBID:-local}  IDX=${LSB_JOBINDEX:-}  HOST=$(hostname)"
echo "=== Running ${DATASET} ==="
echo "  data_dir : ${BASE_DATA_DIR}"
echo "  log_file : ${LOG_FILE}"

set +e
"${PYTHON}" "${MAIN}" \
  --log_fn "${LOG_FILE}" \
  --log_level "${LOG_LEVEL}" \
  --data_dir "${BASE_DATA_DIR}" \
  --dataset "${DATASET}" \
  --prediction_dir "${PREDICTION_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --date_pattern "${DATE_PATTERN}"
exit_code=$?
set -e


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