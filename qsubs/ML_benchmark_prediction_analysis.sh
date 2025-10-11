#!/bin/bash
# ML_benchmark_prediction_analysis.sh — 

# ------- LSF resources (kept here so every job uses the same resources) -------
# You can still override these from the submitter with bsub CLI flags.
#BSUB -J ML_benchmark_prediction_analysis             # Job name
#BSUB -P acc_DiseaseGeneCell   # allocation account
#BSUB -q premium                  # queue
#BSUB -n 1                  # number of compute cores
#BSUB -W 100:00                 # walltime in HH:MM
#BSUB -R rusage[mem=40G]      
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Modules / shell setup ---
module purge
module load anaconda3/latest
module load cuda/12.4.0

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Paths / env ---
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/dti"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"

mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR CONDA_PKGS_DIRS
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

conda activate "${ENV_PREFIX}"
PYTHON="${ENV_PREFIX}/bin/python"

ml proxies/1 || true

export PYTHONUNBUFFERED=1
export TERM=xterm

# --- Project paths ---
LOG_DIR="logs"
LOG_LEVEL="INFO"
mkdir -p "$LOG_DIR"

DATA_FN="output/data/combined_predictions_BindingDB.parquet"
OUTPUT_DIR="output/metrics";   mkdir -p "$OUTPUT_DIR"
PREFIX="All_BindingDB_prediction_analysis"
MAIN="src/ML_Benchmark_Prediction_Analysis.py"

ts=$(date +"%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/${ts}_ML_benchmark_prediction_analysis.log"

echo "JOBID=${LSB_JOBID:-local}  IDX=${LSB_JOBINDEX:-}  HOST=$(hostname)"
echo "=== Running ML_benchmark_prediction_analysis ==="
echo "  log_file : ${log_file}"

set +e
"${PYTHON}" "${MAIN}" \
  --log_fn "${log_file}" \
  --log_level "${LOG_LEVEL}" \
  --data_fn "${DATA_FN}" \
  --prefix "${PREFIX}" \
  --output_dir "${OUTPUT_DIR}"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "OK: ${combo} finished at $(date)" | tee -a "${log_file}"
else
  echo "ERROR: ${combo} failed with exit code ${exit_code} at $(date)" | tee -a "${log_file}"
  exit ${exit_code}
fi
