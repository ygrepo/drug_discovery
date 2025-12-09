#!/bin/bash
# qsubs/ML_benchmark.sh — runs one combo inside an LSF job
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <DATASET> <SPLITMODE> <EMBEDDING>"
  exit 2
fi

DATASET="$1"       # e.g., BindingDB | Davis | Kiba
SPLITMODE="$2"     # random | cold_protein | cold_drug
EMBEDDING="$3"     # ESMv1 | ESM2 | MUTAPLM | ProteinCLIP

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
MODEL_DIR="output/models"; mkdir -p "${MODEL_DIR}"
OUTPUT_DIR="output/metrics"; mkdir -p "${OUTPUT_DIR}"
LOG_LEVEL="INFO"

BASE_DATA_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/BindDBdata/All_BindingDB"
MAIN="src/ML_Benchmark.py"
N=0

combo="${EMBEDDING}_${DATASET}_${SPLITMODE}"
ts=$(date +"%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/${ts}_ML_benchmark_${combo}.log"

echo "JOBID=${LSB_JOBID:-local}  IDX=${LSB_JOBINDEX:-}  HOST=$(hostname)"
echo "=== Running ${combo} ==="
echo "  data_dir : ${BASE_DATA_DIR}"
echo "  log_file : ${log_file}"

set +e
"${PYTHON}" "${MAIN}" \
  --log_fn "${log_file}" \
  --log_level "${LOG_LEVEL}" \
  --data_dir "${BASE_DATA_DIR}" \
  --dataset "${DATASET}" \
  --splitmode "${SPLITMODE}" \
  --embedding "${EMBEDDING}" \
  --N "${N}" \
  --model_dir "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "OK: ${combo} finished at $(date)" | tee -a "${log_file}"
else
  echo "ERROR: ${combo} failed with exit code ${exit_code} at $(date)" | tee -a "${log_file}"
  exit ${exit_code}
fi
