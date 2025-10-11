#!/bin/bash
# ML_benchmark_prediction_analysis.sh —

# ------- LSF resources -------
#BSUB -J ML_benchmark_prediction_analysis
#BSUB -P acc_DiseaseGeneCell
#BSUB -q premium
#BSUB -n 1
#BSUB -W 100:00
#BSUB -R rusage[mem=40G]
# --------------------------------

set -Eeuo pipefail

# ---- Helpful tracing & error trap ----
SCRIPT_NAME="$(basename "$0")"
trap 'ec=$?; echo "[ERROR] ${SCRIPT_NAME}: line ${LINENO} exited with status ${ec}" >&2' ERR

# ---- Resolve repo root so relative paths work no matter where we bsub from ----
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${THIS_DIR}"  # adjust if script lives in repo root or in scripts/

# ---- Logging: create file early and tee all stdout/stderr ----
mkdir -p logs
ts="$(date +"%Y%m%d_%H%M%S")"
log_file="logs/${ts}_ML_benchmark_prediction_analysis.log"
# mirror output to both console (LSF file) and our log_file
exec > >(tee -a "${log_file}") 2>&1

echo "------------------------------------------------------------"
echo "JOB START: $(date)"
echo "SCRIPT    : ${SCRIPT_NAME}"
echo "JOBID     : ${LSB_JOBID:-local}  IDX=${LSB_JOBINDEX:-}"
echo "HOST      : $(hostname)"
echo "LOG FILE  : ${log_file}"
echo "------------------------------------------------------------"

# ---- Modules / shell setup (don’t exit if a module isn’t present) ----
module purge || true
module load anaconda3/latest || true
module load cuda/12.4.0 || true

# Some clusters use `ml` alias; use module consistently
# module load proxies/1 || true

# ---- Conda bootstrap ----
if ! base_dir="$(conda info --base 2>/dev/null)"; then
  echo "[WARN] 'conda info --base' failed; trying common location…" >&2
  base_dir="$HOME/miniconda3"  # fallback; adjust if needed
fi

# shellcheck disable=SC1091
if [[ -f "${base_dir}/etc/profile.d/conda.sh" ]]; then
  source "${base_dir}/etc/profile.d/conda.sh"
else
  echo "[ERROR] Could not source conda.sh from ${base_dir}" >&2
  exit 1
fi

# ---- Paths / env ----
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/dti"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"

mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR CONDA_PKGS_DIRS PYTHONNOUSERSITE=1 TERM=xterm PYTHONUNBUFFERED=1
unset PYTHONPATH || true

echo "Activating conda env: ${ENV_PREFIX}"
conda activate "${ENV_PREFIX}"
PYTHON="${ENV_PREFIX}/bin/python"

# ---- Project paths (relative to script location) ----
LOG_LEVEL="INFO"
DATA_FN="output/data/combined_predictions_BindingDB.parquet"
OUTPUT_DIR="output/metrics"
PREFIX="All_BindingDB_prediction_analysis"
MAIN="src/ML_Benchmark_Prediction_Analysis.py"

mkdir -p "${OUTPUT_DIR}"

# ---- Sanity checks (don’t hard-fail; just warn) ----
[[ -x "${PYTHON}" ]] || echo "[WARN] Python not found at ${PYTHON}; using PATH python" && PYTHON="python"
[[ -f "${MAIN}"    ]] || { echo "[ERROR] MAIN not found: ${MAIN}"; exit 2; }
[[ -f "${DATA_FN}" ]] || echo "[WARN] DATA_FN not found yet: ${DATA_FN}"

echo "Python     : $(command -v "${PYTHON}")"
echo "Main script: ${MAIN}"
echo "Data file  : ${DATA_FN}"
echo "Output dir : ${OUTPUT_DIR}"
echo "Prefix     : ${PREFIX}"
echo "Log level  : ${LOG_LEVEL}"
echo "------------------------------------------------------------"

# ---- Run ----
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
  echo "[OK] ${SCRIPT_NAME} finished at $(date)"
else
  echo "[ERROR] ${SCRIPT_NAME} failed with exit code ${exit_code} at $(date)"
  exit ${exit_code}
fi

echo "JOB END: $(date)"
