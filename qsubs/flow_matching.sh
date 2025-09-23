#!/bin/bash
# flow_matching_worker.sh — run one (embedding, dataset, splitmode) combo on LSF

# ------- LSF resources (kept here so every job uses the same resources) -------
# You can still override these from the submitter with bsub CLI flags.
#BSUB -J flow_matching             # Job name
#BSUB -P acc_DiseaseGeneCell   # allocation account
#BSUB -q gpu                  # queue
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1                   # number of compute cores
#BSUB -W 100:00                 # walltime in HH:MM
#BSUB -R rusage[mem=32G]       #16 GB of memory (8 GB per core)
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <DATASET> <SPLITMODE> <EMBEDDING>"
  exit 2
fi

DATASET="$1"       # e.g., BindingDB | BindDB | Davis | Kiba
SPLITMODE="$2"     # e.g., random | cold_protein | cold_drug
EMBEDDING="$3"     # e.g., ESMv1 | ESM2 | MUTAPLM | ProteinCLIP

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

export RAYON_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Project paths ---
LOG_DIR="logs"
LOG_LEVEL="INFO"
mkdir -p "$LOG_DIR"

BASE_DATA_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/BindDBdata/Mutant_BindingDB"
MAIN="src/flow_matching_run.py"

OUTPUT_DIR="output/data";   mkdir -p "$OUTPUT_DIR"
CHECKPOINTS_DIR="./checkpoints/flow_matching_${DATASET}_${SPLITMODE}_${EMBEDDING}"; 
mkdir -p "$CHECKPOINTS_DIR"
MODEL_LOG_DIR="./output/logs/flow_matching_${DATASET}_${SPLITMODE}_${EMBEDDING}";          
mkdir -p "$MODEL_LOG_DIR"

# --- Training knobs (same as your original) ---
BATCH_SIZE=32
NUM_WORKERS=10
PIN_MEMORY=true
SHUFFLE=true
CHECK_NAN=true
DEVICE="auto"
MAX_EPOCHS=100

combo="${EMBEDDING}_${DATASET}_${SPLITMODE}"
ts=$(date +"%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/${ts}_flow_matching_${combo}.log"

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
  --output_dir "${OUTPUT_DIR}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --pin_memory "${PIN_MEMORY}" \
  --shuffle "${SHUFFLE}" \
  --check_nan "${CHECK_NAN}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --device "${DEVICE}"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "OK: ${combo} finished at $(date)" | tee -a "${log_file}"
else
  echo "ERROR: ${combo} failed with exit code ${exit_code} at $(date)" | tee -a "${log_file}"
  exit ${exit_code}
fi
