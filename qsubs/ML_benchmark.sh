#!/bin/bash
#   ML_benchmark.sh    — submit ML_benchmark jobs to LSF GPU queue

#BSUB -J ML_bench             # Job name
#BSUB -P acc_DiseaseGeneCell   # allocation account
#BSUB -q premium               # queue
#BSUB -n 8                    # number of compute cores
#BSUB -W 100:00                 # walltime in HH:MM
#BSUB -R rusage[mem=48G]     # 16 GB of memory (8 GB per core)
#BSUB -R span[hosts=1]         # all cores from the same node
#BSUB -o logs/ML_bench.%J.out
#BSUB -e logs/ML_bench.%J.err

set -euo pipefail

# --- Modules / shell setup ---
module purge
module load anaconda3/latest
module load cuda/12.4.0     # keep if your nodes provide CUDA 12.4 runtime

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Paths  ---
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/dti"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"  # conda cache (optional but recommended)

# --- Caches & hygiene ---
#mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# --- Activate env (must exist already) ---
conda activate "${ENV_PREFIX}"

# --- Use the env's Python and pip ---
PYTHON="${ENV_PREFIX}/bin/python"


ml proxies/1 || true

export RAYON_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


LOG_DIR="logs"
LOG_LEVEL="INFO"
mkdir -p "$LOG_DIR"

BASE_DATA_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/BindDBdata"
MAIN="src/ML_Benchmark.py"

MODEL_DIR="output/models"
mkdir -p "$MODEL_DIR"
OUTPUT_DIR="output/metrics"
mkdir -p "$OUTPUT_DIR"

# DATASETS=( "BindDB")     
# DATASETS=( "BindDB" "Davis" "Kiba" )     
# SPLITMODES=( "random" )  
# EMBEDDINGS=( "ESMv1" )
DATASETS=( "All_BindingDB" )     
SPLITMODES=( "random" )  
EMBEDDINGS=( "ESMv1" )
# SPLITMODES=( "random" "cold_protein" "cold_drug" )  
# EMBEDDINGS=( "ESMv1" "ESM2" "MUTAPLM" "ProteinCLIP" )

echo "Starting batch at $(date)"
echo "Base data dir: $BASE_DATA_DIR"

for dataset in "${DATASETS[@]}"; do
  for splitmode in "${SPLITMODES[@]}"; do
    for embedding in "${EMBEDDINGS[@]}"; do
      # Per-combo variables
      combo="${embedding}_${dataset}_${splitmode}"
      combo_data_dir="${BASE_DATA_DIR}/${combo}/"

      # Per-combo log dir & file
      ts=$(date +"%Y%m%d_%H%M%S")
      log_file=${LOG_DIR}/"${ts}_ML_benchmark_${combo}.log"

      echo "=== Running ${combo} ==="
      echo "  data_dir : ${combo_data_dir}"
      echo "  log_file : ${log_file}"

      # Run once per combo
      set +e
      "${PYTHON}" "${MAIN}" \
        --log_fn "${log_file}" \
        --log_level "${LOG_LEVEL}" \
        --data_dir "${BASE_DATA_DIR}" \
        --dataset "${dataset}" \
        --splitmode "${splitmode}" \
        --embedding "${embedding}" \
        --model_dir "${MODEL_DIR}" \
        --output_dir "${OUTPUT_DIR}"
    exit_code=$?
    set -e

    if [[ ${exit_code} -eq 0 ]]; then
      echo "OK: ${combo} finished at $(date)" | tee -a "${log_file}"
    else
      echo "ERROR: ${combo} failed with exit code ${exit_code} at $(date)" | tee -a "${log_file}"
      # Uncomment to stop on first failure:
      # exit ${exit_code}
    fi

    echo
    done
  done
done

echo "Batch complete at $(date)"