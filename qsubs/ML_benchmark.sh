#!/bin/bash
#   ML_benchmark.sh    — submit ML_benchmark jobs to LSF GPU queue

#BSUB -J ML_bench             # Job name
#BSUB -P acc_DiseaseGeneCell   # allocation account
#BSUB -q premium               # queue
#BSUB -n 8                    # number of compute cores
#BSUB -W 72:00                 # walltime in HH:MM
#BSUB -R rusage[mem=8000]     # 8 GB of memory (8 GB per core)
#BSUB -R span[hosts=1]         # all cores from the same node
#BSUB -o logs/ML_bench.%J.out
#BSUB -e logs/ML_bench.%J.err

set -euo pipefail

# --- Clean environment to avoid ~/.local issues ---
module purge
module load cuda/12.4.0
#module load cuda/11.8 cudnn
module load anaconda3/latest
source $(conda info --base)/etc/profile.d/conda.sh

export PROJ=/sc/arion/projects/DiseaseGeneCell/Huang_lab_data
export CONDARC="$PROJ/conda/condarc"
conda activate /sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env

ml proxies/1 || true

export RAYON_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="logs"
LOG_LEVEL="INFO"
mkdir -p "$LOG_DIR"

BASE_DATA_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/BindDBdata"
PYTHON="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env/bin/python"
MAIN="src/ML_Benchmark.py"

MODEL_DIR="output/models"
mkdir -p "$MODEL_DIR"
OUTPUT_DIR="output/data"
mkdir -p "$OUTPUT_DIR"

DATASETS=( "BindDB" "Davis" )     
SPLITMODES=( "random" "random" )  

echo "Starting batch at $(date)"
echo "Base data dir: $BASE_DATA_DIR"

for dataset in "${DATASETS[@]}"; do
  for splitmode in "${SPLITMODES[@]}"; do
    # Per-combo variables
    combo="${dataset}_${splitmode}"
    combo_data_dir="${BASE_DATA_DIR}/${combo}/"

    # Per-combo log dir & file
    ts=$(date +"%Y%m%d_%H%M%S")
    log_file=${LOG_DIR}/"${ts}_${combo}.log"

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
      --model_dir "${MODEL_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      2>&1 | tee -a "${log_file}"
    exit_code=${PIPESTATUS[0]}
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

echo "Batch complete at $(date)"