#!/bin/bash
#   gnn.sh    — submit gnn jobs to LSF GPU queue

#BSUB -J gnn             # Job name
#BSUB -P acc_DiseaseGeneCell   # allocation account
#BSUB -q gpu                  # queue
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1                   # number of compute cores
#BSUB -W 100:00                 # walltime in HH:MM
#BSUB -R rusage[mem=512G]       #16 GB of memory (8 GB per core)
#BSUB -o logs/gnn.%J.out
#BSUB -e logs/gnn.%J.err


set -euo pipefail

# --- Clean environment to avoid ~/.local issues ---
module purge
module load cuda/12.4.0
module load anaconda3/latest
source $(conda info --base)/etc/profile.d/conda.sh

export PROJ=/sc/arion/projects/DiseaseGeneCell/Huang_lab_data
export CONDARC="$PROJ/conda/condarc"
conda activate /sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env

ml proxies/1 || true

export RAYON_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONNOUSERSITE=1
unset PYTHONPATH


LOG_DIR="logs"
LOG_LEVEL="INFO"
mkdir -p "$LOG_DIR"

BASE_DATA_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/BindDBdata/Embedding_Benchmark_Data"
PYTHON="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env/bin/python"
MAIN="src/gnn_run.py"

MODEL_DIR="output/models/gnn"
mkdir -p "$MODEL_DIR"
OUTPUT_DIR="output/data"
mkdir -p "$OUTPUT_DIR"

DATASETS=( "BindDB" "Davis" "Kiba" )     
SPLITMODES=( "random" "cold_protein" "cold_drug" )  
EMBEDDINGS=( "ESMv1" "ESM2" "MUTAPLM" "ProteinCLIP" )

echo "Starting batch at $(date)"
echo "Base data dir: $BASE_DATA_DIR"

BATCH_SIZE=32
NUM_WORKERS=10
PIN_MEMORY=true
SHUFFLE=true
CHECK_NAN=true
#SCALE="zscore"
DEVICE="auto"
MAX_EPOCHS=100
CHECKPOINTS_DIR="output/checkpoints/gnn"
mkdir -p "$CHECKPOINTS_DIR"

for dataset in "${DATASETS[@]}"; do
  for splitmode in "${SPLITMODES[@]}"; do
    for embedding in "${EMBEDDINGS[@]}"; do  
    # Per-combo variables
    combo="${embedding}_${dataset}_${splitmode}"
    combo_data_dir="${BASE_DATA_DIR}/${combo}/"

    # Per-combo log dir & file
    ts=$(date +"%Y%m%d_%H%M%S")
    log_file=${LOG_DIR}/"${ts}_gnn_${combo}.log"

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
      # Uncomment to stop on first failure:
      # exit ${exit_code}
    fi

    echo
    done
  done
done

echo "Batch complete at $(date)"