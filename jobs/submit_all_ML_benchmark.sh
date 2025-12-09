#!/bin/bash
# submit_all_ML_benchmark.sh — submit one LSF job per (embedding, dataset, splitmode)
set -euo pipefail

DATASETS=( "BindingDB" )
SPLITMODES=( "random" "cold_protein" "cold_drug" )
EMBEDDINGS=( "ESMv1" "ESM2" "MUTAPLM" "ProteinCLIP" )

ACCOUNT="acc_DiseaseGeneCell"
QUEUE="premium"
NCORES=8
WALL="100:00"
MEM="48G"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

BASE_DATA_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/BindDBdata/All_BindingDB"
echo "Starting batch at $(date)"
echo "Base data dir: ${BASE_DATA_DIR}"

for dataset in "${DATASETS[@]}"; do
  for splitmode in "${SPLITMODES[@]}"; do
    for embedding in "${EMBEDDINGS[@]}"; do
      combo="${embedding}_${dataset}_${splitmode}"
      bsub \
        -J "ML_bench_${combo}" \
        -P "${ACCOUNT}" \
        -q "${QUEUE}" \
        -n "${NCORES}" \
        -R "rusage[mem=${MEM}] span[hosts=1]" \
        -W "${WALL}" \
        -o "logs/ML_benchmark.${combo}.%J.out" \
        -e "logs/ML_benchmark.${combo}.%J.err" \
        ./qsubs/ML_benchmark.sh "${dataset}" "${splitmode}" "${embedding}"
    done
  done
done

echo "Batch complete at $(date)"
  