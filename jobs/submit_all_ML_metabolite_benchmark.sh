#!/bin/bash
# submit_all_ML_metabolite_benchmark.sh — submit one LSF job per (embedding, dataset, splitmode)
set -euo pipefail

DATASETS=( "EITLEM_kkm" "EITLEM_kcat" "EITLEM_km" "MPEK_kcat" "MPEK_km" "catpred_kcat" "catpred_ki" "catpred_km" )
SPLITMODES=( "random" )
EMBEDDINGS=( "ESMv1" )
# SPLITMODES=( "random" "cold_protein" "cold_drug" )
# EMBEDDINGS=( "ESMv1" "ESM2" "MUTAPLM" "ProteinCLIP" )

ACCOUNT="acc_DiseaseGeneCell"
QUEUE="premium"
NCORES=8
WALL="100:00"
MEM="48G"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

echo "Starting batch at $(date)"

for dataset in "${DATASETS[@]}"; do
  for splitmode in "${SPLITMODES[@]}"; do
    for embedding in "${EMBEDDINGS[@]}"; do
      combo="${dataset}_${embedding}_embedding_${splitmode}"
      bsub \
        -J "ML_metabolite_bench_${combo}" \
        -P "${ACCOUNT}" \
        -q "${QUEUE}" \
        -n "${NCORES}" \
        -R "rusage[mem=${MEM}] span[hosts=1]" \
        -W "${WALL}" \
        -o "logs/ML_metabolite_benchmark.${combo}.%J.out" \
        -e "logs/ML_metabolite_benchmark.${combo}.%J.err" \
        ./jobs/ML_metabolite_benchmark.sh "${dataset}" "${splitmode}" "${embedding}"
    done
  done
done

echo "Batch complete at $(date)"
  