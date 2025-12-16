#!/bin/bash
# submit_all_ML_metabolite_benchmark.sh — submit one LSF job per (embedding, dataset, splitmode)
set -euo pipefail

DATASETS=( "EITLEM")
REACTION=( "kcat")
SPLITMODES=( "random" )
EMBEDDINGS=( "ESMv1" )
#DATASETS=( "EITLEM" "MPEK" "catpred" )
#REACTION=( "kcat" "kkm" "km" "kd" "ki" )
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
  for reaction in "${REACTION[@]}"; do
    for splitmode in "${SPLITMODES[@]}"; do
      for embedding in "${EMBEDDINGS[@]}"; do
        combo="${dataset}_${reaction}_${embedding}_embedding_${splitmode}"
        bsub \
          -J "ML_metabolite_bench_${combo}" \
          -P "${ACCOUNT}" \
          -q "${QUEUE}" \
          -n "${NCORES}" \
          -R "rusage[mem=${MEM}] span[hosts=1]" \
          -W "${WALL}" \
          -o "logs/ML_metabolite_benchmark.${combo}.%J.out" \
          -e "logs/ML_metabolite_benchmark.${combo}.%J.err" \
          ./jobs/ML_metabolite_benchmark.sh "${dataset}" "${reaction}" "${splitmode}" "${embedding}"
      done
    done
  done
done

echo "Batch complete at $(date)"
  