#!/bin/bash
# submit_all_diffusion_regressor.sh — submit one LSF job per (embedding, dataset, splitmode)

set -euo pipefail

# Define your grids:
DATASETS=( "BindingDB" )
#DATASETS=( "BindingDB" "BindDB" "Davis" "Kiba" )
SPLITMODES=( "random" "cold_protein" "cold_drug" )
#SPLITMODES=( "random" "cold_protein" "cold_drug" )
#SPLITMODES=( "random" "cold_protein" "cold_drug" )
EMBEDDINGS=( "ESMv1" )
#EMBEDDINGS=( "ESMv1" "ESM2" "MUTAPLM" "ProteinCLIP" )

mkdir -p logs

for dataset in "${DATASETS[@]}"; do
  for splitmode in "${SPLITMODES[@]}"; do
    for embedding in "${EMBEDDINGS[@]}"; do
      combo="${embedding}_${dataset}_${splitmode}"

      # Submit one job for this combo.
      # We set name, logs, and also pass resource flags here (they override #BSUB in the script if duplicated).
      bsub \
        -J "flow_${combo}" \
        -P "acc_DiseaseGeneCell" \
        -q "gpu" \
        -gpu "num=1" \
        -R "a10080g" \
        -n 1 \
        -R "rusage[mem=32G]" \
        -W 100:00 \
        -oo "logs/diffusion_regressor.${combo}.%J.out" \
        -eo "logs/diffusion_regressor.${combo}.%J.err" \
        ./qsubs/diffusion_regressor.sh "${dataset}" "${splitmode}" "${embedding}"
    done
  done
done

