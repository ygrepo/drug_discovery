#!/bin/bash
# submit_all_flow_matching.sh — submit one LSF job per (embedding, dataset, splitmode)

set -euo pipefail

# Define your grids:
DATASETS=( "BindingDB" )
#DATASETS=( "BindingDB" "BindDB" "Davis" "Kiba" )
SPLITMODES=( "random" "cold_protein" "cold_drug" )
EMBEDDINGS=( "ESMv1" "ESM2" "MUTAPLM" "ProteinCLIP" )

mkdir -p logs

for dataset in "${DATASETS[@]}"; do
  for splitmode in "${SPLITMODES[@]}"; do
    for embedding in "${EMBEDDINGS[@]}"; do
      combo="${embedding}_${dataset}_${splitmode}"

      # Submit one job for this combo.
      # We set name, logs, and also pass resource flags here (they override #BSUB in the script if duplicated).
      bsub \
        -J "flow_${combo}" \
        -W 100:00 \
        -oo "logs/flow_matching.${combo}.%J.out" \
        -eo "logs/flow_matching.${combo}.%J.err" \
        ./flow_matching_worker.sh "${dataset}" "${splitmode}" "${embedding}"
    done
  done
done

