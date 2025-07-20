#!/bin/bash
# submit_embeddings.sh — submit extract_embeddings.sh as a GPU job

# Create log directory if it doesn't exist
mkdir -p logs

# Optional: customize multiple input/output combos like this
params=(
  "mutadescribe_data/structural_split/train.csv output/data/structural_split_train_with_embeddings.csv"
  # Add more combinations here if needed
)

# Loop through all jobs
for entry in "${params[@]}"; do
  read -r data_fn output_fn <<< "$entry"

  jobname="embed_$(basename "$data_fn" .csv)"

  bsub \
    -J "$jobname" \
    -P acc_DiseaseGeneCell \
    -q gpu \
    -gpu "num=1:mode=exclusive_process" \
    -n 1 \
    -R "rusage[mem=32000]" \
    -W 2:00 \
    -o "logs/${jobname}.%J.out" \
    -eo "logs/${jobname}.%J.err" \
    "bash run/extract_embeddings.sh \
      --data_fn '$data_fn' \
      --output_fn '$output_fn' \
      --model_name 'facebook/esm2_t6_8M_UR50D' \
      --n 15 \
      --log_dir logs \
      --log_level INFO \
      --seed 42"
done
