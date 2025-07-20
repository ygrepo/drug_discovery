#!/bin/bash
# submit_embeddings.sh — submit embedding jobs to LSF GPU queue

# Create log directory
mkdir -p ../logs

# Define dataset input/output combinations
params=(
  "mutadescribe_data/structural_split/train.csv output/data/structural_split_train_with_embeddings.csv"
)

# Loop through each job definition
for entry in "${params[@]}"; do
  read -r data_fn output_fn <<< "$entry"

  jobname="embed_$(basename "$data_fn" .csv)"

  bsub \
    -J "$jobname" \
    -P acc_DiseaseGeneCell \
    -q gpu \
    -gpu "num=4" \
    -n 1 \
    -R "rusage[mem=32000]" \
    -W 2:00 \
    -o "logs/${jobname}.%J.out" \
    -e "logs/${jobname}.%J.err" \
    "module purge; module load python/3.12.5 cuda cudnn; \
     source ~/.venvs/drug_discovery/bin/activate && \
     which python && \
     python src/extract_embeddings.py \
       --data_fn \"$data_fn\" \
       --output_fn \"$output_fn\" \
       --model_name facebook/esm2_t6_8M_UR50D \
       --n 15 \
       --log_dir logs \
       --log_level INFO \
       --seed 42"
done
