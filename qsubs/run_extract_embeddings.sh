#!/bin/bash
# submit_embeddings.sh — submit embedding jobs to LSF GPU queue

# Create log directory
mkdir -p logs

# Define dataset input/output combinations
params=(
  "mutadescribe_data/structural_split/train.csv output/data/structural_split_train_with_embeddings.csv"
)

# Loop through each job definition
for entry in "${params[@]}"; do
  read -r data_fn output_fn <<< "$entry"

  jobname="embed_$(basename "$data_fn" .csv)"

  # Submit job with correct environment and quoting
  bsub <<EOF
#!/bin/bash
#BSUB -J $jobname
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=4"
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 2:00
#BSUB -o logs/${jobname}.%J.out
#BSUB -e logs/${jobname}.%J.err

module purge
module load cuda/11.8 cudnn
module load anaconda3/latest
source /hpc/packages/minerva-centos7/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate drug_discovery_env

python src/extract_embeddings.py \
  --data_fn "$data_fn" \
  --output_fn "$output_fn" \
  --model_name facebook/esm2_t6_8M_UR50D \
  --n 15 \
  --log_dir logs \
  --log_level INFO \
  --seed 42
EOF

done
