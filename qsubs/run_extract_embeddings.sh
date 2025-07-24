#!/bin/bash
# submit_embeddings.sh — submit embedding jobs to LSF GPU queue

#BSUB -J $embeddings
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "rusage[mem=3000]"
#BSUB -W 2:00
#BSUB -o logs/${jobname}.%J.out
#BSUB -e logs/${jobname}.%J.err

module purge
module load cuda/11.8 cudnn
module load anaconda3/latest
source /hpc/packages/minerva-centos7/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate drug_discovery_env
which python

# python src/extract_embeddings.py \
#   --data_fn "$data_fn" \
#   --output_fn "$output_fn" \
#   --model_name facebook/esm2_t6_8M_UR50D \
#   --n 15 \
#   --log_dir logs \
#   --log_level INFO \
#   --seed 42
# EOF

# done
