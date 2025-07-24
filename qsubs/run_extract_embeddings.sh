#!/bin/bash
# submit_embeddings.sh — submit embedding jobs to LSF GPU queue


#BSUB -J embeddings
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "rusage[mem=3000]"
#BSUB -W 2:00
#BSUB -o logs/embeddings.%J.out
#BSUB -e logs/embeddings.%J.err

module purge
module load cuda/11.8 cudnn
module load anaconda3/latest
source /hpc/packages/minerva-centos7/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate drug_discovery_env

DATASET_DIR="mutadescribe_data"
DATA_FN="${DATASET_DIR}/structural_split/train.csv"
OUTPUT_DIR="output/data"
OUTPUT_FN="${OUTPUT_DIR}/structural_split_train_with_embeddings.csv"
MODEL_NAME="facebook/esm2_t6_8M_UR50D"
N=15
LOG_DIR="logs"
LOG_LEVEL="INFO"
SEED=42

/hpc/users/greaty01/.conda/envs/drug_discovery_env/bin/python src/extract_embeddings.py \
  --data_fn "$DATA_FN" \
  --output_fn "$OUTPUT_FN" \
  --model_name "$MODEL_NAME" \
  --n "$N" \
  --log_dir "$LOG_DIR" \
  --log_level "$LOG_LEVEL" \
  --seed "$SEED"
