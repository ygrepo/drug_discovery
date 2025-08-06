#!/bin/bash


#BSUB -J ativate_conda_env
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 0:30
#BSUB -o logs/activate_conda_env.%J.out
#BSUB -e logs/activate_conda_env.%J.err

set -euo pipefail

module purge
module load cuda/11.8 cudnn
module load anaconda3/latest
source $(conda info --base)/etc/profile.d/conda.sh

export PROJ=/sc/arion/projects/DiseaseGeneCell/Huang_lab_data
export CONDARC="$PROJ/conda/condarc"
conda activate /sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env

ml proxies/1 || true

export HF_HOME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.cache/huggingface"
mkdir -p "$HF_HOME"

# Must export for Python to see it
export MODEL_NAME="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models/esm1v_t33_650M_UR90S_5"

# ------------------ Torch + Model Test ----------------
python <<'PYCODE'
import os, torch, sys
from transformers import AutoModel

print("="*40)
print("✅ PyTorch version:", torch.__version__)
print("✅ CUDA version:", torch.version.cuda)
print("✅ GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ GPU device:", torch.cuda.get_device_name(0))
print("="*40)

model_path = os.environ.get("MODEL_NAME")
print(f"Attempting to load model from: {model_path}")
model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
print("✅ Model loaded successfully")
print("✅ All tests passed!")
PYCODE

