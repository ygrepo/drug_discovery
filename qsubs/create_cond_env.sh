#!/bin/bash
set -euo pipefail

# --- Modules / shell setup ---
module purge
module load anaconda3/latest
# If your cluster has a CUDA 12 module, uncomment this line:
module load cuda/12.4.0

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Paths you can tweak ---
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/dti_env"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"

# --- Create cache dirs & set env vars ---
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# --- Create env (idempotent if it already exists) ---
if [ ! -d "${ENV_PREFIX}" ]; then
  conda create --prefix "${ENV_PREFIX}" python=3.12 -y
fi

# --- Activate it ---
conda activate "${ENV_PREFIX}"

# --- Upgrade pip ---
python -m pip install --upgrade pip

# --- Install PyTorch built for CUDA 12.x (cu124) ---
# If your cluster is CUDA 12.1, change cu124 -> cu121
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- Core libs ---
python -m pip install transformers safetensors accelerate
python -m pip install pandas numpy tqdm

# --- Sanity checks ---
python - <<'PY'
import os, torch
print("Torch:", torch.__version__)
print("CUDA built for:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("PIP_CACHE_DIR:", os.environ.get("PIP_CACHE_DIR"))
print("CONDA_PKGS_DIRS:", os.environ.get("CONDA_PKGS_DIRS"))
from transformers import AutoModel
print("Transformers import: OK")
PY
