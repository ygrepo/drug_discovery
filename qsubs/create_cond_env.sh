#!/bin/bash
set -euo pipefail

# --- Modules / shell setup ---
module purge
module load anaconda3/latest
# module load cuda/12.4.0   # uncomment if your cluster provides CUDA 12 at runtime

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Paths (edit if needed) ---
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/dti"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"

# --- Keep installs off $HOME and avoid user-site leakage ---
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# --- Create env if missing ---
if [ ! -d "${ENV_PREFIX}" ]; then
  conda create --prefix "${ENV_PREFIX}" python=3.12 -y
fi

# --- Activate env (critical before any pip/conda ops) ---
conda activate "${ENV_PREFIX}"

# --- Upgrade pip INSIDE the env (safe) ---
python -m pip install --upgrade pip

# --- Install PyTorch for CUDA 12.x wheels (pick your minor) ---
CUDA_INDEX_URL="https://download.pytorch.org/whl/cu124"   # use cu121 if your cluster is CUDA 12.1
python -m pip install torch torchvision torchaudio --index-url "${CUDA_INDEX_URL}"

# --- Common libs ---
python -m pip install transformers safetensors accelerate
python -m pip install pandas numpy tqdm

# --- Sanity checks ---
python - <<'PY'
import os, sys, torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "Built for CUDA:", torch.version.cuda, "CUDA available:", torch.cuda.is_available())
print("PIP_CACHE_DIR:", os.environ.get("PIP_CACHE_DIR"))
print("CONDA_PKGS_DIRS:", os.environ.get("CONDA_PKGS_DIRS"))
from transformers import AutoModel
print("Transformers import: OK")
PY
