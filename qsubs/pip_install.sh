#!/bin/bash
set -euo pipefail

# --- Modules / shell setup ---
module purge
module load anaconda3/latest
module load cuda/12.4.0     # keep if your nodes provide CUDA 12.4 runtime

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Paths (edit if needed) ---
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/dti"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"  # conda cache (optional but recommended)

# --- Caches & hygiene ---
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# --- Activate env (must exist already) ---
conda activate "${ENV_PREFIX}"

# --- Use the env's Python and pip ---
PYTHON="${ENV_PREFIX}/bin/python"

# Upgrade pip inside the env (safe)
"${PYTHON}" -m pip install --upgrade pip

# --- Install PyTorch built for CUDA 12.x ---
CUDA_INDEX_URL="https://download.pytorch.org/whl/cu124"   # switch to cu121 if your cluster is CUDA 12.1
"${PYTHON}" -m pip install torch torchvision torchaudio --index-url "${CUDA_INDEX_URL}"

# --- Common libs ---
"${PYTHON}" -m pip install transformers safetensors accelerate
"${PYTHON}" -m pip install pandas numpy tqdm

# --- Sanity checks ---
"${PYTHON}" - <<'PY'
import os, sys, torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__,
      "Built for CUDA:", torch.version.cuda,
      "CUDA available:", torch.cuda.is_available())
print("PIP_CACHE_DIR:", os.environ.get("PIP_CACHE_DIR"))
print("CONDA_PKGS_DIRS:", os.environ.get("CONDA_PKGS_DIRS"))
from transformers import AutoModel
print("Transformers import: OK")
PY
