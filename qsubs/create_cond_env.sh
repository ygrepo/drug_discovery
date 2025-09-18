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
