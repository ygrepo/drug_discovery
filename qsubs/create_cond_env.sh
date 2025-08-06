#!/bin/bash
module purge
module load anaconda3/latest

source $(conda info --base)/etc/profile.d/conda.sh

# Create new env named end in project .conda
conda create --prefix /sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env python=3.12 -y

# Activate it
conda activate /sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/drug_discovery_env


# Upgrade pip first
pip install --upgrade pip

# Install GPU-compatible PyTorch 2.6.0+ (with CUDA 11.8 for your cluster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face + safetensors
pip install transformers safetensors accelerate
pip install pandas numpy tqdm

python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)"
python -c "from transformers import AutoModel; print('Transformers OK')"
