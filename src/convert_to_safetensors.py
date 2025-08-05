#!/usr/bin/env python

"""
convert_to_safetensors.py

Download a Hugging Face model and convert any .bin weights to .safetensors.
Stores the converted model in your project directory for secure HPC use.
"""

import os
from pathlib import Path
import torch
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

# ---------------- Configuration ----------------
MODEL_NAME = "facebook/esm1v_t33_650M_UR90S_5"
PROJECT_MODELS_DIR = Path("/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models")
LOCAL_MODEL_DIR = PROJECT_MODELS_DIR / "esm1v_t33_650M_UR90S_5"
LOCAL_SAFE_DIR = PROJECT_MODELS_DIR / "esm1v_t33_650M_UR90S_5_safe"

# Ensure directories exist
PROJECT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SAFE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading model '{MODEL_NAME}' to {LOCAL_MODEL_DIR} ...")

# ---------------- Step 1: Download model snapshot ----------------
model_path = snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_MODEL_DIR,
    local_dir_use_symlinks=False,  # safer for HPC
    ignore_patterns=["*.msgpack", "*.h5"],  # skip non-PyTorch files
)

print(f"Model downloaded to: {model_path}")

# ---------------- Step 2: Convert .bin to .safetensors ----------------
for root, _, files in os.walk(model_path):
    for file in files:
        if file.endswith(".bin"):
            src_file = Path(root) / file
            dst_file = LOCAL_SAFE_DIR / file.replace(".bin", ".safetensors")
            print(f"Converting {src_file} -> {dst_file}")

            state_dict = torch.load(src_file, map_location="cpu")
            save_file(state_dict, dst_file)

print(f"✅ Conversion complete. Safe model directory: {LOCAL_SAFE_DIR}")
print("You can now load the model like this:")
print(f"  from transformers import AutoModel\n"
      f"  model = AutoModel.from_pretrained('{LOCAL_SAFE_DIR}').eval()")
