#!/usr/bin/env python

"""
convert_to_safetensors.py

Download a Hugging Face model and safely convert it to safetensors,
handling shared weights (e.g., tied embeddings) automatically.
"""

import os
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# ---------------- Configuration ----------------
MODEL_NAME = "facebook/esm1v_t33_650M_UR90S_5"
PROJECT_MODELS_DIR = Path("/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models")

# Directories for raw and safe models
LOCAL_MODEL_DIR = PROJECT_MODELS_DIR / "esm1v_t33_650M_UR90S_5"
LOCAL_SAFE_DIR = PROJECT_MODELS_DIR / "esm1v_t33_650M_UR90S_5_safe"

LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SAFE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading model '{MODEL_NAME}' to {LOCAL_MODEL_DIR} ...")

# ---------------- Step 1: Download & Load Model ----------------
# This will cache weights in LOCAL_MODEL_DIR
model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=str(LOCAL_MODEL_DIR))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(LOCAL_MODEL_DIR))

print(f"✅ Model and tokenizer downloaded to {LOCAL_MODEL_DIR}")

# ---------------- Step 2: Save with Safe Serialization ----------------
print(f"Saving model and tokenizer to {LOCAL_SAFE_DIR} as safetensors...")

# This automatically handles shared weights
model.save_pretrained(LOCAL_SAFE_DIR, safe_serialization=True)
tokenizer.save_pretrained(LOCAL_SAFE_DIR)

print(f"✅ Conversion complete! Safe model directory: {LOCAL_SAFE_DIR}")
print("You can now load the model like this in your script:")
print(
    f"  from transformers import AutoModel\n"
    f"  model = AutoModel.from_pretrained('{LOCAL_SAFE_DIR}').eval()"
)
