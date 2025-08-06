#!/usr/bin/env python

"""
convert_to_safetensors.py

Download a Hugging Face model and safely convert it to safetensors,
handling shared weights (e.g., tied embeddings) automatically.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------- Configuration ----------------
MODEL_NAME = "facebook/esm1v_t33_650M_UR90S_5"
PROJECT_MODELS_DIR = Path("/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models")

# Directories for raw and safe models
LOCAL_MODEL_DIR = PROJECT_MODELS_DIR / "esm1v_t33_650M_UR90S_5"
LOCAL_SAFE_DIR = PROJECT_MODELS_DIR / "esm1v_t33_650M_UR90S_5_safe"

LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SAFE_DIR.mkdir(parents=True, exist_ok=True)

def main():
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


model = AutoModel.from_pretrained(LOCAL_SAFE_DIR).eval()

print(f"✅ Conversion complete! Safe model directory: {LOCAL_SAFE_DIR}")
print("You can now load the model like this in your script:")
print(
    f"  from transformers import AutoModel\n"
    f"  model = AutoModel.from_pretrained('{LOCAL_SAFE_DIR}').eval()"
)
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model and convert it to safetensors format for use on HPC."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm1v_t33_650M_UR90S_5",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Output directory for the downloaded model",
    )  
    parser.add_argument(
        "--safe_dir",
        type=str,
        default="models_safe",
        help="Output directory for the safetensors model",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g., 'INFO', 'DEBUG')",
    )
    return parser.parse_args()

def main():
    args = parse_args()
        # Convert paths to absolute paths relative to project root
    logger = setup_logging(Path(args.log_fn), args.log_level)

    try:
        # Log configuration
        logger.info("Starting training with configuration:")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"  Data fn: {args.data_fn}")
        logger.info(f"  Output fn: {args.output_fn}")
        logger.info(f"  Model name: {args.model_name}")
        logger.info(f"  Log file: {args.log_fn}")
        logger.info(f"  Log level: {args.log_level}")
        logger.info(f"  Random seed: {args.seed}")

        logger.info("Extracting embeddings...")

    except Exception as e:
        logger.exception("Training failed", e)
        sys.exit(1)



if __name__ == "__main__":
    main()
