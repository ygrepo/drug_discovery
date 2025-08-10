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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load(
    model_name: str,
    model_dir: Path,
    safe_dir: Path,
) -> None:
    logger.info(f"Downloading model '{model_name}' to {model_dir} ...")

    # This will cache weights in model_dir
    model = AutoModel.from_pretrained(model_name, cache_dir=str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(model_dir))
    logger.info(f"Model and tokenizer downloaded to {model_dir}")

    # ---------------- Step 2: Save with Safe Serialization ----------------
    logger.info(f"Saving model and tokenizer to {safe_dir} as safetensors...")

    # This automatically handles shared weights
    model.save_pretrained(safe_dir, safe_serialization=True)
    tokenizer.save_pretrained(safe_dir)


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
        "--log_fn",
        type=str,
        default="logs",
        help="Path to save log file",
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
        logger.info(f"  Model name: {args.model_name}")
        logger.info(f"  Model dir: {args.model_dir}")
        logger.info(f"  Safe dir: {args.safe_dir}")
        logger.info(f"  Log file: {args.log_fn}")
        logger.info(f"  Log level: {args.log_level}")

        logger.info("Loading model and tokenizer and saving safe embeddings")
        load(args.model_name, Path(args.model_dir), Path(args.safe_dir))

    except Exception as e:
        logger.exception("Script failed", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
