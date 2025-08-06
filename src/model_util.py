from transformers import AutoTokenizer, AutoModel

import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(model_name: str) -> AutoModel:
    """
    Load an ESM model safely.

    - Prefers safetensors if available (no torch.load / pickle)
    - Works offline with local paths
    - Enforces HF_HOME for caching on HPC
    """
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f"Loading model: {model_name}")

    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load ESM tokenizer."""
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f"Loading Tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
