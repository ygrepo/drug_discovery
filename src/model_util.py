from transformers import AutoTokenizer, AutoModel

import os
import logging
import numpy as np
import torch

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
    max_len = getattr(model.config, "max_position_embeddings", 1024)
    logger.info(f"Model max token length (from config): {max_len}")
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load ESM tokenizer."""
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f"Loading Tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def embed_sequence_sliding(tokenizer, model, seq, window_size=None, overlap=64):
    max_len = getattr(model.config, "max_position_embeddings", 1024)
    if window_size is None:
        window_size = max_len - 2

    if len(seq) <= window_size:
        return _embed_single_sequence(tokenizer, model, seq, logger)

    if logger:
        logger.warning(
            f"Sequence length {len(seq)} exceeds model max {max_len}, using sliding windows."
        )

    embeddings = []
    positions = range(0, len(seq), window_size - overlap)

    for start in positions:
        window_seq = seq[start : start + window_size]
        emb = _embed_single_sequence(tokenizer, model, window_seq, logger)
        embeddings.append(emb)
        if start + window_size >= len(seq):
            break

    return np.mean(embeddings, axis=0)


def _embed_single_sequence(tokenizer, model, seq, logger=None):
    max_len = getattr(model.config, "max_position_embeddings", 1024)
    if len(seq) > max_len - 2:
        if logger:
            logger.warning(f"Truncating sequence from {len(seq)} to {max_len - 2}")
        seq = seq[: max_len - 2]

    tokens = tokenizer(seq, return_tensors="pt", truncation=True, padding=False)

    with torch.no_grad():
        outputs = model(**tokens).last_hidden_state

    if "attention_mask" in tokens:
        mask = tokens["attention_mask"]
        sum_embeddings = (outputs * mask.unsqueeze(-1)).sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True)
        embedding = sum_embeddings / lengths
    else:
        embedding = outputs.mean(dim=1)

    return embedding.squeeze().cpu().numpy()
