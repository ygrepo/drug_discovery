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
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load ESM tokenizer."""
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f"Loading Tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = tokenizer.model_max_length
    logger.info(f"Model max token length: {max_len}")
    return tokenizer


def embed_sequence_sliding(
    tokenizer, model, seq, window_size=None, overlap=64
):
    """
    Embed a protein sequence using mean-pooled ESM embeddings with sliding windows.

    - Handles sequences longer than model max length
    - Uses overlapping windows and mean-pools across windows
    - Excludes padding tokens from mean-pooling
    """
    max_len = tokenizer.model_max_length
    if window_size is None:
        window_size = max_len - 2  # Leave room for BOS/EOS tokens

    # Short sequence: no sliding required
    if len(seq) <= window_size:
        return _embed_single_sequence(tokenizer, model, seq)

    logger.warning(
        f"Sequence length {len(seq)} exceeds model max {max_len}, using sliding windows."
    )

    embeddings = []
    positions = range(0, len(seq), window_size - overlap)

    for start in positions:
        window_seq = seq[start:start + window_size]
        emb = _embed_single_sequence(tokenizer, model, window_seq)
        embeddings.append(emb)

        if start + window_size >= len(seq):
            break

    # Mean-pool across all windows
    return np.mean(embeddings, axis=0)


def _embed_single_sequence(tokenizer, model, seq):
    """Helper to embed a single sequence without sliding window."""
    tokens = tokenizer(seq, return_tensors="pt", truncation=True, padding=False)
    with torch.no_grad():
        outputs = model(**tokens).last_hidden_state  # [1, L, H]

    if "attention_mask" in tokens:
        mask = tokens["attention_mask"]
        sum_embeddings = (outputs * mask.unsqueeze(-1)).sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True)
        embedding = sum_embeddings / lengths
    else:
        embedding = outputs.mean(dim=1)


    return embedding.squeeze().cpu().numpy()
