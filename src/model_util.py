from transformers import AutoTokenizer, AutoModel

import os
import logging
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        return _embed_single_sequence(tokenizer, model, seq, max_len)

    logger.warning(
        f"Sequence length {len(seq)} exceeds model max {max_len}, using sliding windows..."
    )

    embeddings = []
    step = window_size - overlap
    for start in range(0, len(seq), step):
        window_seq = seq[start : start + window_size]
        emb = _embed_single_sequence(tokenizer, model, window_seq, max_len)
        embeddings.append(emb)
        if start + window_size >= len(seq):
            break

    return np.mean(embeddings, axis=0)


def _embed_single_sequence(tokenizer, model, seq, max_len, *, return_nan_on_error=True):
    try:
        max_input_length = max_len - 2  # Leave room for BOS/EOS
        if len(seq) > max_input_length:
            logger.warning(f"Truncating sequence from {len(seq)} to {max_input_length}")
            seq = seq[:max_input_length]

        tokens = tokenizer(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
        )

        input_ids = tokens["input_ids"]
        if input_ids.shape[1] > max_len:
            logger.error(f"Tokenized input too long: {input_ids.shape[1]} > {max_len}")
            raise ValueError("Tokenized input exceeds model max length.")

        with torch.no_grad():
            outputs = model(**tokens).last_hidden_state  # [1, L, H]

        if "attention_mask" in tokens:
            mask = tokens["attention_mask"]
            sum_embeddings = (outputs * mask.unsqueeze(-1)).sum(dim=1)
            lengths = mask.sum(dim=1, keepdim=True)
            embedding = sum_embeddings / lengths
        else:
            embedding = outputs.mean(dim=1)

        logger.debug(f"Embedding shape: {embedding.shape}")
        return embedding.squeeze().cpu().numpy()

    except Exception as e:
        logger.error(f"IndexError for sequence:\n{seq}")
        logger.error(f"Input IDs:\n{tokens.get('input_ids', 'Unavailable')}")
        if return_nan_on_error:
            return np.full(model.config.hidden_size, np.nan)
        else:
            raise
