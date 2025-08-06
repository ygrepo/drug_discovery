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


def _embed_single_sequence(tokenizer, model, seq, max_len):
    max_input_length = max_len - 2  # reserve space for BOS and EOS

    # Sanitize sequence: keep only valid amino acids
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    seq = "".join([aa for aa in seq if aa in valid_aa])

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

    # Diagnostic checks before model call
    input_ids = tokens["input_ids"]
    token_len = input_ids.shape[1]
    max_token_id = input_ids.max().item()

    vocab_size = model.embeddings.word_embeddings.num_embeddings
    pos_limit = model.embeddings.position_embeddings.num_embeddings

    logger.debug(f"Tokenized input shape: {input_ids.shape}")
    logger.debug(f"Max token ID: {max_token_id}, vocab size: {vocab_size}")
    logger.debug(f"Tokenized length: {token_len}, positional limit: {pos_limit}")

    try:
        with torch.no_grad():
            outputs = model(**tokens).last_hidden_state  # [1, L, H]
    except IndexError as e:
        logger.error(f"IndexError for sequence:\n{seq}")
        logger.error(f"Input IDs:\n{tokens['input_ids']}")
        raise e

    # Mean pooling using attention mask
    if "attention_mask" in tokens:
        mask = tokens["attention_mask"]
        sum_embeddings = (outputs * mask.unsqueeze(-1)).sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True)
        embedding = sum_embeddings / lengths
    else:
        embedding = outputs.mean(dim=1)

    return embedding.squeeze().cpu().numpy()
