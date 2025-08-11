from __future__ import annotations

from transformers import AutoTokenizer, AutoModel

import os
import logging
import numpy as np
import torch
from torch.serialization import add_safe_globals
from torch import nn
import yaml
import re
from transformers import AutoTokenizer, AutoModel
from src.mutaplm import MutaPLM
from enum import Enum
from pathlib import Path

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union


SYS_INFER = (
    "You are an expert at biology and life science. Now a user gives you several protein sequences "
    "and mutations. Please follow user instructions and answer their questions."
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Enum for model types
class ModelType(Enum):
    ESMV1 = "ESMv1"
    ESM2 = "ESM2"
    MUTAPLM = "MUTAPLM"
    PROTEINCLIP = "ProteinCLIP"

    @property
    def path(self) -> Path:
        """Local default path for this model type (override via env if you like)."""
        base = Path(
            os.getenv(
                "MODEL_BASE", "/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models"
            )
        )
        mapping = {
            ModelType.ESMV1: base / "esm1v_t33_650M_UR90S_5",
            ModelType.ESM2: Path(
                os.getenv("ESM2_PATH", str(base / "esm2_t33_650M_UR50D_safe"))
            ),
            ModelType.MUTAPLM: base / "mutaplm.pth",
            ModelType.PROTEINCLIP: base / "proteinclip",
        }
        return mapping[self]

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, s: str) -> "ModelType":
        """Case-insensitive, accepts value or name."""
        s_norm = s.strip().lower()
        for m in cls:
            if m.value.lower() == s_norm or m.name.lower() == s_norm:
                return m
        raise ValueError(f"Unknown model type: {s}")


MODEL_TYPE = list(ModelType)


def select_device(pref: str) -> torch.device:
    pref = (pref or "auto").lower()
    if pref.startswith("cuda"):
        return torch.device(pref) if torch.cuda.is_available() else torch.device("cpu")
    if pref == "mps":
        return (
            torch.device("mps")
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    if pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_or_default(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_HF_model(model_name: str) -> AutoModel:
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
    device = _device_or_default(None)
    model.to(device)
    model.eval()
    return model


def load_HF_tokenizer(model_name: str) -> AutoTokenizer:
    """Load ESM tokenizer."""
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f"Loading Tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


# Load Model Factory Function
def load_model_factory(
    model_type: ModelType,
    *,
    config_path: Path = Path("configs/mutaplm_inference.yaml"),
):
    """Factory function to return the correct model based on the model_type."""
    device = _device_or_default(None)
    logger.info(f"Using device: {device}")
    if model_type == ModelType.ESMV1 or model_type == ModelType.ESM2:
        model_path = str(model_type.path)
        model = load_HF_model(model_path)
        tokenizer = load_HF_tokenizer(model_path)
        logger.info(f"Loaded tokenizer: {model_path}")
        logger.info(f"Loaded model: {model_path}")
        return model, tokenizer
    if model_type == ModelType.MUTAPLM:
        model = create_mutaplm_model(config_path, device)
        model_path = model_type.path
        model = load_mutaplm_model(model, model_path)
        logger.info(f"Loaded model: {model_path}")
        return model, None
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def embed_sequence_sliding(tokenizer, model, seq, window_size=None, overlap=64):
    max_len = getattr(model.config, "max_position_embeddings", 1026)
    if window_size is None:
        window_size = max_len - 2
    logger.info(f"Window size: {window_size}")
    logger.info(f"Overlap: {overlap}")
    logger.info(f"Max len: {max_len}")

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

        logger.info(f"Embedding shape: {embedding.shape}")
        return embedding.squeeze().cpu().numpy()

    except Exception as e:
        logger.error(f"Error for sequence:\n{seq}")
        logger.error(f"Input IDs:\n{tokens.get('input_ids', 'Unavailable')}")
        logger.error(f"Error: {e}")
        if return_nan_on_error:
            return np.full(model.config.hidden_size, np.nan)
        else:
            raise


def create_mutaplm_model(cfg_path: Path, device):
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open() as f:
        model_cfg = yaml.safe_load(f)

    model_cfg["device"] = device
    model = MutaPLM(**model_cfg).to(device).eval()

    # Keep CPU in float32 (your class defaults to bf16 for from_pretrained)
    if device.type != "cuda":
        model.float()

    logger.info("Model loaded successfully.")
    return model


def load_mutaplm_model(model, checkpoint_path: Path, weights_only=False, strict=False):
    logger.info(f"Loading model checkpoint from {checkpoint_path}")
    new_ckpt = torch.load(
        open(checkpoint_path, "rb"), map_location="cpu", weights_only=weights_only
    )["model"]
    logger.info("Model checkpoint loaded successfully.")
    logger.info("Loading model state dict...")
    model.load_state_dict(new_ckpt, strict=strict)
    logger.info("Model state dict loaded successfully.")
    model.eval()
    return model


def load_mutaplmmodel_safely(
    model, checkpoint_path, device="cuda", weights_only=True, strict=True
):
    logger.info(f"Loading model checkpoint from {checkpoint_path}")
    # Allowlist the global(s) the checkpoint needs
    add_safe_globals([getattr])  # add more if error lists others

    # Load on CPU to avoid OOM during deserialization
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)

    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # Strip common prefixes from DDP/compile/PEFT saves
    cleaned = {}
    for k, v in state_dict.items():
        k = re.sub(r"^(module\.|_orig_mod\.)", "", k)
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    logger.info("Model state dict loaded successfully.")

    model.to(device)
    model.eval()
    return model


def load_mutaplm_model_from_config(device, config_path: Path, checkpoint_path: Path):
    logger.info(f"Using device: {device}")
    model = create_mutaplm_model(Path(config_path), device)
    model = load_mutaplm_model(model, checkpoint_path)
    logger.info("Model loaded successfully.")
    return model


def check_mutaplm_min(model) -> None:
    if model is None:
        logger.error("model is None")
        return

    def stat(name, t):
        if t is None:
            logger.warning("%s: <missing>", name)
            return
        t = t.detach().float()
        logger.info(
            "%-22s mean=%+.6f std=%.6f shape=%s req_grad=%s",
            name,
            t.mean().item(),
            t.std().item(),
            tuple(t.shape),
            getattr(t, "requires_grad", False),
        )

    # LLM input embeddings
    try:
        emb_w = model.llm.get_input_embeddings().weight
        stat("llm.emb.weight", emb_w)
    except Exception as e:
        logger.error("llm input embeddings not accessible: %s", e)

    # Key bridge params
    stat(
        "proj_protein1.weight",
        getattr(getattr(model, "proj_protein1", None), "weight", None),
    )
    logger.info(
        "proj_protein1.weight shape: %s",
        getattr(getattr(model, "proj_protein1", None), "weight", None).shape,
    )
    stat(
        "proj_protein2.weight",
        getattr(getattr(model, "proj_protein2", None), "weight", None),
    )
    logger.info(
        "proj_protein2.weight shape: %s",
        getattr(getattr(model, "proj_protein2", None), "weight", None).shape,
    )
    stat("proj_text.weight", getattr(getattr(model, "proj_text", None), "weight", None))
    logger.info(
        "proj_text.weight shape: %s",
        getattr(getattr(model, "proj_text", None), "weight", None).shape,
    )
    stat("query_protein1", getattr(model, "query_protein1", None))
    logger.info(
        "query_protein1 shape: %s", getattr(model, "query_protein1", None).shape
    )
    stat("query_protein2", getattr(model, "query_protein2", None))
    logger.info(
        "query_protein2 shape: %s", getattr(model, "query_protein2", None).shape
    )
    stat("soft_tokens", getattr(model, "soft_tokens", None))
    logger.info("soft_tokens shape: %s", getattr(model, "soft_tokens", None).shape)

    # Shape consistency
    try:
        H_esm = model.protein_model.config.hidden_size
        H_llm = model.llm.config.hidden_size
        W1 = getattr(getattr(model, "proj_protein1", None), "weight", None)
        W2 = getattr(getattr(model, "proj_protein2", None), "weight", None)
        Wt = getattr(getattr(model, "proj_text", None), "weight", None)

        if W1 is not None and W1.shape != (H_llm, H_esm):
            logger.error(
                "proj_protein1.weight %s != (%d, %d)", tuple(W1.shape), H_llm, H_esm
            )
        if W2 is not None and W2.shape != (H_llm, H_esm):
            logger.error(
                "proj_protein2.weight %s != (%d, %d)", tuple(W2.shape), H_llm, H_esm
            )
        if Wt is not None and Wt.shape != (H_esm, H_llm):
            logger.error(
                "proj_text.weight %s != (%d, %d)", tuple(Wt.shape), H_esm, H_llm
            )
    except Exception as e:
        logger.warning("shape checks skipped: %s", e)

    # Smoke test: one tiny forward to ensure no NaNs
    try:
        v = llm_context_embed_abs(model, "M")  # single Met is valid for ESM
        if torch.isnan(v).any():
            logger.error("NaNs in llm_context_embed_abs('M')")
        else:
            logger.info(
                "smoke test ok: llm_context_embed_abs('M') -> %s", tuple(v.shape)
            )
    except Exception as e:
        logger.error("smoke test failed: %s", e)
    logger.info("smoke test ok: llm_context_embed_abs('M') -> %s", tuple(v.shape))


def check_mutaplm_model(model):
    if model is None or not hasattr(model, "llm"):
        logger.error("model or model.llm is missing")
        return

    emb = next(model.llm.parameters(), None)
    if emb is None:
        logger.warning("llm has no parameters")
    else:
        emb = emb.detach().float()
        logger.info(
            "llm[first param]: mean=%.6f std=%.6f", emb.mean().item(), emb.std().item()
        )

    name_to_param = dict(model.named_parameters())
    for n in ["proj_protein1.weight", "query_protein1", "soft_tokens"]:
        p = name_to_param.get(n)
        if p is None:
            logger.warning("param '%s' not found", n)
            continue
        p = p.detach().float()
        logger.info("%s: mean=%.6f std=%.6f", n, p.mean().item(), p.std().item())


@torch.no_grad()
def llm_context_embed_abs(model, seq: str) -> torch.Tensor:
    """
    LLM-contextualized absolute embedding of a single sequence.
    Uses Stage-1 wrapper: [BOS, SYS, BOP, P1, EOP].
    Returns a float32 tensor [H_llm].
    """
    model.eval()

    # 1) Converts the raw amino acid sequence to LLM-aligned embeddings
    with model.maybe_autocast():
        p1 = model._encode_protein([seq], None)  # [1, Q1, H_llm]
    Q1 = p1.shape[1]

    # 2) Inserts special tokens around P1 to create the full Stage-1 LLM input
    wrapped_embeds, attn_mask = model._wrapped_sentence_inference(
        p1, None, muta_prompt=None, predict_function=None
    )  # [1, T, H_llm], [1, T]

    # 3) Run LLM and get last hidden states
    with model.maybe_autocast():
        hs = model.llm(
            inputs_embeds=wrapped_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[
            -1
        ]  # [1, T, H_llm]

    # 4) Compute the span for P1 residues
    sys_len = len(model.llm_tokenizer(SYS_INFER, add_special_tokens=False).input_ids)
    p1_start = 1 + sys_len + 1  # BOS(1) + SYS(sys_len) + BOP(1)
    p1_end = p1_start + Q1

    # 5) Mean-pools the hidden states for P1 residues
    v = hs[:, p1_start:p1_end, :].mean(dim=1).squeeze(0).float()  # [H_llm]
    return v


@torch.no_grad()
def llm_context_cosine(model, wt: str, mut: str) -> float:
    v_wt = llm_context_embed_abs(model, wt)
    v_mut = llm_context_embed_abs(model, mut)
    return (
        v_wt,
        v_mut,
        F.cosine_similarity(v_wt.unsqueeze(0), v_mut.unsqueeze(0)).item(),
    )


@torch.no_grad()
def fused_abs_pair(model, wt: str, mut: str):
    # Absolute LLM-contextual vectors
    wt_ctx = llm_context_embed_abs(model, wt)  # [H_llm]
    mut_ctx = llm_context_embed_abs(model, mut)  # [H_llm]
    # Pre-LLM absolute means (optional)
    p1_wt = model._encode_protein([wt], None).mean(dim=1).squeeze(0).float()
    p1_mut = model._encode_protein([mut], None).mean(dim=1).squeeze(0).float()
    fused = torch.cat([p1_wt, p1_mut, wt_ctx, mut_ctx, (mut_ctx - wt_ctx)], dim=-1)
    return dict(wt_ctx=wt_ctx, mut_ctx=mut_ctx, fused=fused)


@torch.no_grad()
def llm_context_embed_batch(model, seqs: list[str]) -> torch.Tensor:
    """
    Batched LLM-contextualized embeddings for many sequences.
    Returns [N, H_llm] float32.
    """
    model.eval()
    with model.maybe_autocast():
        p1 = model._encode_protein(seqs, None)  # [N, Q1, H_llm]
    N, Q1, H = p1.shape

    wrapped_embeds, attn_mask = model._wrapped_sentence_inference(
        p1, None, muta_prompt=None, predict_function=None
    )  # [N, T, H_llm], [N, T]

    with model.maybe_autocast():
        hs = model.llm(
            inputs_embeds=wrapped_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[
            -1
        ]  # [N, T, H_llm]

    sys_len = len(model.llm_tokenizer(SYS_INFER, add_special_tokens=False).input_ids)
    p1_start = 1 + sys_len + 1
    p1_end = p1_start + Q1

    v = hs[:, p1_start:p1_end, :].mean(dim=1).float()  # [N, H_llm]
    return v


@torch.no_grad()
def fused_in_llm(
    model,
    wt: str,
    mut: str,
    *,
    func_text: str = "Describe the protein function.",
    muta_prompt: str = "Describe the mutation impact.",
):
    """
    Returns a dict with contextualized LLM embeddings for WT & Mut and a fused vector.

    Preconditions:
      - `model` is a MutaPLM with a **pretrained LLM** loaded.
      - Ideally, the MutaPLM bridge (query tokens, projections, soft tokens) is finetuned.
      - Call model.eval() beforehand. On CPU, ensure model.float().

    Outputs:
      - "llm_ctx_wt"  : [1, H_llm]
      - "llm_ctx_mut" : [1, H_llm]
      - "llm_ctx_delta": [1, H_llm]
      - "esm_llm_wt", "esm_llm_mut": [1, H_llm] (pre-LLM pooled)
      - "fused"       : [1, 5*H_llm] (concat of above)
      - "spans"       : dict with (start, end) indices for P1/P2 in the LLM sequence
    """

    # 1) ESM→LLM pooled tokens for WT & Mut
    #    p1, p2: [1, Q, H_llm] (already projected into LLM space)
    with model.maybe_autocast():  # safe on CUDA; nullcontext on CPU
        p1, p2 = model._encode_protein([wt], [mut])
    Q1, Q2 = p1.shape[1], p2.shape[1]

    # 2) Build the same wrapped inputs the FT path uses (we only need the first two outputs)
    #    NOTE: pass func_text without "</s>" (the method will append it internally)
    dummy_text = ["Short answer."]  # not used for pooling
    wrapped = model._wrapped_sentence_ft(
        protein1_embeds=p1,
        protein2_embeds=p2,
        mut_entry=["[1]"],  # dummy; only used if t2m=True, which we don't need here
        p_function=[func_text],  # DO NOT append "</s>" here; method does it
        muta_prompt=[muta_prompt],
        text=dummy_text,
    )
    batched_embeds1, batched_attn_mask1 = (
        wrapped[0],
        wrapped[1],
    )  # [1, T, H_llm], [1, T]

    # 3) LLM forward to get contextual hidden states over the whole wrapped sequence
    with model.maybe_autocast():
        out = model.llm(
            inputs_embeds=batched_embeds1,
            attention_mask=batched_attn_mask1,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[
            -1
        ]  # [1, T, H_llm]

    # 4) Compute exact spans for P1 and P2
    #     Use the exact SYS string hardcoded in _wrapped_sentence_ft of your class:
    sys_str = (
        "You are an expert at biology and life science. Now a user gives you several protein sequences "
        "and mutations. Please follow user instructions and answer their questions. Based on the following "
        "protein sequence, please describe its function."
    )
    # Token counts (no special tokens)
    sys_len = len(model.llm_tokenizer(sys_str, add_special_tokens=False).input_ids)
    func_len = len(
        model.llm_tokenizer(func_text + "</s>", add_special_tokens=False).input_ids
    )  # they append "</s>"
    mut_len = len(model.llm_tokenizer(muta_prompt, add_special_tokens=False).input_ids)

    # Layout:
    # [BOS(1), SYS(sys_len), BOP(1), P1(Q1), EOP(1), FUNC(func_len), MUT(mut_len), BOM(1), P2(Q2), EOM(1), TEXT(...)]
    idx = 0
    idx += 1  # BOS
    idx += sys_len  # SYS
    idx += 1  # BOP
    p1_start = idx
    p1_end = p1_start + Q1
    idx = p1_end
    idx += 1  # EOP
    idx += func_len  # FUNC
    idx += mut_len  # MUT
    idx += 1  # BOM
    p2_start = idx
    p2_end = p2_start + Q2
    # (we don't need to advance further for pooling)

    # 5) Pool the LLM hidden states across P1/P2 spans
    p1_ctx = out[:, p1_start:p1_end, :].mean(dim=1)  # [1, H_llm]
    p2_ctx = out[:, p2_start:p2_end, :].mean(dim=1)  # [1, H_llm]
    delta_ctx = p2_ctx - p1_ctx  # [1, H_llm]

    # 6) (Optional) also include the pre-LLM pooled vectors
    p1_mean = p1.mean(dim=1)  # [1, H_llm]
    p2_mean = p2.mean(dim=1)  # [1, H_llm]

    fused = torch.cat(
        [p1_mean, p2_mean, p1_ctx, p2_ctx, delta_ctx], dim=-1
    )  # [1, 5*H_llm]

    return {
        "llm_ctx_wt": p1_ctx,
        "llm_ctx_mut": p2_ctx,
        "llm_ctx_delta": delta_ctx,
        "esm_llm_wt": p1_mean,
        "esm_llm_mut": p2_mean,
        "fused": fused,
        "spans": {"p1": (p1_start, p1_end), "p2": (p2_start, p2_end)},
    }


@torch.no_grad()
def fused_pre_llm(model, wt: str, mut: str):
    # ESM→LLM pooled tokens
    p1, p2 = model._encode_protein([wt], [mut])  # each: [1, Q, Hllm]
    p1_mean = p1.mean(dim=1)  # [1, Hllm]
    p2_mean = p2.mean(dim=1)  # [1, Hllm]
    delta = p2_mean - p1_mean  # [1, Hllm]
    # Simple fusions you can try:
    fused_cat = torch.cat([p1_mean, p2_mean, delta], dim=-1)  # [1, 3*Hllm]
    return {
        "esm_llm_wt_tokens": p1,
        "esm_llm_mut_tokens": p2,
        "esm_llm_wt": p1_mean,
        "esm_llm_mut": p2_mean,
        "esm_llm_delta": delta,
        "fused": fused_cat,
    }


@torch.no_grad()
def soft_mutation_embed(model, wt: str, *, func_text: str, mut_text: str):
    """
    Returns a single [1, H_llm] vector summarizing the mutation via the soft-token span.

    - Uses the inference helper that exposes batched_regress_ids (mask for soft tokens).
    - Does NOT require model.t2m=True.
    """

    # 1) Get pooled ESM->LLM tokens for the WT only
    with model.maybe_autocast():  # autocast if CUDA, nullcontext if CPU
        p1 = model._encode_protein([wt], None)  # [1, Q1, H_llm]

    # 2) Build the inference sequence with mut_text to obtain soft-token mask
    # predict_function must be provided; muta_prompt is not used in this branch
    be, am, soft_ids = model._wrapped_sentence_inference(
        protein1_embeds=p1,
        protein2_embeds=None,
        muta_prompt=[""],  # unused here
        predict_function=[func_text],  # your function text (no </s> needed)
        mut_text=[mut_text],  # textual description of the mutation/effect
    )
    # shapes: be [1, T, H_llm], am [1, T], soft_ids [1, T] (bool)

    # 3) LLM forward, then mean-pool over the soft-token positions
    with model.maybe_autocast():
        hs_last = model.llm(
            inputs_embeds=be,
            attention_mask=am,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[
            -1
        ]  # [1, T, H_llm]

    # Select the soft-token block; ensure we have at least one position
    n_soft = int(soft_ids.sum().item())
    assert n_soft > 0, "soft_mutation_embed: soft_ids mask is empty."
    soft_vec = (
        hs_last[soft_ids].view(1, n_soft, hs_last.size(-1)).mean(dim=1)
    )  # [1, H_llm]
    return soft_vec


@torch.no_grad()
def fused_in_llm_plus_soft(
    model,
    wt: str,
    mut: str,
    *,
    func_text: str = "Describe the protein function.",
    muta_prompt: str = "Describe the mutation impact.",
    soft_mut_text: str | None = None,
):
    """
    - Builds the fused vector from fused_in_llm (WT_ctx, Mut_ctx, Δ_ctx, pre-LLM means).
    - Adds a soft-token mutation vector and concatenates it.
    Returns a dict with all parts plus 'fused_plus_soft'.
    """
    # 1) Base fused vectors (WT/Mut contextual + pre-LLM)
    base = fused_in_llm(model, wt, mut, func_text=func_text, muta_prompt=muta_prompt)
    fused = base["fused"]  # [1, 5*H_llm]

    # 2) Soft-token mutation embedding (uses WT only + mut_text)
    # If no custom text is provided, derive a minimal one from mut vs wt (optional; here we require explicit)
    if soft_mut_text is None:
        # A safe default; you can pass something richer (e.g., "A70K substitution in catalytic pocket")
        soft_mut_text = "Summarize the mutation effect."

    soft_vec = soft_mutation_embed(
        model, wt, func_text=func_text, mut_text=soft_mut_text
    )  # [1, H_llm]

    # 3) Concatenate
    fused_plus_soft = torch.cat([fused, soft_vec], dim=-1)  # [1, 6*H_llm]

    base.update(
        {
            "soft_mut_vec": soft_vec,  # [1, H_llm]
            "fused_plus_soft": fused_plus_soft,  # [1, 6*H_llm]
        }
    )
    return base


@torch.no_grad()
def fused_soft(model, wt, mut, site):
    model.eval()
    if model.device.type != "cuda":
        model.float()  # keep CPU in fp32

    site = "A70K"
    func_text = "Describe the protein's function."  # or your stage-1 predicted function
    soft_mut_text = f"Mutation {site[0]}→{site[-1]} at position {site[1:-1]}."

    out = fused_in_llm_plus_soft(
        model,
        wt=wt,
        mut=mut,
        func_text=func_text,
        muta_prompt=f"Mutation {site[0]}→{site[-1]} at position {site[1:-1]}.",
        soft_mut_text=soft_mut_text,
    )

    vec = out["fused_plus_soft"]  # [1, 6*H_llm]
    soft_only = out["soft_mut_vec"]  # [1, H_llm]
    return vec, soft_only


@torch.no_grad()
def get_fused(model, wt, mut, site):
    # fusedA = fused_pre_llm(model, wt, mut)
    model.eval()
    if model.device.type != "cuda":
        model.float()  # keep CPU in fp32

    res = fused_in_llm(
        model,
        wt=wt,
        mut=mut,
        func_text="Summarize the protein's function.",
        muta_prompt=f"Mutation {site[0]}→{site[-1]} at position {site[1:-1]}.",
    )

    vec = res["fused"]  # [1, 5*H_llm] fused embedding
    delta = res["llm_ctx_delta"]  # [1, H_llm] mutation delta (contextual)

    return vec, delta
