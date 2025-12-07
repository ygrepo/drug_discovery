from __future__ import annotations

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import os
import numpy as np
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
import torch.nn.functional as F
import yaml
import re
from src.mutaplm import MutaPLM
from enum import Enum
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Optional, Union, List, Literal
import onnxruntime as ort
import warnings

MODEL_DIR = Path(__file__).parent.parent / "pretrained"

# map ESM2 layer counts to HF repos (used when local path isn't available)
_ESM2_REPO = {
    6: "facebook/esm2_t6_8M_UR50D",
    12: "facebook/esm2_t12_35M_UR50D",
    30: "facebook/esm2_t30_150M_UR50D",
    33: "facebook/esm2_t33_650M_UR50D",
    36: "facebook/esm2_t36_3B_UR50D",
}

from src.utils import cosine_similarity, save_csv_parquet_torch, get_logger

logger = get_logger(__name__)


SYS_INFER = (
    "You are an expert at biology and life science. Now a user gives you several protein sequences "
    "and mutations. Please follow user instructions and answer their questions."
)


try:
    from esm import pretrained  # FAIR’s original library
except ImportError:
    logger.warn("FAIR esm not installed. `pip install fair-esm`")

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_sequence(seq: str) -> tuple[str, int]:
    """
    Replace invalid amino acids with 'X'. Return (sanitized_seq, n_replaced).
    """
    seq = seq.strip().upper()
    replaced = sum(1 for aa in seq if aa not in VALID_AAS)
    clean_seq = "".join([aa if aa in VALID_AAS else "X" for aa in seq])
    return clean_seq, replaced


# Enum for model types
class ModelType(Enum):
    ESMV1 = "ESMv1"
    ESM2 = "ESM2"
    MUTAPLM = "MUTAPLM"
    PROTEINCLIP = "ProteinCLIP"
    LLAMA = "LLAMA"

    @property
    def path(self) -> Path:
        """Local default path for this model type."""
        base = Path(
            os.getenv(
                "MODEL_BASE", "/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models"
            )
        )
        mapping = {
            # ESMv1: return hub alias (cleaner)
            # ModelType.ESMV1: "esm1v_t33_650M_UR90S_5",
            # ModelType.ESMV1: "/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models/esm1v_t33_650M_UR90S_5",
            # ESM2 can be an HF repo id or a local dir
            ModelType.ESMV1: Path(
                "/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/drug_discovery/output/esm1v_local"
            ),
            ModelType.ESM2: Path(
                os.getenv("ESM2_PATH", str(base / "esm2_t33_650M_UR50D_safe"))
            ),
            ModelType.MUTAPLM: base / "mutaplm.pth",
            ModelType.PROTEINCLIP: base / "proteinclip",
            ModelType.LLAMA: "meta-llama/Meta-Llama-3-8B-Instruct",
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


PLM_MODEL = [ModelType.ESMV1, ModelType.ESM2, ModelType.MUTAPLM, ModelType.PROTEINCLIP]
PLM_MODEL_NO_ESM1V = [ModelType.ESM2, ModelType.MUTAPLM, ModelType.PROTEINCLIP]
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


def load_HF_AutoModel(model_name: str) -> AutoModelForCausalLM:
    """
    Load an AutoModelForCausalLM model.
    - If `model_name` is a local path that exists: load from disk.
    - If it's a local path that does not exist: fallback to HF Hub.
    - If it's a Hub repo ID: use local cache or download if not cached.
    """
    logger.info(f"HF_HOME: {os.environ.get('HF_HOME')}")
    logger.info(f"Loading model: {model_name}")

    # --- optional auth token
    HF_TOKEN_PATH = os.environ.get("HF_TOKEN_PATH")
    HF_TOKEN = None
    if HF_TOKEN_PATH is not None and os.path.exists(HF_TOKEN_PATH):
        with open(HF_TOKEN_PATH, "r") as f:
            HF_TOKEN = f.read().strip()

    if os.path.isdir(model_name):
        # Local path exists → use it
        logger.info(f"Loading model from local path: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    else:
        # Treat as Hub repo (or fallback if local path is missing)
        logger.info(f"Loading model from HF Hub or cache: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
        ).eval()

    device = _device_or_default(None)
    model.to(device)
    return model


def load_HF_tokenizer(
    model_name: str,
    *,
    HF_TOKEN: str | None = None,
    CACHE_DIR: str | None = None,
) -> AutoTokenizer:
    """Load ESM tokenizer."""

    if HF_TOKEN is not None:
        logger.info(f"Using HF_TOKEN: {HF_TOKEN}, CACHE_DIR: {CACHE_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, token=HF_TOKEN, cache_dir=CACHE_DIR
        )
    elif CACHE_DIR is not None:
        logger.info(f"Using CACHE_DIR: {CACHE_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    else:
        logger.info("Using HF_TOKEN: None, CACHE_DIR: None")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def _resolve_parent_esm2_path(layers: int) -> str:
    """
    Prefer your local mirror for 33-layer if present, otherwise fall back to HF repo.
    Extend here if you have local mirrors for 6/12/30/36.
    """
    if layers == 33:
        local = ModelType.ESM2.path  # your /.../esm2_t33_650M_UR50D_safe
        if Path(local).exists():
            return str(local)
    # fall back to HF repo id
    return _ESM2_REPO[layers]


def _ensure_torch_home() -> Path:
    hub_dir = (
        os.environ.get("TORCH_HOME")
        or "/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.torch_hub"
    )
    hub_dir = Path(hub_dir)
    (hub_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    try:
        torch.hub.set_dir(str(hub_dir))
    except Exception:
        pass
    return hub_dir


def _hub_name_from_ref(model_ref: str | Path) -> str:
    """
    Accepts either a hub alias (esm1v_t33_650M_UR90S_5) or a filesystem path.
    - If it's a .pt path: return its basename without directory (used for cache file name).
    - If it's a path-like string without .pt: use basename as the hub alias.
    - Else, return the string as-is.
    """
    ref = str(model_ref)
    if ref.endswith(".pt"):
        return Path(ref).name.replace(".pt", "")
    if os.path.sep in ref:
        return Path(ref).name
    return ref


def load_fair_esm_v1_cached(model_ref: str | Path, *, device: torch.device):
    """
    ESMv1 loader with cache detection:
    - If model_ref is a local .pt path -> load local.
    - Else treat it as a hub alias and check $TORCH_HOME/checkpoints/<alias>.pt:
        - If exists -> load local from cache.
        - Else -> download from hub (to cache) and load.
    Also handles corrupted cache files by removing and re-downloading once.
    """
    hub_dir = _ensure_torch_home()
    ref = str(model_ref)
    is_local = ref.endswith(".pt") and Path(ref).is_file()
    logger.info(f"Loading ESMv1 model: {ref} (is_local={is_local})")
    if is_local:
        # Local .pt explicitly provided
        model, alphabet = pretrained.load_model_and_alphabet_local(ref)
        model = model.to(device).eval()
        return model, alphabet, f"local:{ref}"

    # Hub path: resolve expected cache file
    hub_name = _hub_name_from_ref(ref)  # e.g., "esm1v_t33_650M_UR90S_5"
    cache_ckpt = hub_dir / "checkpoints" / f"{hub_name}.pt"

    # 1) Try cached file if present
    if cache_ckpt.is_file():
        try:
            model, alphabet = pretrained.load_model_and_alphabet_local(str(cache_ckpt))
            model = model.to(device).eval()
            return model, alphabet, f"cache:{cache_ckpt}"
        except Exception as e:
            # Corrupted/incompatible cache -> delete and re-download
            try:
                cache_ckpt.unlink(missing_ok=True)
            except Exception:
                pass  # best effort
            # fall through to hub download

    # 2) Download from hub (this writes into $TORCH_HOME/checkpoints/)
    model, alphabet = pretrained.load_model_and_alphabet(hub_name)
    model = model.to(device).eval()
    return model, alphabet, f"hub:{hub_name}"


# Load Model Factory Function
def load_model_factory(
    model_type: ModelType,
    *,
    config_path: Path = Path("configs/mutaplm_inference.yaml"),
):
    """
    Returns:
      (model, tokenizer) for HF models and PROTEINCLIP (parent PLM + tokenizer).
      (model, None)      for MutaPLM.
    For PROTEINCLIP: loads the matching parent ESM2 (by layers) and attaches ONNX head at `model.proteinclip`.
    """
    device = _device_or_default(None)
    logger.info("Using device: %s", device)

    # if model_type == ModelType.ESMV1:
    #     model_ref = model_type.path  # can be a hub name *or* a local .pt path
    #     model, alphabet, src = load_fair_esm_v1_cached(model_ref, device=device)
    #     _attach_max_len(model, model_type)
    #     logger.info("Loaded FAIR ESMv1 model and Alphabet (%s)", src)
    #     return model, alphabet
    if model_type == ModelType.ESMV1:
        model_path = str(model_type.path)
        model = load_HF_model(model_path)
        model.eval()
        CACHE_DIR = os.environ.get("HF_CACHE_DIR")
        logger.info(f"Loading Tokenizer from {CACHE_DIR}")
        tokenizer = load_HF_tokenizer(model_path, HF_TOKEN=None, CACHE_DIR=CACHE_DIR)
        _attach_max_len(model, model_type)
        logger.info("Loaded HF ESM1V: %s", model_path)
        return model, tokenizer

    if model_type == ModelType.ESM2:
        # (unchanged HF path)
        model_path = str(model_type.path)
        model = load_HF_model(model_path)
        model.eval()
        CACHE_DIR = os.environ.get("HF_CACHE_DIR")
        logger.info(f"Loading Tokenizer from {CACHE_DIR}")
        tokenizer = load_HF_tokenizer(model_path, HF_TOKEN=None, CACHE_DIR=CACHE_DIR)
        _attach_max_len(model, model_type)
        logger.info("Loaded HF ESM2: %s", model_path)
        return model, tokenizer

    if model_type == ModelType.MUTAPLM:
        model = create_mutaplm_model(config_path, device)
        model_path = model_type.path
        model = load_mutaplm_model(model, model_path)
        _attach_max_len(model, model_type)
        logger.info("Loaded model: %s", model_path)
        return model, None

    if model_type == ModelType.LLAMA:
        model_path = model_type.path
        model = load_HF_AutoModel(model_path)
        tokenizer = load_HF_tokenizer(model_path)
        _attach_max_len(model, model_type)
        logger.info("Loaded model: %s", model_path)
        return model, tokenizer

    if model_type == ModelType.PROTEINCLIP:
        # 1) Decide which ESM2 depth to use (default 33); allow override via env
        layers = int(os.getenv("PROTEINCLIP_ESM_LAYERS", "33"))
        parent_ref = _resolve_parent_esm2_path(layers)

        # 2) Load the matching parent PLM + tokenizer
        model = load_HF_model(parent_ref)
        tokenizer = load_HF_tokenizer(parent_ref)
        _attach_max_len(model, model_type)
        logger.info("Loaded parent PLM (ESM2-%d layers): %s", layers, parent_ref)

        # 3) Sanity check: confirm actual num_hidden_layers matches requested
        actual_layers = int(getattr(model.config, "num_hidden_layers", layers))
        if actual_layers != layers:
            logger.warning(
                "Requested ESM2 layers=%d, but loaded model has %d layers.",
                layers,
                actual_layers,
            )
            layers = actual_layers  # keep them in sync for the head

        # 4) Load the ProteinCLIP head for that depth and validate input dim
        hidden_size = int(getattr(model.config, "hidden_size", 1280))
        clip_head = load_proteinclip(
            "esm",
            layers,
            model_dir=ModelType.PROTEINCLIP.path,
            expected_in_dim=hidden_size,
        )

        # 5) Attach to model
        setattr(model, "proteinclip", clip_head)
        logger.info(
            "Attached ProteinCLIP head (input_dim=%s, output_dim=%s)",
            clip_head.input_dim,
            clip_head.output_dim,
        )
        return model, tokenizer

    raise ValueError(f"Unknown model type: {model_type}")


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


class ONNXModel:
    """Wrapper for an ONNX model to provide a more familiar interface."""

    def __init__(self, path: Path):
        # Use CPU provider by default; swap to CUDA EP if your ORT build supports it
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape
        # shapes like [None, 1280] -> take the last dim if static
        self.input_dim = (
            int(in_shape[-1]) if isinstance(in_shape[-1], (int, np.integer)) else None
        )
        self.output_dim = (
            int(out_shape[-1]) if isinstance(out_shape[-1], (int, np.integer)) else 128
        )

    def predict(self, x: np.ndarray, apply_norm: bool = True):
        assert x.ndim == 1
        if apply_norm:
            n = float(np.linalg.norm(x))
            if n > 0:
                x = x / n
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return self.session.run(None, {self.input_name: x[None, :]})[0].squeeze()

    def predict_batch(self, x: np.ndarray, apply_norm: bool = True):
        assert x.ndim == 2
        if apply_norm:
            n = np.linalg.norm(x, axis=1, keepdims=True)
            n[n == 0.0] = 1.0
            x = x / n
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return self.session.run(None, {self.input_name: x})[0]


# --------------- ProteinCLIP loader ---------------
def load_proteinclip(
    model_arch: Literal["esm", "t5"],
    model_size: Optional[int] = None,
    *,
    model_dir: Optional[Path] = None,
    expected_in_dim: Optional[
        int
    ] = None,  # <— new: validate parent hidden size vs ONNX input
) -> ONNXModel:
    """
    Load the ProteinCLIP ONNX head that maps parent PLM embeddings -> 128-d unit vectors.
    By default, looks under ModelType.PROTEINCLIP.path.
    """
    if model_dir is None:
        model_dir = ModelType.PROTEINCLIP.path
    assert model_dir.is_dir(), f"ProteinCLIP directory does not exist: {model_dir}"

    if model_arch == "esm":
        assert (
            model_size is not None
        ), "ESM model requires a size (e.g., 33 for ESM2-33)."
        valid = {6, 12, 30, 33, 36}
        assert (
            model_size in valid
        ), f"Invalid ESM model size: {model_size} (valid: {sorted(valid)})"
        model_path = model_dir / f"proteinclip_esm2_{model_size}.onnx"
    elif model_arch == "t5":
        assert model_size is None, "T5 model does not have different sizes."
        model_path = model_dir / "proteinclip_prott5.onnx"
    else:
        raise ValueError(f"Invalid model architecture: {model_arch}")

    assert model_path.exists(), f"ProteinCLIP model path does not exist: {model_path}"
    logger.info("Loading ProteinCLIP head from %s", model_path)
    head = ONNXModel(model_path)

    # Validate parent hidden size against ONNX expected input
    if (
        expected_in_dim is not None
        and head.input_dim is not None
        and head.input_dim != expected_in_dim
    ):
        raise ValueError(
            f"ProteinCLIP head expects input_dim={head.input_dim}, "
            f"but parent hidden_size={expected_in_dim}. "
            f"Make sure you pair esm2_{model_size} with the matching head."
        )
    return head


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
def llm_context_cosine(
    model, wt: str, mut: str
) -> tuple[torch.Tensor, torch.Tensor, float]:
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


@torch.no_grad()
def embed_df(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:

    # Embed all protein1 and protein2 sequences
    protein1_embeddings = []
    protein2_embeddings = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding pairs"):
        try:
            emb1 = embed_sequence_sliding(tokenizer, model, row["protein1"])
            emb2 = embed_sequence_sliding(tokenizer, model, row["protein2"])
            protein1_embeddings.append(emb1)
            protein2_embeddings.append(emb2)
        except Exception as e:
            logger.exception(f"Embedding error at row {idx}: {e}")
            protein1_embeddings.append(np.full(model.config.hidden_size, np.nan))
            protein2_embeddings.append(np.full(model.config.hidden_size, np.nan))

    # Save embeddings (note: storing arrays in DF is ok for moderate size)
    df["protein1_embedding"] = protein1_embeddings
    df["protein2_embedding"] = protein2_embeddings

    # Compute cosine similarity
    logger.info("Computing cosine similarity...")
    df["cosine_similarity"] = [
        cosine_similarity(emb1, emb2)
        for emb1, emb2 in zip(protein1_embeddings, protein2_embeddings)
    ]
    return df


# ---------- single-sequence sliding window (reuses your _embed_single_sequence) ----------


def _detect_max_len(model, model_type) -> int:
    """
    Return the maximum supported sequence length for this model.
    The value is conservative (does not include BOS/EOS padding budget).
    """
    # Import lazily or use your existing ModelType
    MT = (
        ModelType if isinstance(model_type, ModelType) else ModelType
    )  # no-op, just clarity

    if model_type == MT.ESMV1:
        # FAIR ESMv1 commonly 1022
        return int(getattr(model, "max_positions", 1022))

    if model_type in (MT.ESM2, MT.PROTEINCLIP):
        # HF ESM2 exposes .config.max_position_embeddings (commonly 1024)
        return int(
            getattr(getattr(model, "config", None), "max_position_embeddings", 1024)
        )

    if model_type == MT.MUTAPLM:
        # Prefer explicit attribute if your model defines it
        if hasattr(model, "max_len"):
            return int(model.max_len)
        if hasattr(model, "sequence_length"):
            return int(model.sequence_length)
        # Last resort: conservative default; warn once
        warnings.warn(
            "[MutaPLM] max_len not found on model; defaulting to 1024. "
            "Attach `model.max_len = <int>` after constructing your model to silence this.",
            RuntimeWarning,
        )
        return 1024

    raise ValueError(f"Unknown model type for max_len detection: {model_type}")


def _attach_max_len(model, model_type) -> int:
    """
    Detect and attach `model.max_len` if missing. Return the value.
    """
    if hasattr(model, "max_len"):
        try:
            val = int(model.max_len)
            return val
        except Exception:
            pass  # fall through to re-detect

    val = _detect_max_len(model, model_type)
    try:
        setattr(model, "max_len", int(val))
    except Exception:
        # If model is a torchscript or restricts setattr, just ignore
        pass
    return int(val)


@torch.no_grad()
def embed_sequence_sliding(
    tokenizer_or_alphabet,
    model,
    seq: str,
    *,
    window_size: Optional[int] = None,
    overlap: int = 64,
    agg: str = "mean",
    return_nan_on_error: bool = True,
) -> np.ndarray:
    """
    Sliding-window wrapper for both ESMv1 (FAIR esm) and ESMv2 (HF).
    - Detects model type by presence of `.alphabet` (FAIR ESMv1) vs HF config.
    - Uses mean pooling for now.
    """
    # --- detect max length
    if hasattr(model, "config"):  # HuggingFace (ESMv2)
        max_len = int(getattr(model.config, "max_position_embeddings", 1026))
    else:  # FAIR ESMv1
        max_len = getattr(model, "max_positions", 1026)

    if window_size is None:
        window_size = max_len - 2  # leave BOS/EOS space

    # fast path
    if len(seq) <= window_size:
        return _embed_single_sequence(
            tokenizer_or_alphabet,
            model,
            seq,
            max_len,
            return_nan_on_error=return_nan_on_error,
        )

    logger.warning(
        "Sequence length %d exceeds window_size %d; using sliding windows...",
        len(seq),
        window_size,
    )

    step = max(1, window_size - max(0, overlap))
    window_vecs: List[np.ndarray] = []

    start = 0
    while True:
        win = seq[start : start + window_size]
        if not win:
            break

        vec = _embed_single_sequence(
            tokenizer_or_alphabet,
            model,
            win,
            max_len,
            return_nan_on_error=return_nan_on_error,
        )
        window_vecs.append(vec)

        if start + window_size >= len(seq):
            break
        start += step

    if len(window_vecs) == 0:
        H = int(getattr(model.config, "hidden_size", 1280))
        return np.full(H, np.nan, dtype=np.float32)

    if agg == "mean":
        return np.mean(window_vecs, axis=0).astype(np.float32)
    raise ValueError(f"Unknown agg '{agg}'")


@torch.no_grad()
def retrieve_embeddings(
    model_type,  # ModelType
    model,
    df: pd.DataFrame,
    *,
    tokenizer=None,  # required for ESM1/ESM2/PROTEINCLIP
    seq_col: str = "protein1",
    # Sliding-window params for ESM1/ESM2/PROTEINCLIP:
    window_size: Optional[int] = None,
    overlap: int = 64,
    agg: str = "mean",
    # Row-batch size:
    batch_size: int = 16,
    output_fn: Optional[Path] = None,
    # --- ProteinCLIP options ---
    project_to_clip: bool = False,  # set True to project parent PLM vectors to CLIP space
    clip_model=None,  # ONNX head (will default to model.proteinclip if None)
) -> pd.DataFrame:
    """
    Adds columns: protein1_embedding, protein2_embedding, cosine_similarity.

    Modes:
      - ESM1/ESM2: embed via sliding windows (parent PLM space). If `project_to_clip=True`
                   (or model_type==PROTEINCLIP), project with ProteinCLIP and compute cosine in CLIP space.
      - PROTEINCLIP: same as ESM2 path but forces projection through ProteinCLIP.
      - MUTAPLM: uses llm_context_cosine(model, wt, mut) per pair.

    Notes:
      - When projecting to CLIP, embeddings in the output are the 128-d CLIP vectors.
      - If you want both spaces, run twice (once with projection, once without) and save in different columns.
    """
    out = df.copy()
    del df
    model.eval()

    # figure out which path we're on
    is_hf_parent = model_type.name in ("ESMV1", "ESM2", "PROTEINCLIP")
    use_clip = project_to_clip or (model_type.name == "PROTEINCLIP")

    if is_hf_parent and tokenizer is None:
        raise ValueError(f"{model_type.name} mode requires a tokenizer.")

    # resolve clip head if needed
    if is_hf_parent and use_clip and clip_model is None:
        clip_model = getattr(model, "proteinclip", None)
        if clip_model is None:
            raise ValueError(
                "ProteinCLIP projection requested but no clip_model provided or attached at model.proteinclip."
            )
    # for NaN fallbacks
    clip_out_dim = (
        int(getattr(clip_model, "output_dim", 128))
        if (is_hf_parent and use_clip)
        else None
    )
    parent_hidden_size = (
        int(getattr(getattr(model, "config", None), "hidden_size", 1280))
        if is_hf_parent
        else None
    )

    p_vecs: List[np.ndarray] = []
    # batching over rows
    for start in tqdm(range(0, len(out), batch_size), desc="Embedding"):
        batch_df = out.iloc[start : start + batch_size]
        s_list = batch_df[seq_col].astype(str).tolist()

        if is_hf_parent:
            for i, s in enumerate(s_list):
                idx = start + i
                try:
                    v = embed_sequence_sliding(
                        tokenizer,
                        model,
                        s,
                        window_size=window_size,
                        overlap=overlap,
                        agg=agg,
                        return_nan_on_error=True,
                    )

                    # If ProteinCLIP projection requested, project both parent vectors
                    if use_clip:
                        # ONNX head does its own unit-norm by default (if your wrapper’s predict applies_norm=True)
                        v = clip_model.predict(v)  # -> (128,)

                    p_vecs.append(v)

                except Exception as e:
                    logger.exception(
                        "Embedding error (HF%s) at row %d: %s",
                        " + CLIP" if use_clip else "",
                        idx,
                        e,
                    )
                    if use_clip:
                        H = clip_out_dim or 128
                    else:
                        H = parent_hidden_size or 1280
                    p_vecs.append(np.full(H, np.nan, dtype=np.float32))

        else:
            # MutaPLM path (unchanged)
            maybe_ctx = getattr(model, "maybe_autocast", None)
            ctx = (
                maybe_ctx()
                if callable(maybe_ctx)
                else torch.autocast(
                    device_type="cuda",
                    enabled=(next(model.parameters()).device.type == "cuda"),
                )
            )
            with ctx:
                for i, s in enumerate(s_list):
                    idx = start + i
                    try:
                        v = llm_context_embed_abs(model, s)
                        p_vecs.append(v.detach().cpu().numpy())
                    except Exception as e:
                        logger.exception(
                            "Embedding error (MutaPLM) at row %d: %s", idx, e
                        )
                        H = int(
                            getattr(
                                model, "llm_hidden", getattr(model, "hidden_size", 1024)
                            )
                        )
                        p_vecs.append(np.full(H, np.nan, dtype=np.float32))

    out[f"{seq_col}_embedding"] = p_vecs
    if output_fn is not None:
        save_csv_parquet_torch(out, output_fn)
    return out


# ---------- unified DF pipeline for ESM1/ESM2 (sliding) and MutaPLM (pair-wise) ----------
@torch.no_grad()
def retrieve_pair_embeddings(
    model_type,  # ModelType
    model,
    df: pd.DataFrame,
    *,
    tokenizer=None,  # required for ESM1/ESM2/PROTEINCLIP
    seq1_col: str = "protein1",
    seq2_col: str = "protein2",
    # Sliding-window params for ESM1/ESM2/PROTEINCLIP:
    window_size: Optional[int] = None,
    overlap: int = 64,
    agg: str = "mean",
    # Row-batch size:
    batch_size: int = 16,
    output_fn: Optional[Path] = None,
    # --- ProteinCLIP options ---
    project_to_clip: bool = False,  # set True to project parent PLM vectors to CLIP space
    clip_model=None,  # ONNX head (will default to model.proteinclip if None)
) -> pd.DataFrame:
    """
    Adds columns: protein1_embedding, protein2_embedding, cosine_similarity.

    Modes:
      - ESM1/ESM2: embed via sliding windows (parent PLM space). If `project_to_clip=True`
                   (or model_type==PROTEINCLIP), project with ProteinCLIP and compute cosine in CLIP space.
      - PROTEINCLIP: same as ESM2 path but forces projection through ProteinCLIP.
      - MUTAPLM: uses llm_context_cosine(model, wt, mut) per pair.

    Notes:
      - When projecting to CLIP, embeddings in the output are the 128-d CLIP vectors.
      - If you want both spaces, run twice (once with projection, once without) and save in different columns.
    """
    out = df.copy()
    model.eval()

    # figure out which path we're on
    is_hf_parent = model_type.name in ("ESMV1", "ESM2", "PROTEINCLIP")
    use_clip = project_to_clip or (model_type.name == "PROTEINCLIP")

    if is_hf_parent and tokenizer is None:
        raise ValueError(f"{model_type.name} mode requires a tokenizer.")

    # resolve clip head if needed
    if is_hf_parent and use_clip and clip_model is None:
        clip_model = getattr(model, "proteinclip", None)
        if clip_model is None:
            raise ValueError(
                "ProteinCLIP projection requested but no clip_model provided or attached at model.proteinclip."
            )

    # for NaN fallbacks
    clip_out_dim = (
        int(getattr(clip_model, "output_dim", 128))
        if (is_hf_parent and use_clip)
        else None
    )
    parent_hidden_size = (
        int(getattr(getattr(model, "config", None), "hidden_size", 1280))
        if is_hf_parent
        else None
    )

    p1_vecs: List[np.ndarray] = []
    p2_vecs: List[np.ndarray] = []
    sims: List[float] = []

    # batching over rows
    for start in tqdm(range(0, len(out), batch_size), desc="Embedding pairs"):
        batch_df = out.iloc[start : start + batch_size]
        s1_list = batch_df[seq1_col].astype(str).tolist()
        s2_list = batch_df[seq2_col].astype(str).tolist()

        if is_hf_parent:
            for i, (s1, s2) in enumerate(zip(s1_list, s2_list)):
                idx = start + i
                try:
                    v1 = embed_sequence_sliding(
                        tokenizer,
                        model,
                        s1,
                        window_size=window_size,
                        overlap=overlap,
                        agg=agg,
                        return_nan_on_error=True,
                    )
                    v2 = embed_sequence_sliding(
                        tokenizer,
                        model,
                        s2,
                        window_size=window_size,
                        overlap=overlap,
                        agg=agg,
                        return_nan_on_error=True,
                    )

                    # If ProteinCLIP projection requested, project both parent vectors
                    if use_clip:
                        # ONNX head does its own unit-norm by default (if your wrapper’s predict applies_norm=True)
                        v1 = clip_model.predict(v1)  # -> (128,)
                        v2 = clip_model.predict(v2)  # -> (128,)

                    p1_vecs.append(v1)
                    p2_vecs.append(v2)
                    sims.append(cosine_similarity(v1, v2))

                except Exception as e:
                    logger.exception(
                        "Embedding error (HF%s) at row %d: %s",
                        " + CLIP" if use_clip else "",
                        idx,
                        e,
                    )
                    if use_clip:
                        H = clip_out_dim or 128
                    else:
                        H = parent_hidden_size or 1280
                    p1_vecs.append(np.full(H, np.nan, dtype=np.float32))
                    p2_vecs.append(np.full(H, np.nan, dtype=np.float32))
                    sims.append(float("nan"))

        else:
            # MutaPLM path (unchanged)
            maybe_ctx = getattr(model, "maybe_autocast", None)
            ctx = (
                maybe_ctx()
                if callable(maybe_ctx)
                else torch.autocast(
                    device_type="cuda",
                    enabled=(next(model.parameters()).device.type == "cuda"),
                )
            )
            with ctx:
                for i, (wt, mut) in enumerate(zip(s1_list, s2_list)):
                    idx = start + i
                    try:
                        v_wt, v_mut, cos = llm_context_cosine(model, wt, mut)
                        p1_vecs.append(v_wt.detach().cpu().numpy())
                        p2_vecs.append(v_mut.detach().cpu().numpy())
                        sims.append(float(cos))
                    except Exception as e:
                        logger.exception(
                            "Embedding error (MutaPLM) at row %d: %s", idx, e
                        )
                        H = int(
                            getattr(
                                model, "llm_hidden", getattr(model, "hidden_size", 1024)
                            )
                        )
                        p1_vecs.append(np.full(H, np.nan, dtype=np.float32))
                        p2_vecs.append(np.full(H, np.nan, dtype=np.float32))
                        sims.append(float("nan"))

    out[f"{seq1_col}_embedding"] = p1_vecs
    out[f"{seq2_col}_embedding"] = p2_vecs
    out["cosine_similarity"] = sims

    logger.info(
        "Computed embeddings for %d pairs. Example sims: %s", len(out), sims[:5]
    )

    # Count how many had sanitization
    n_with_Xs = sum("X" in seq for seq in out[seq1_col]) + sum(
        "X" in seq for seq in out[seq2_col]
    )
    logger.info("Sequences with invalid residues replaced by 'X': %d", n_with_Xs)

    if output_fn is not None:
        logger.info("Saving embeddings to %s", output_fn)
        output_fn = Path(output_fn)

        # Save DataFrame metadata without the heavy arrays
        meta_df = out.drop(columns=[f"{seq1_col}_embedding", f"{seq2_col}_embedding"])
        meta_df.to_csv(output_fn.with_suffix(".csv"), index=False)

        # Save embeddings as compressed NumPy archive
        np.savez_compressed(
            output_fn.with_suffix(".npz"),
            protein1=np.stack(p1_vecs),
            protein2=np.stack(p2_vecs),
            cosine=np.array(sims),
        )

        logger.info("Saved metadata (.csv) and embeddings (.npz).")

    return out


@torch.no_grad()
def _embed_single_sequence(
    tokenizer_or_alphabet,
    model,
    seq: str,
    max_len: int,
    *,
    return_nan_on_error: bool = True,
    exclude_special: bool = True,
):
    try:
        device = next(model.parameters()).device

        # --- Path A: HuggingFace (ESMv2) ---
        if hasattr(model, "config"):
            tokens = tokenizer_or_alphabet(
                seq,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_len,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                add_special_tokens=True,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(
                    **{
                        k: v
                        for k, v in tokens.items()
                        if k in ("input_ids", "attention_mask")
                    }
                ).last_hidden_state

            # mean pool over non-special, non-pad tokens
            mask = tokens.get(
                "attention_mask", torch.ones(outputs.shape[:2], device=device)
            )
            if exclude_special and "special_tokens_mask" in tokens:
                mask = mask * (1 - tokens["special_tokens_mask"])
            mask = mask.to(outputs.dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            embedding = (outputs * mask.unsqueeze(-1)).sum(dim=1) / denom

        # --- Path B: FAIR ESMv1 ---
        else:

            # 1. Sanitize
            seq = seq.strip().upper()
            clean = []
            replaced = 0
            for aa in seq:
                if aa in "ACDEFGHIKLMNPQRSTVWY":
                    clean.append(aa)
                else:
                    clean.append("X")
                    replaced += 1
            seq = "".join(clean)
            if replaced > 0:
                logger.debug(
                    "Sequence len %d → %d residues replaced with 'X'",
                    len(seq),
                    replaced,
                )

            # 2. Enforce max length
            max_len_model = getattr(model, "max_positions", 1022)
            if len(seq) > max_len_model:
                logger.warning(
                    "Truncating sequence from %d → %d aa (model limit)",
                    len(seq),
                    max_len_model,
                )
                seq = seq[:max_len_model]

            if len(seq) == 0:
                return np.full(model.embed_dim, np.nan, dtype=np.float32)

            # 3. Tokenize
            batch_converter = tokenizer_or_alphabet.get_batch_converter()
            _, _, toks = batch_converter([("protein", seq)])

            # 4. Validate IDs *before* sending to CUDA
            vocab_size = model.embed_tokens.num_embeddings
            max_id = toks.max().item()
            if max_id >= vocab_size:
                bad_idx = (toks >= vocab_size).nonzero(as_tuple=True)[1].tolist()
                bad_chars = [seq[i] for i in bad_idx if i < len(seq)]
                logger.error(
                    "Invalid token IDs for seq len %d (max id %d, vocab %d). "
                    "Chars: %s",
                    len(seq),
                    max_id,
                    vocab_size,
                    bad_chars,
                )
                return np.full(model.embed_dim, np.nan, dtype=np.float32)

            toks = toks.to(device).contiguous()

            # 5. Forward
            out = model(toks, repr_layers=[model.num_layers])
            rep = out["representations"][model.num_layers]
            embedding = rep.mean(1)

        # 6. Convert to numpy
        emb_np = embedding.squeeze(0).detach().cpu().to(torch.float32).numpy()
        return emb_np

    except Exception as e:
        logger.error("Error embedding sequence (%d aa): %s", len(seq), e)
        if return_nan_on_error:
            H = (
                int(getattr(model.config, "hidden_size", 1280))
                if hasattr(model, "config")
                else model.embed_dim
            )
            return np.full(H, np.nan, dtype=np.float32)
        raise


def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        # He initialization (good for ReLU/SiLU activations)
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=1.0)
