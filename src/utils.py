import sys
from pathlib import Path
import numpy as np
from typing import Union, Dict, Tuple, Optional
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.preprocessing import StandardScaler

import logging

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_file: Path to save log file
        log_level: Logging level (e.g., 'INFO', 'DEBUG')

    Returns:
        Configured logger instance
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def cosine_similarity(
    a: Union[np.ndarray, list[float]],
    b: Union[np.ndarray, list[float]],
    eps: float = 1e-8,
) -> float:
    """
    Compute cosine similarity between two 1D vectors.

    Args:
        a: First vector (1D array or list of floats)
        b: Second vector (1D array or list of floats)

    Returns:
        Cosine similarity as a float in [-1, 1]
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(
            "One of the vectors has zero norm, cannot compute cosine similarity."
        )

    return float(np.dot(a, b) / (norm_a * norm_b))


def load_embeddings_pair(
    base_path: str | Path,
    *,
    seq1_col: str = "protein1",
    seq2_col: str = "protein2",
    return_arrays: bool = False,
    log_level: str = "INFO",
):
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    base_path = Path(base_path)

    # Read metadata
    meta_df = pd.read_csv(base_path.with_suffix(".csv"))

    # Read embeddings
    npz = np.load(base_path.with_suffix(".npz"))
    p1_vecs = npz[seq1_col]
    p2_vecs = npz[seq2_col]
    cosine = npz["cosine"]

    # Recombine into a single DataFrame
    meta_df[f"{seq1_col}_embedding"] = list(p1_vecs)
    meta_df[f"{seq2_col}_embedding"] = list(p2_vecs)
    meta_df["cosine_similarity"] = cosine

    if return_arrays:
        logger.info("Returning embeddings as arrays")
        return meta_df, p1_vecs, p2_vecs
    logger.info("Returning embeddings as DataFrame")
    return meta_df


def UMAP_reduce(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    if np.isnan(embeddings).any():
        raise ValueError("[UMAP_reduce] Input embeddings still contain NaN values.")
    umap = UMAP(n_components=2, random_state=random_state)
    umap_coords = umap.fit_transform(embeddings)
    return umap_coords


def tSNE_reduce(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    if np.isnan(embeddings).any():
        raise ValueError("[tSNE_reduce] Input embeddings still contain NaN values.")
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_coords = tsne.fit_transform(embeddings)
    return tsne_coords


def stack_embeddings(
    df: pd.DataFrame,
    *,
    seq1_col: str = "protein1_embedding",
    seq2_col: str = "protein2_embedding",
    log_level: str = "INFO",
) -> tuple[np.ndarray, int]:
    """
    Extract and stack embeddings from a dataframe (wild-type first, then mutant).
    Removes rows with NaNs and logs how many were found/removed.

    Returns
    -------
    all_embeddings : np.ndarray
        Shape (2*n_clean, d). Wild-type stacked first, then mutant.
    n : int
        Number of clean wild-type embeddings (so mutant starts at n).
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    protein1_embeddings = []
    protein2_embeddings = []
    dropped = 0

    for _, row in df.iterrows():
        try:
            emb1 = np.array(row[seq1_col], dtype=np.float32)
            emb2 = np.array(row[seq2_col], dtype=np.float32)

            if np.isnan(emb1).any() or np.isnan(emb2).any():
                dropped += 1
                continue

            protein1_embeddings.append(emb1)
            protein2_embeddings.append(emb2)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            dropped += 1

    n = len(protein1_embeddings)
    all_embeddings = np.vstack([protein1_embeddings, protein2_embeddings])

    logger.info(
        f"[stack_embeddings] Stacked {n} pairs of embeddings "
        f"(dropped {dropped} rows). Final shape: {all_embeddings.shape}"
    )

    return all_embeddings, n


def _pad_or_truncate(
    X: np.ndarray, target_width: int, pad_value: float = 0.0, side: str = "right"
) -> np.ndarray:
    """
    Ensure X has shape (N, target_width), padding with pad_value or truncating columns if needed.
    side='right' pads/truncates at the end (columns >= target_width).
    """
    N, D = X.shape
    if D == target_width:
        return X
    if D > target_width:
        # truncate columns
        if side == "right":
            return X[:, :target_width]
        else:
            # left truncation (rarely needed)
            return X[:, D - target_width :]
    # pad columns
    pad_cols = target_width - D
    if side == "right":
        pad_block = np.full((N, pad_cols), pad_value, dtype=X.dtype)
        return np.concatenate([X, pad_block], axis=1)
    else:
        pad_block = np.full((N, pad_cols), pad_value, dtype=X.dtype)
        return np.concatenate([pad_block, X], axis=1)


def prepare_shared_umap(
    embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    target_width: Optional[int] = None,  # if None -> max width across models
    pad_value: float = 0.0,
    pad_side: str = "right",  # 'right' (default) or 'left'
    standardize: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    return_objects: bool = False,
):
    """
    Put all models into the **same feature space** (equal width) by padding/truncating,
    then fit **one** UMAP on the concatenated WT+Mut across all models.

    Parameters
    ----------
    embeddings : dict model -> (wt, mut)
        wt, mut: arrays of shape (N_i, D_i). D_i may differ across models.
    target_width : int or None
        If None, use max(D_i) across all models. Otherwise, pad/truncate all to this width.
    pad_value : float
        Value used to pad shorter embeddings (0.0 is standard).
    pad_side : 'right' | 'left'
        Where to pad/truncate columns.
    standardize : bool
        If True, z-score *globally* after padding (recommended).
    n_neighbors, min_dist, random_state
        UMAP hyperparameters.

    Returns
    -------
    coords_2d : dict model -> (wt_2d, mut_2d)
    (scaler, reducer) : optional when return_objects=True
    """
    if not embeddings:
        raise ValueError("No embeddings provided.")

    # 0) determine target width
    widths = []
    for m, (wt, mut) in embeddings.items():
        if wt.ndim != 2 or mut.ndim != 2:
            raise ValueError(
                f"[{m}] WT/Mut must be 2D arrays; got {wt.ndim}D and {mut.ndim}D."
            )
        if np.isnan(wt).any() or np.isnan(mut).any():
            raise ValueError(f"[{m}] NaNs detected; clean embeddings first.")
        widths += [wt.shape[1], mut.shape[1]]
        if wt.shape[1] != mut.shape[1]:
            raise ValueError(
                f"[{m}] WT and Mut dims differ: {wt.shape[1]} vs {mut.shape[1]}."
            )

    if target_width is None:
        target_width = max(widths)  # e.g., 1280 if ESM
    target_width = int(target_width)

    # 1) pad/truncate every model’s WT/Mut to target_width
    harmonized = {}
    for m, (wt, mut) in embeddings.items():
        wt_h = _pad_or_truncate(wt, target_width, pad_value=pad_value, side=pad_side)
        mut_h = _pad_or_truncate(mut, target_width, pad_value=pad_value, side=pad_side)
        harmonized[m] = (wt_h, mut_h)

    # 2) concatenate: [m1_wt, m1_mut, m2_wt, m2_mut, ...]
    X_list, segments = [], []
    for m, (wt_h, mut_h) in harmonized.items():
        X_list.extend([wt_h, mut_h])
        segments.append((m, wt_h.shape[0], mut_h.shape[0]))
    X = np.vstack(X_list)  # (sum(2*N_i), target_width)

    # 3) global standardization (shared space)
    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)

    # 4) shared UMAP on all rows
    reducer = UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    )
    X2 = reducer.fit_transform(X)  # (sum(2*N_i), 2)

    # 5) split back per model
    coords_2d = {}
    cur = 0
    for m, n_wt, n_mut in segments:
        wt2 = X2[cur : cur + n_wt]
        mut2 = X2[cur + n_wt : cur + n_wt + n_mut]
        coords_2d[m] = (wt2, mut2)
        cur += n_wt + n_mut

    if return_objects:
        return coords_2d, (scaler, reducer)
    return coords_2d
