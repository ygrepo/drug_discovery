import sys
from pathlib import Path
import numpy as np
from typing import Union
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
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
):
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

    return meta_df

    # Example usage
    # df = load_embeddings_pair("my_embeddings_file")


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


import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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
