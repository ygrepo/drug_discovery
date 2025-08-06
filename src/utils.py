import sys
from pathlib import Path
import numpy as np
from typing import Union

import logging


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
