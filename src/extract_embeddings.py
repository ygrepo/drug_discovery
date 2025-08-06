import sys
import os
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

import torch
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_util import load_model, load_tokenizer, embed_sequence_sliding
from src.utils import setup_logging, cosine_similarity


def embed_sequence(tokenizer, model, seq):
    """Embed a protein sequence using mean-pooled ESM embeddings."""
    tokens = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens).last_hidden_state
    return outputs.mean(dim=1).squeeze().numpy()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from protein sequences"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=15,
        help="Number of rows to process (for testing)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="ESM model name",
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to the output dataset file",
    )
    parser.add_argument(
        "--log_fn",
        type=str,
        default="logs",
        help="Path to save log file",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g., 'INFO', 'DEBUG')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_data(data_fn: Path, n: int, seed: int) -> pd.DataFrame:
    """Load the dataset."""
    df = pd.read_csv(
        Path(data_fn),
        low_memory=False,
    )
    df.drop(columns=["Unnamed: 0"], inplace=True)
    # Drop missing sequences
    df = df.dropna(subset=["protein1", "protein2"])
    logger.info(f"Loaded dataset: {len(df)} rows")
    if n > 0:
        logger.info(f"Sampling {n} rows")
        df = df.sample(n=n, random_state=seed)
    logger.info(f"Loaded dataset: {len(df)} rows")
    return df


def main():
    # Parse command line arguments
    args = parse_args()

    # Convert paths to absolute paths relative to project root
    logger = setup_logging(Path(args.log_fn), args.log_level)

    try:
        # Log configuration
        logger.info("Starting training with configuration:")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"  Data fn: {args.data_fn}")
        logger.info(f"  Output fn: {args.output_fn}")
        logger.info(f"  Model name: {args.model_name}")
        logger.info(f"  Log file: {args.log_fn}")
        logger.info(f"  Log level: {args.log_level}")
        logger.info(f"  Random seed: {args.seed}")

        logger.info("Extracting embeddings...")

        # Load HF model
        model_name = args.model_name
        model = load_model(model_name)
        tokenizer = load_tokenizer(model_name)
        logger.info(f"Loaded tokenizer: {model_name}")
        logger.info(f"Loaded model: {model_name}")

        # Load data
        df = load_data(Path(args.data_fn), args.n, args.seed)

        # Embed all protein1 and protein2 sequences
        protein1_embeddings = []
        protein2_embeddings = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding pairs"):
            try:
                emb1 = embed_sequence_sliding(
                    tokenizer, model, row["protein1"], logger=logger
                )
                emb2 = embed_sequence_sliding(
                    tokenizer, model, row["protein2"], logger=logger
                )
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

        logger.info(f"{df.head()}")
        df.to_csv(
            Path(args.output_fn),
            index=False,
        )

        logger.info(f"Saved embeddings to {args.output_fn}")

    except Exception as e:
        logger.exception("Training failed", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
