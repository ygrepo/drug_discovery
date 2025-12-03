#!/usr/bin/env python3
"""
merge_KM_embeddings.py - Merge KM embeddings."""
from __future__ import annotations
import sys
import argparse
import os
from pathlib import Path
from typing import Iterable
import pandas as pd
import torch
import joblib
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import (
    setup_logging,
    get_logger,
    read_csv_parquet_torch,
    save_csv_parquet_torch,
)

logger = get_logger(__name__)


def iter_files_glob(root: Path, pattern: str) -> Iterable[Path]:
    # pathlib handles ** recursion; match files only
    yield from (p for p in root.rglob(pattern))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge KM embeddings")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to process (for testing)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=0,
        help="Number of rows to process (for testing)",
    )
    parser.add_argument(
        "--data_fn",
        type=Path,
        default="",
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--embedding_fn",
        type=Path,
        default="",
        help="Path to the embedding file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="output/data",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to the output dataset file",
    )
    parser.add_argument(
        "--log_fn",
        type=Path,
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


def is_binding_data(data_fn: Path) -> bool:
    return "bind" in str(data_fn)


def is_BindDB(data_fn: Path) -> bool:
    return "BindingDB" in str(data_fn)


def is_KM(data_fn: Path) -> bool:
    return "data_km" in str(data_fn)


def load_binding_data(
    data_fn: Path, n_samples: int, nrows: int, seed: int
) -> pd.DataFrame:
    """Load the dataset."""
    if is_BindDB(data_fn):
        if nrows > 0:
            df = pd.read_csv(data_fn, sep="\t", nrows=nrows)
        else:
            df = pd.read_csv(data_fn, sep="\t")
        df.drop_duplicates(inplace=True)

    if is_binding_data(data_fn):
        df = torch.load(data_fn, weights_only=False)
        if nrows > 0:
            df = df.head(nrows)

    if is_KM(data_fn):
        df = joblib.load(data_fn)
        if nrows > 0:
            df = df.head(nrows)

    logger.info(f"Loaded dataset: {len(df)} rows")
    if n_samples > 0:
        logger.info(f"Sampling {n_samples} rows")
        df = df.sample(n=n_samples, random_state=seed)
    logger.info(f"Loaded dataset: {len(df)} rows")
    return df


def main():
    # Parse command line arguments
    args = parse_args()

    # Convert paths to absolute paths relative to project root
    logger = setup_logging(Path(args.log_fn), args.log_level)

    try:
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Logging to: {args.log_fn}")
        logger.info(f"Data fn: {args.data_fn}")
        logger.info(f"Embedding fn: {args.embedding_fn}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"Output fn: {args.output_fn}")
        logger.info(f"Number of samples: {args.n_samples}")
        logger.info(f"Number of rows: {args.nrows}")
        logger.info(f"Random seed: {args.seed}")

        # Load data
        is_BindDB_flag = is_BindDB(Path(args.data_fn))
        logger.info(f"Is BindDB: {is_BindDB_flag}")
        is_binding_flag = is_binding_data(Path(args.data_fn))
        logger.info(f"Is binding data: {is_binding_flag}")
        is_KM_flag = is_KM(Path(args.data_fn))
        logger.info(f"Is KM data: {is_KM_flag}")
        data_fn = args.data_fn.resolve()
        df = load_binding_data(data_fn, args.n_samples, args.nrows, args.seed)
        embedding_fn = args.embedding_fn.resolve()
        emb_df = read_csv_parquet_torch(embedding_fn)
        logger.info(f"Loaded embedding df: {len(emb_df)} rows")
        df = df.merge(emb_df, on="Sequence", how="left")
        logger.info(f"Merged df: {len(df)} rows")
        timestamp = datetime.now().strftime("%Y%m%d")
        output_dir = args.output_dir.resolve()
        output_file = output_dir / f"{timestamp}_{args.output_fn}.pt"
        logger.info(f"Saving embeddings to: {output_file}")
        save_csv_parquet_torch(df, output_file)

    except Exception as e:
        logger.exception("Script failed: %s", e)  # or this
        raise e


if __name__ == "__main__":
    main()
