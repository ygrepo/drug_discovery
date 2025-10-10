#!/usr/bin/env python3
"""
add_ML_benchmark_predictions.py - Process predictions from ML benchmark."""
from __future__ import annotations
import sys
import argparse
import os
from pathlib import Path
from typing import Iterable
import pandas as pd
from typing import List


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Find files matching a pattern inside a folder."
    )
    ap.add_argument("--log_fn", type=str, default="")
    ap.add_argument("--log_level", type=str, default="INFO")
    ap.add_argument("--dataset", type=str, default="")
    ap.add_argument("--splitmode", type=str, default="")
    ap.add_argument("--embedding", type=str, default="")
    ap.add_argument("--model_dir", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="")
    ap.add_argument("--data_dir", type=str, default="")

    ap.add_argument(
        "-f",
        "--folder",
        type=Path,
        default=Path("output/models"),
        help="Root folder to search (default: current directory)",
    )
    ap.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*predictions.csv",
        help="Pattern (glob by default). Use --regex to treat as regex.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)

    try:
        # Log configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Logging to: {args.log_fn}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Split mode: {args.splitmode}")
        logger.info(f"Embedding: {args.embedding}")
        logger.info(f"Model dir: {args.model_dir}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"Data dir: {args.data_dir}")
        data_dir = Path(args.data_dir)
        data_dir = data_dir / f"{args.embedding}_{args.dataset}_{args.splitmode}"
        logger.info(f"Data dir: {data_dir}")
        df = pd.read_parquet(data_dir / "test.parquet")
        logger.info(f"Loaded dataset: {len(df)} test")

        model_dir = Path(args.model_dir)
        files = sorted(iter_files_glob(model_dir, args.pattern))
        if not files:
            logger.error(f"No files matched pattern '{args.pattern}' under {model_dir}")
            return 1
        logger.info(f"Found {len(files)} prediction file(s).")

        frames: List[pd.DataFrame] = []
        for p in files:
            fn = str(p.resolve())
            model_name = fn.split("_")[0]
            # model_name = infer_model_name(p)
            logger.info(f"Processing: {fn}  |  model_name='{model_name}'")

            pred_df = read_csv_parquet_torch(fn)
            logger.info(f"  raw predictions rows: {len(pred_df):,}")

            # Merge to keep a consistent row set, keyed by Drug + Target Name
            # (adjust keys if your data uses different columns)
            merged = df.merge(pred_df, on=["Drug", "Target Name"], how="inner")
            merged["model_name"] = model_name
            logger.info(f"  merged rows: {len(merged):,}")
            frames.append(merged)

        # Concatenate all models into one DataFrame
        all_df = pd.concat(frames, ignore_index=True)
        logger.info(f"Combined rows total: {len(all_df):,}")

        # Write output
        out_base = f"combined_predictions_{args.embedding}_{args.dataset}_{args.splitmode}.parquet"
        out_path = model_dir / out_base

        save_csv_parquet_torch(all_df, out_path)

        return 0

    except Exception as e:
        logger.exception("Script failed: %s", e)  # or this
        sys.exit(1)


if __name__ == "__main__":
    main()
