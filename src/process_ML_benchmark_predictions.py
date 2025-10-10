#!/usr/bin/env python3
"""
process_ML_benchmark_predictions.py - Process predictions from ML benchmark."""
from __future__ import annotations
import sys
import argparse
import re
import os
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import setup_logging, get_logger
from src.data_util import (
    load_data,
    DTIDataset,
    append_predictions_csv,
)
from src.ML_benchmark_util import evaluate_model_with_loaders, save_model

logger = get_logger(__name__)


def iter_files_glob(root: Path, pattern: str, follow_symlinks: bool) -> Iterable[Path]:
    # pathlib handles ** recursion; match files only
    yield from (
        p
        for p in root.rglob(pattern)
        if p.is_file() or (follow_symlinks and p.is_symlink())
    )


def iter_files_regex(
    root: Path, pattern: str, flags: int, follow_symlinks: bool
) -> Iterable[Path]:
    rx = re.compile(pattern, flags=flags)
    # Walk everything and filter by regex against the filename (not full path)
    for p in root.rglob("*"):
        if p.is_file() or (follow_symlinks and p.is_symlink()):
            if rx.search(p.name):
                yield p


def within_max_depth(root: Path, path: Path, max_depth: int | None) -> bool:
    if max_depth is None:
        return True
    # Depth relative to root (files directly under root have depth 0)
    rel = path.relative_to(root)
    return len(rel.parts) - 1 <= max_depth


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
    ap.add_argument("--regex", action="store_true", help="Treat pattern as regex")
    ap.add_argument(
        "--ignore-case", action="store_true", help="Case-insensitive (regex only)"
    )
    ap.add_argument("--absolute", action="store_true", help="Print absolute paths")
    ap.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Include files reached via symlinks",
    )
    ap.add_argument("--max-depth", type=int, default=None, help="Limit search depth")
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

        root = args.folder.resolve()
        if not root.exists() or not root.is_dir():
            print(f"Error: folder not found or not a directory: {root}")
            return 2

        if args.regex:
            flags = re.IGNORECASE if args.ignore_case else 0
            candidates = iter_files_regex(
                root, args.pattern, flags, follow_symlinks=args.follow_symlinks
            )
        else:
            candidates = iter_files_glob(
                root, args.pattern, follow_symlinks=args.follow_symlinks
            )

        found_any = False
        for p in candidates:
            if not within_max_depth(root, p, args.max_depth):
                continue
            print(str(p.resolve() if args.absolute else p.relative_to(root)))
            found_any = True
    except Exception as e:
        logger.exception("Script failed: %s", e)  # or this
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
