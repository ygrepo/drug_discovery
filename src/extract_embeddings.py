import sys
import os
import logging
from pathlib import Path
import pandas as pd
import argparse
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_util import retrieve_embeddings, ModelType, load_model_factory
from src.utils import setup_logging


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
        "--model_type",
        type=str,
        default="",
        help="Model type",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "mutaplm_inference.yaml"),
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


def load_binding_data(data_fn: Path, n: int, seed: int) -> pd.DataFrame:
    """Load the dataset."""
    df = torch.load(data_fn, weights_only=False)
    logger.info(f"Loaded dataset: {len(df)} rows")
    if n > 0:
        logger.info(f"Sampling {n} rows")
        df = df.sample(n=n, random_state=seed)
    df = df[["Target_ID", "Target"]].drop_duplicates()
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
        logger.info(f"Config: {args.config}")
        logger.info(f"Data fn: {args.data_fn}")
        logger.info(f"Output fn: {args.output_fn}")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Number of rows: {args.n}")
        logger.info(f"  Random seed: {args.seed}")

        logger.info("Loading model...")
        logger.info(f"Model type: {args.model_type}")
        mt = ModelType.from_str(args.model_type)
        model, tokenizer = load_model_factory(mt, config_path=Path(args.config))
        logger.info("Model loaded successfully.")

        # Load data
        if "bind" in args.data_fn:
            df = load_binding_data(Path(args.data_fn), args.n, args.seed)
        else:
            df = load_data(Path(args.data_fn), args.n, args.seed)

        logger.info("Extracting embeddings...")
        retrieve_embeddings(
            model_type=mt,
            model=model,
            df=df,
            seq_col="Target",
            tokenizer=tokenizer,
            output_fn=Path(args.output_fn),
        )

    except Exception as e:
        logger.exception("Script failed", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
