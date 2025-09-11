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

from src.model_util import (
    retrieve_embeddings,
    PLM_MODEL,
    load_model_factory,
    ModelType,
)
from src.utils import setup_logging, save_csv_parquet_torch

BINDDB_COLS = [
    "UniProt (SwissProt) Primary ID of Target Chain 1",
    "BindingDB Target Chain Sequence 1",
]

BIND_COLS = ["Target_ID", "Target"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from protein sequences"
    )
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


def is_binding_data(data_fn: Path) -> bool:
    return "bind" in str(data_fn)


def is_BindDB(data_fn: Path) -> bool:
    return "BindingDB" in str(data_fn)


def load_binding_data(
    data_fn: Path, n_samples: int, nrows: int, seed: int
) -> pd.DataFrame:
    """Load the dataset."""
    if is_BindDB(data_fn):
        if nrows > 0:
            df = pd.read_csv(data_fn, sep="\t", usecols=BINDDB_COLS, nrows=nrows)
        else:
            df = pd.read_csv(data_fn, sep="\t", usecols=BINDDB_COLS)
        df.drop_duplicates(inplace=True)
    else:
        df = torch.load(data_fn, weights_only=False)
        df = df[BIND_COLS].drop_duplicates()
    logger.info(f"Loaded dataset: {len(df)} rows")
    if n_samples > 0:
        logger.info(f"Sampling {n_samples} rows")
        df = df.sample(n=n_samples, random_state=seed)
    logger.info(f"Loaded dataset: {len(df)} rows")
    return df


def get_target_col(data_fn: Path) -> str:
    """Get the target column name."""
    if is_BindDB(data_fn):
        return BINDDB_COLS[1]
    else:
        return BIND_COLS[1]


def get_target_id_col(data_fn: Path) -> str:
    """Get the target ID column name."""
    if is_BindDB(data_fn):
        return BINDDB_COLS[0]
    else:
        return BIND_COLS[0]


def merge_embeddings(
    df: pd.DataFrame,
    emb_df: pd.DataFrame,
    target_col: str,
    target_id_col: str,
    model_type: ModelType,
) -> pd.DataFrame:
    """Merge the embeddings into the original dataframe."""
    # The embedding column name is based on the target_col
    embedding_col_name = f"{target_col}_embedding"
    emb_df.rename({embedding_col_name: f"{model_type}_embedding"}, axis=1, inplace=True)
    return df.merge(emb_df, on=[target_id_col], how="left")


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
        logger.info(f"Number of samples: {args.n_samples}")
        logger.info(f"Number of rows: {args.nrows}")
        logger.info(f"Random seed: {args.seed}")

        logger.info("Loading model...")

        # Load data
        is_BindDB_flag = is_BindDB(Path(args.data_fn))
        is_binding_flag = is_binding_data(Path(args.data_fn))
        logger.info(f"Is BindDB: {is_BindDB_flag}")
        logger.info(f"Is binding data: {is_binding_flag}")

        if is_binding_flag:
            df = load_binding_data(
                Path(args.data_fn), args.n_samples, args.nrows, args.seed
            )
        else:
            df = load_data(Path(args.data_fn), args.n_samples, args.seed)

        target_id_col = get_target_id_col(Path(args.data_fn))
        target_col = get_target_col(Path(args.data_fn))
        logger.info(f"Target id col: {target_id_col}-target col: {target_col}")
        # df_out = df[[target_id_col, target_col]].copy()
        for mt in PLM_MODEL:
            logger.info(f"Extracting embeddings for {mt}...")
            model, tokenizer = load_model_factory(mt, config_path=Path(args.config))
            logger.info("Model loaded successfully.")
            logger.info("Extracting embeddings...")
            # Don't pass output_fn to retrieve_embeddings since we save at end
            emb_df = retrieve_embeddings(
                model_type=mt,
                model=model,
                df=df,
                seq_col=target_col,
                tokenizer=tokenizer,
                output_fn=None,
            )
            emb_df.drop(columns=[target_col], inplace=True)
            logger.debug(f"Columns in emb_df before merge: {list(emb_df.columns)}")
            df_out = merge_embeddings(df, emb_df, target_col, target_id_col, mt)
            #          df_out = merge_embeddings(df_out, emb_df, target_col, target_id_col, mt)
            logger.debug(f"Columns in df_out after merge: {list(df_out.columns)}")

            embedding_col = f"{mt}_embedding"
            if embedding_col in df_out.columns:
                missing_count = df_out[df_out[embedding_col].isnull()].shape[0]
                logger.info(f"Missing embeddings for {mt}: {missing_count}")
            else:
                available_cols = list(df_out.columns)
                logger.error(
                    f"Column {embedding_col} not found. Available: {available_cols}"
                )

            # Determine output path - handle the pattern from shell script
            if args.output_fn:
                # If output_fn ends with underscore, append model and extension
                if args.output_fn.endswith("_"):
                    output_file = Path(f"{args.output_fn}{mt}.pt")
                else:
                    # Otherwise treat as base path and add model suffix
                    output_path = Path(args.output_fn)
                    if output_path.suffix:
                        # Has extension, insert model type before extension
                        stem = output_path.stem
                        suffix = output_path.suffix
                        parent = output_path.parent
                        output_file = parent / f"{stem}_{mt}{suffix}"
                    else:
                        # No extension, add model type and .pt
                        output_file = Path(f"{args.output_fn}_{mt}.pt")
            else:
                # Default to current directory
                output_file = Path(f"{mt}_embeddings.pt")

            save_csv_parquet_torch(df_out, output_file)

    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
