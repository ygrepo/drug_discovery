import sys
import os
from pathlib import Path
import pandas as pd
import argparse
import torch
import joblib
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_util import (
    retrieve_embeddings,
    PLM_MODEL,
    load_model_factory,
    ModelType,
)
from src.utils import setup_logging, save_csv_parquet_torch, get_logger, setup_logging

logger = get_logger(__name__)

BINDDB_COLS = [
    "UniProt (SwissProt) Primary ID of Target Chain 1",
    "BindingDB Target Chain Sequence 1",
]

BIND_COLS = ["Target_ID", "Target"]

KM_COLS = ["Sequence", "Sequence"]


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


def is_binding_data(data_fn: Path) -> bool:
    return "bind" in str(data_fn)


def is_BindDB(data_fn: Path) -> bool:
    return "BindingDB" in str(data_fn)


def is_KM_KCAT(data_fn: Path) -> bool:
    return "data_km" in str(data_fn) or "kcat" in str(data_fn)


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

    elif is_binding_data(data_fn):
        df = torch.load(data_fn, weights_only=False)
        if nrows > 0:
            df = df.head(nrows)
        df = df[BIND_COLS].drop_duplicates()

    elif is_KM_KCAT(data_fn):
        df = joblib.load(data_fn)
        if nrows > 0:
            df = df.head(nrows)
        df = df[[KM_COLS[0]]].drop_duplicates()

    else:
        raise ValueError(f"Unknown data format: {data_fn}")
    
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
    if is_binding_data(data_fn):
        return BIND_COLS[1]
    if is_KM(data_fn):
        return KM_COLS[1]
    raise ValueError(f"Unknown data format: {data_fn}")


def get_target_id_col(data_fn: Path) -> str:
    """Get the target ID column name."""
    if is_BindDB(data_fn):
        return BINDDB_COLS[0]
    if is_binding_data(data_fn):
        return BIND_COLS[0]
    if is_KM(data_fn):
        return KM_COLS[0]
    raise ValueError(f"Unknown data format: {data_fn}")


def merge_embeddings(
    df: pd.DataFrame,
    emb_df: pd.DataFrame,
    target_col: str,
    target_id_col: str,
    model_type: ModelType,
) -> pd.DataFrame:
    """Merge the embeddings into the original dataframe."""
    embedding_col_name = f"{target_col}_embedding"

    # Work on a copy and keep only join key + embedding
    emb_df = emb_df.copy()
    cols_to_keep = [
        c for c in emb_df.columns if c in (target_id_col, embedding_col_name)
    ]
    emb_df = emb_df[cols_to_keep]

    emb_df.rename(
        {embedding_col_name: f"{model_type}_embedding"},
        axis=1,
        inplace=True,
    )

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
        logger.info(f"Is BindDB: {is_BindDB_flag}")
        is_binding_flag = is_binding_data(Path(args.data_fn))
        logger.info(f"Is binding data: {is_binding_flag}")
        is_KM_flag = is_KM(Path(args.data_fn))
        logger.info(f"Is KM data: {is_KM_flag}")

        df = load_binding_data(
            Path(args.data_fn), args.n_samples, args.nrows, args.seed
        )

        target_id_col = get_target_id_col(Path(args.data_fn))
        target_col = get_target_col(Path(args.data_fn))
        logger.info(f"Target id col: {target_id_col}-target col: {target_col}")

        for mt in PLM_MODEL:
            logger.info(f"Extracting embeddings for {mt}...")
            model, tokenizer = load_model_factory(mt, config_path=Path(args.config))
            logger.info("Model loaded successfully.")

            # Build a minimal dataframe for embedding extraction
            if target_id_col == target_col:
                df_seq = df[[target_col]].drop_duplicates()
            else:
                df_seq = df[[target_id_col, target_col]].drop_duplicates()

            emb_df = retrieve_embeddings(
                model_type=mt,
                model=model,
                df=df_seq,
                seq_col=target_col,
                tokenizer=tokenizer,
                output_fn=None,
            )

            logger.debug(f"emb_df before column filtering: {emb_df.head()}")

            # Only drop the sequence column if it is NOT the id column
            if target_col != target_id_col and target_col in emb_df.columns:
                logger.info(f"Dropping column {target_col} from emb_df")
                emb_df.drop(columns=[target_col], inplace=True)

            logger.debug(f"Columns in emb_df before merge: {list(emb_df.columns)}")
            logger.debug(f"Columns in df before merge: {list(df.columns)}")

            df = merge_embeddings(df, emb_df, target_col, target_id_col, mt)

            logger.debug(f"Columns in df after merge: {list(df.columns)}")

            embedding_col = f"{mt}_embedding"
            if embedding_col in df.columns:
                missing_count = df[df[embedding_col].isnull()].shape[0]
                logger.info(f"Missing embeddings for {mt}: {missing_count}")
            else:
                available_cols = list(df.columns)
                logger.error(
                    f"Column {embedding_col} not found. Available: {available_cols}"
                )

        timestamp = datetime.now().strftime("%Y%m%d")

        if args.output_fn:
            out_path = Path(args.output_fn)
            parent = out_path.parent
            base = out_path.stem  # removes any extension, safe even if no extension
            output_file = parent / f"{timestamp}_{base}.pt"
        else:
            output_file = Path(f"{timestamp}_embeddings.pt")

        logger.info(f"Saving embeddings to: {output_file}")
        save_csv_parquet_torch(df, output_file)

    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
