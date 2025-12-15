import sys
from pathlib import Path
import argparse
import os
import torch
import joblib
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import (
    setup_logging,
    get_logger,
)
from src.data_util import (
    load_data,
)
from src.ML_benchmark_util import evaluate_model_with_loaders_no_smiles, save_model

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


def is_KM_KCAT_KI(data_fn: Path) -> bool:
    return "km" in str(data_fn) or "kcat" in str(data_fn) or "ki" in str(data_fn)


def is_inhouse(data_fn: Path) -> bool:
    return "inhouse" in str(data_fn)


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

    elif is_KM_KCAT_KI(data_fn):
        df = joblib.load(data_fn)
        if nrows > 0:
            df = df.head(nrows)
        df = df[[KM_COLS[0]]].drop_duplicates()

    elif is_inhouse(data_fn):
        df = pd.read_csv(data_fn)
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
    if is_KM_KCAT_KI(data_fn):
        return KM_COLS[1]
    if is_inhouse(data_fn):
        return KM_COLS[1]
    raise ValueError(f"Unknown data format: {data_fn}")


def get_target_id_col(data_fn: Path) -> str:
    """Get the target ID column name."""
    if is_BindDB(data_fn):
        return BINDDB_COLS[0]
    if is_binding_data(data_fn):
        return BIND_COLS[0]
    if is_KM_KCAT_KI(data_fn):
        return KM_COLS[0]
    if is_inhouse(data_fn):
        return KM_COLS[0]
    raise ValueError(f"Unknown data format: {data_fn}")


def main():
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)

    try:
        # Log configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        # Load data
        is_BindDB_flag = is_BindDB(Path(args.data_fn))
        logger.info(f"Is BindDB: {is_BindDB_flag}")
        is_binding_flag = is_binding_data(Path(args.data_fn))
        logger.info(f"Is binding data: {is_binding_flag}")
        is_KM_flag = is_KM_KCAT_KI(Path(args.data_fn))
        logger.info(f"Is KM/Kcat/Ki data: {is_KM_flag}")
        is_inhouse_flag = is_inhouse(Path(args.data_fn))
        logger.info(f"Is inhouse data: {is_inhouse_flag}")

        df = load_binding_data(
            Path(args.data_fn), args.n_samples, args.nrows, args.seed
        )

        target_id_col = get_target_id_col(Path(args.data_fn))
        target_col = get_target_col(Path(args.data_fn))
        logger.info(f"Target id col: {target_id_col}-target col: {target_col}")

    except Exception as e:
        logger.exception("Script failed", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
