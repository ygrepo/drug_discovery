import sys
import os
from pathlib import Path
import pandas as pd
import argparse
import numpy as np
from datetime import datetime
from typing import Optional


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_util import (
    retrieve_embeddings,
    PLM_MODEL,
    load_model_factory,
    ModelType,
)
from src.utils import (
    setup_logging,
    save_csv_parquet_torch,
    get_logger,
    setup_logging,
    norm_smiles,
)

logger = get_logger(__name__)


def smiles_to_morgan_fingerprint(
    smiles: str, radius: int = 2, n_bits: int = 2048
) -> Optional[np.ndarray]:
    """
    Convert SMILES string to Morgan fingerprint.

    Args:
        smiles: SMILES string
        radius: Morgan fingerprint radius (default: 2)
        n_bits: Number of bits in fingerprint (default: 2048)

    Returns:
        numpy array of fingerprint bits, or None if conversion fails
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not parse SMILES: {smiles}")
            return None

        # Generate Morgan fingerprint
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits
        )
        # Convert to numpy array
        return np.array(fp, dtype=np.float32)

    except Exception as e:
        logger.warning(f"Error generating fingerprint for {smiles}: {e}")
        return None


def add_smiles_fingerprints(
    df: pd.DataFrame, smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """
    Add Morgan fingerprints for SMILES column.
    Deduplicates SMILES before processing for efficiency.

    Args:
        df: DataFrame with SMILES column
        smiles_col: Name of SMILES column

    Returns:
        DataFrame with added fingerprint column
    """
    if smiles_col not in df.columns:
        logger.warning(
            f"Column {smiles_col} not found. Skipping fingerprint generation."
        )
        return df

    logger.info(f"Generating Morgan fingerprints for {len(df)} rows...")

    # Normalize SMILES first to handle variations
    df_normalized = df.copy()
    df_normalized[f"{smiles_col}_normalized"] = norm_smiles(df[smiles_col])

    # Get unique normalized SMILES for efficient processing
    unique_smiles = df_normalized[f"{smiles_col}_normalized"].drop_duplicates()
    logger.info(
        f"Found {len(unique_smiles)} unique SMILES to process (after normalization)"
    )

    # Generate fingerprints for unique SMILES only
    smiles_to_fp = {}
    failed_count = 0

    for smiles in unique_smiles:
        fp = smiles_to_morgan_fingerprint(smiles)
        if fp is not None:
            smiles_to_fp[smiles] = fp
        else:
            # Use zero vector for failed conversions
            smiles_to_fp[smiles] = np.zeros(2048, dtype=np.float32)
            failed_count += 1

    # Map fingerprints back to all rows using normalized SMILES
    fingerprints = [
        smiles_to_fp[smiles] for smiles in df_normalized[f"{smiles_col}_normalized"]
    ]

    df = df.copy()
    df[f"{smiles_col}_fingerprint"] = fingerprints

    if failed_count > 0:
        logger.warning(
            f"Failed to generate fingerprints for {failed_count} unique SMILES"
        )

    success_count = len(unique_smiles) - failed_count
    efficiency_ratio = len(df) / len(unique_smiles) if len(unique_smiles) > 0 else 1
    logger.info(
        f"Successfully generated fingerprints for {success_count} unique SMILES"
    )
    logger.info(f"Mapped fingerprints to {len(df)} total rows")
    logger.info(
        f"Efficiency gain: {efficiency_ratio:.1f}x (processed {len(unique_smiles)} instead of {len(df)})"
    )

    return df


def load_data(data_fn: Path, n: int) -> pd.DataFrame:
    """Load the dataset."""
    df = pd.read_csv(data_fn)
    logger.info(f"Loaded dataset: {len(df)} rows")
    if n > 0:
        df = df.head(n)
    df = df[["WA", "Pos", "MA", "Protein", "SMILES", "Y"]]
    logger.info(f"Loaded dataset: {len(df)} rows")
    return df


def merge_embeddings(
    df: pd.DataFrame,
    emb_df: pd.DataFrame,
    model_type: ModelType,
    target_col: str = "Protein",
    target_id_col: str = "Protein",
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from protein sequences"
    )
    parser.add_argument(
        "--n",
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
        "--generate_fingerprints",
        action="store_true",
        help="Generate Morgan fingerprints from SMILES column",
    )

    return parser.parse_args()


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
        logger.info(f"Number of rows: {args.n}")

        logger.info("Loading model...")

        # Load data
        df = load_data(Path(args.data_fn), args.n)

        # Add SMILES fingerprints if requested
        if args.generate_fingerprints:
            df = add_smiles_fingerprints(df, smiles_col="SMILES")

        target_col = "Protein"
        logger.info(f"Target col: {target_col}")

        for mt in PLM_MODEL:
            try:
                logger.info(f"Extracting embeddings for {mt}...")
                model, tokenizer = load_model_factory(mt, config_path=Path(args.config))
                logger.info("Model loaded successfully.")

                # Build a minimal dataframe for embedding extraction
                df_seq = df[[target_col]].drop_duplicates()
                emb_df = retrieve_embeddings(
                    model_type=mt,
                    model=model,
                    df=df_seq,
                    seq_col=target_col,
                    tokenizer=tokenizer,
                    output_fn=None,
                )

                logger.debug(f"emb_df before column filtering: {emb_df.head()}")
                logger.debug(f"Columns in emb_df before merge: {list(emb_df.columns)}")
                logger.debug(f"Columns in df before merge: {list(df.columns)}")

                df = merge_embeddings(df, emb_df, mt)

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
            except Exception as e:
                logger.exception(f"Failed to extract embeddings for {mt}: {e}")

        # Reorganize columns in desired order
        desired_columns = [
            "WA",
            "Pos",
            "MA",
            "Protein",
            "ESMv1_embedding",
            "ESM2_embedding",
            "MUTAPLM_embedding",
            "ProteinCLIP_embedding",
            "SMILES",
            "SMILES_fingerprint",
            "Y",
        ]

        # Only include columns that exist in the DataFrame
        available_columns = [col for col in desired_columns if col in df.columns]

        # Add any remaining columns that weren't in the desired order
        remaining_columns = [col for col in df.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns

        df = df[final_columns]
        logger.info(f"Reorganized columns: {list(df.columns)}")

        timestamp = datetime.now().strftime("%Y%m%d")

        if args.output_fn:
            out_path = Path(args.output_fn)
            parent = out_path.parent
            base = out_path.stem  # removes any extension, safe even if no extension
            output_file = parent / f"{timestamp}_{base}.csv"
        else:
            output_file = Path(f"{timestamp}_embeddings.csv")

        logger.info(f"Final DataFrame shape: {df.shape}")
        logger.info(f"Saving embeddings to: {output_file}")
        save_csv_parquet_torch(df, output_file)

    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
