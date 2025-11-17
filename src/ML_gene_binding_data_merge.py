# Import the calculate_metrics function
import sys
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from typing import Sequence, Union, Optional
from scipy.stats import pearsonr
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

EC_TOP_MAP = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}


def normalize_ec(df: pd.DataFrame, ec_col: str = "EC number") -> pd.DataFrame:
    """
    Normalize EC annotations and add helper columns.

    Adds:
      - EC_top_digit: first EC field ("1".."7") or NaN
      - EC_major:     first two EC fields (e.g., "3.1") or NaN
      - EC_top_class: mapped 7-class name or "Unknown/Unannotated"

    Returns the input DataFrame with these added/overwritten columns.
    """
    if ec_col not in df.columns:
        raise KeyError(f"{ec_col!r} not in DataFrame")

    s = df[ec_col].astype("string")

    # Strip everything except digits, dots, and '-' so things like "EC 3.1.-.-" become "3.1.-.-"
    s_clean = s.str.upper().str.replace(r"[^0-9.\-]", "", regex=True)

    parts = s_clean.str.split(".")
    top_digit = parts.str[0].str.extract(
        r"^([1-7])$", expand=False
    )  # only 1..7 accepted
    # EC_major = "<top>.<second>" if both exist
    second = parts.str[1]
    ec_major = pd.Series(
        np.where(top_digit.notna() & second.notna(), top_digit + "." + second, np.nan),
        index=df.index,
        dtype="string",
    )

    # Map to class name; unknowns -> "Unknown/Unannotated"
    ec_top_class = (
        top_digit.map(EC_TOP_MAP)
        .fillna("Unknown/Unannotated")
        .astype("category")
        .cat.set_categories(
            [
                "Oxidoreductases",
                "Transferases",
                "Hydrolases",
                "Lyases",
                "Isomerases",
                "Ligases",
                "Translocases",
                "Unknown/Unannotated",
            ],
            ordered=False,
        )
    )

    df["EC_top_digit"] = top_digit
    df["EC_major"] = ec_major
    df["EC_top"] = ec_top_class
    return df


def _one_group(g: pd.DataFrame, y_col: str, yhat_col: str) -> pd.Series:
    """Calculate metrics for one group, handling small sample sizes"""
    y_true = g[y_col].astype(float).to_numpy()
    y_pred = g[yhat_col].astype(float).to_numpy()

    # Check if we have enough data points
    if len(y_true) < 2:
        return pd.Series(
            {
                "n": len(g),
                "rmse": np.nan,
                "mae": np.nan,
                "mse": np.nan,
                "r2": np.nan,
                "pearson": np.nan,
                "median_ae": np.nan,
                "explained_variance": np.nan,
            }
        )

    try:
        # Calculate metrics safely
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        median_ae = np.median(np.abs(y_true - y_pred))

        # Handle R2 calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        # Handle correlation calculation
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            pearson_corr, _ = pearsonr(y_true, y_pred)
        else:
            pearson_corr = np.nan

        # Explained variance
        var_y = np.var(y_true)
        explained_variance = (
            1 - np.var(y_true - y_pred) / var_y if var_y != 0 else np.nan
        )

        return pd.Series(
            {
                "n": len(g),
                "rmse": rmse,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "pearson": pearson_corr,
                "median_ae": median_ae,
                "explained_variance": explained_variance,
            }
        )

    except Exception as e:
        logger.info(f"Error calculating metrics for group of size {len(g)}: {e}")
        return pd.Series(
            {
                "n": len(g),
                "rmse": np.nan,
                "mae": np.nan,
                "mse": np.nan,
                "r2": np.nan,
                "pearson": np.nan,
                "median_ae": np.nan,
                "explained_variance": np.nan,
            }
        )


def sanitize(a, *, to_lower=False, strip=True, drop_empty=True):
    """Remove NaN/None, coerce to str (optionally normalize)."""
    out = []
    for x in np.asarray(a, dtype=object):
        if x is None:
            continue
        if isinstance(x, float) and np.isnan(x):
            continue
        # bytes → str
        if isinstance(x, (bytes, bytearray)):
            x = x.decode("utf-8", "ignore")
        # everything else → str
        x = str(x)
        if strip:
            x = x.strip()
        if to_lower:
            x = x.lower()
        if drop_empty and x == "":
            continue
        out.append(x)
    return np.array(out, dtype=object)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze ML benchmark predictions by various categories"
    )
    parser.add_argument(
        "--N",
        type=int,
        default=0,
        help="Limit to N rows",
    )
    parser.add_argument(
        "--data_fn",
        type=Path,
        default=Path("output/data/combined_predictions_BindingDB.parquet"),
        required=True,
        help="Path to the combined predictions parquet/CSV file",
    )
    parser.add_argument(
        "--gene_fn",
        type=Path,
        default=Path(
            "output/data/gtdb_causal_onco_tsg_gene_disease_icd10_protein_class.csv"
        ),
        help="Path to the gene names file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/metrics",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--min_n",
        type=int,
        default=30,
        help="Minimum number of samples per category",
    )
    parser.add_argument(
        "--log_fn", type=str, default="logs/ML_benchmark_prediction_analysis.log"
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)
    try:
        # Log configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Logging to: {args.log_fn}")
        logger.info(f"Data file: {args.data_fn}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"Minimum samples per category: {args.min_n}")
        logger.info(f"Limit to N rows: {args.N}")
        data_fn = Path(args.data_fn).resolve()
        logger.info(f"Data fn: {data_fn}")

        # Load data
        df = read_csv_parquet_torch(data_fn)
        logger.info(f"Shape: {df.shape}")
        logger.info(df["Split mode"].unique())
        if args.N > 0:
            df = df.head(n=args.N)
            logger.info(f"Limited to {len(df)} samples")
        features = [
            "BindingDB Ligand Name",
            "Drug",
            #'Uniprot',
            "Target Name",
            "Target",
            "Target Class",
            "Target_Class_Level_1",
            "Target_Class_Level_2",
            "Target_Class_Level_3",
            "Mutant",
            "Affinity",
            "pred_affinity",
            "Embedding",
            "Split mode",
        ]
        df = df[features]
        df.rename(
            columns={
                "BindingDB Ligand Name": "Drug_Name",
                "Affinity": "True_Affinity",
                "Embedding": "Model",
                "Target Class": "Target_Class",
                "pred_affinity": "Predicted_Affinity",
                "Split mode": "Dataset",
            },
            inplace=True,
        )
        logger.info(f"Shape: {df.shape}")

        gene_fn = Path(args.gene_fn).resolve()
        logger.info(f"Gene fn: {gene_fn}")
        gene_df = read_csv_parquet_torch(gene_fn)
        logger.info(f"Loaded {len(gene_df)} gene df")
        logger.info(
            f"Before HomoSapiens filter. Unique genes: {gene_df['Gene'].nunique()}"
        )
        mask = gene_df["Organism"].notna() & gene_df["Organism"].str.contains(
            "Homo sapiens", regex=False
        )
        gene_df = gene_df[mask]
        logger.info(f"HomoSapiens filter. Unique genes: {gene_df['Gene'].nunique()}")
        protein_seq_gene_df = gene_df[
            ["Gene", "Role", "Sequence", "Association_Type", "Disease_Name"]
        ]
        logger.info(
            f"After HomoSapiens filter. Unique proteins: {gene_df['Sequence'].nunique()}"
        )
        s_seq = sanitize(protein_seq_gene_df["Sequence"], to_lower=True, strip=True)
        s_target = sanitize(df["Target"], to_lower=True, strip=True)
        logger.info(f"Common sequences: {len(np.intersect1d(s_seq, s_target))}")
        features = ["Gene", "Role", "Sequence"]
        protein_seq_gene_df = protein_seq_gene_df[features]
        protein_seq_gene_df.rename(columns={"Sequence": "Target"}, inplace=True)
        logger.info(f"protein_seq_gene_df shape: {protein_seq_gene_df.shape}")

        df = df.merge(protein_seq_gene_df, on="Target", how="left")

        logger.info(f"Unique genes: {df['Gene'].nunique()}")
        logger.info(f"Unique proteins: {df['Target'].nunique()}")
        logger.info(f"Shape:{df.shape}")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        datestamp = datetime.now().strftime("%Y%m%d")
        save_csv_parquet_torch(
            df, output_dir / f"{datestamp}_all_binding_db_genes.parquet"
        )

    except Exception as e:
        logger.exception("Analysis failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
