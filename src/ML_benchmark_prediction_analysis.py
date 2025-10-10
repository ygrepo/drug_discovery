# Import the calculate_metrics function
import sys
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import (
    setup_logging,
    get_logger,
    read_csv_parquet_torch,
    save_csv_parquet_torch,
)
from src.ML_benchmark_util import (
    calculate_metrics,
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
    y_true = g[y_col].astype(float).to_numpy()
    y_pred = g[yhat_col].astype(float).to_numpy()
    rmse, mae, mse, r2, pearson_corr, median_ae, explained_variance = calculate_metrics(
        y_true, y_pred
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


def metrics_per_category(
    df: pd.DataFrame,
    category_col: str,
    y_col: str = "Affinity",
    yhat_col: str = "affinity_pred",
    by: (
        list[str] | None
    ) = None,  # e.g., ["Dataset","Split mode","Embedding","model_name"]
    top_k: int | None = None,  # keep only K biggest categories
    min_n: int = 20,  # drop tiny categories
) -> pd.DataFrame:
    """
    Compute regression metrics per category (and optional facets) using `calculate_metrics`.
    Returns a tidy DataFrame: one row per (facet..., category).
    """
    dfe = df.copy()

    # basic cleaning
    dfe = dfe.dropna(subset=[y_col, yhat_col, category_col])
    dfe = dfe[
        np.isfinite(dfe[y_col].astype(float)) & np.isfinite(dfe[yhat_col].astype(float))
    ]

    # restrict to top-K most frequent categories (by count)
    if top_k is not None:
        top = dfe[category_col].value_counts().nlargest(top_k).index
        dfe = dfe[dfe[category_col].isin(top)]

    # enforce minimum group size
    counts = dfe[category_col].value_counts()
    keep = counts[counts >= min_n].index
    dfe = dfe[dfe[category_col].isin(keep)]

    group_cols = ([] if by is None else list(by)) + [category_col]

    out = (
        dfe.groupby(group_cols, dropna=False, sort=False)
        .apply(lambda g: _one_group(g, y_col, yhat_col))
        .reset_index()
        .rename(columns={category_col: "category"})
        .sort_values(["n", "rmse"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return out


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze ML benchmark predictions by various categories"
    )
    parser.add_argument(
        "--data_fn",
        type=Path,
        default=Path("output/data/combined_predictions_BindingDB.parquet"),
        required=True,
        help="Path to the combined predictions parquet/CSV file",
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
        "--top_k",
        type=int,
        default=None,
        help="Show only top K results per category (None = all results)",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix for output filenames"
    )
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
        logger.info(f"Top K results per category: {args.top_k}")
        logger.info(f"Prefix: {args.prefix}")
        data_fn = Path(args.data_fn).resolve()
        logger.info(f"Data fn: {data_fn}")

        # Load data
        df = read_csv_parquet_torch(data_fn)
        logger.info(f"Loaded {len(df)} samples")

        # Target class
        res = metrics_per_category(
            df,
            "Target Class",
            by=["Dataset", "Split mode", "Embedding", "model_name"],
            top_k=args.top_k,
            min_n=args.min_n,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_csv_parquet_torch(res, output_dir / f"{args.prefix}_metrics_by_class.csv")

        # Finer taxonomies
        res = metrics_per_category(
            df,
            "Target_Class_Level_1",
            by=["Dataset", "Split mode", "Embedding", "model_name"],
            top_k=args.top_k,
            min_n=args.min_n,
        )
        save_csv_parquet_torch(
            res, output_dir / f"{args.prefix}_metrics_by_class_level_1.csv"
        )

        # Target class level 2
        res = metrics_per_category(
            df,
            "Target_Class_Level_2",
            by=["Dataset", "Split mode", "Embedding", "model_name"],
            top_k=args.top_k,
            min_n=args.min_n,
        )
        save_csv_parquet_torch(
            res, output_dir / f"{args.prefix}_metrics_by_class_level_2.csv"
        )

        # Target class level 3
        res = metrics_per_category(
            df,
            "Target_Class_Level_3",
            by=["Dataset", "Split mode", "Embedding", "model_name"],
            top_k=args.top_k,
            min_n=args.min_n,
        )
        save_csv_parquet_torch(
            res, output_dir / f"{args.prefix}_metrics_by_class_level_3.csv"
        )

        # Target class level 4
        res = metrics_per_category(
            df,
            "Target_Class_Level_4",
            by=["Dataset", "Split mode", "Embedding", "model_name"],
            top_k=args.top_k,
            min_n=args.min_n,
        )
        save_csv_parquet_torch(
            res, output_dir / f"{args.prefix}_metrics_by_class_level_4.csv"
        )

        # Target class level 5
        res = metrics_per_category(
            df,
            "Target_Class_Level_5",
            by=["Dataset", "Split mode", "Embedding", "model_name"],
            top_k=args.top_k,
            min_n=args.min_n,
        )
        save_csv_parquet_torch(
            res, output_dir / f"{args.prefix}_metrics_by_class_level_5.csv"
        )

        # Target class level 6
        res = metrics_per_category(
            df,
            "Target_Class_Level_6",
            by=["Dataset", "Split mode", "Embedding", "model_name"],
            top_k=args.top_k,
            min_n=args.min_n,
        )
        save_csv_parquet_torch(
            res, output_dir / f"{args.prefix}_metrics_by_class_level_6.csv"
        )

        # EC hierarchy (full)
        df = normalize_ec(df, ec_col="EC number")

        res = metrics_per_category(
            df,
            "EC_top",
            by=["Dataset", "Split mode", "Embedding", "model_name"],
            top_k=args.top_k,
            min_n=args.min_n,
        )
        save_csv_parquet_torch(res, output_dir / f"{args.prefix}_metrics_by_ec.csv")

        # FDA approved vs not
        res = metrics_per_category(
            df,
            "FDA Approved",
            by=["Dataset", "Split mode", "Embedding", "model_name"],
            min_n=30,
        )
        save_csv_parquet_torch(
            res, output_dir / f"{args.prefix}_metrics_by_fda_approved.csv"
        )

    except Exception as e:
        logger.exception("Analysis failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
