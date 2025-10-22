# Import the calculate_metrics function
import sys
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from typing import Sequence, Union, Optional
from scipy.stats import pearsonr


# import numpy as np
# import pandas as pd
# from typing import Sequence, Union, Optional

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


def metrics_per_category(
    df: pd.DataFrame,
    category_col: Union[str, Sequence[str]],
    y_col: str = "Affinity",
    yhat_col: str = "pred_affinity",
    by: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
    min_n: int = 20,
    add_combined_label: bool = False,  # <- default: don't add the trailing "category"
    combined_sep: str = " | ",
) -> pd.DataFrame:
    """Calculate metrics per category (single column or list of columns)."""

    # --- setup / normalize inputs ---
    dfe = df.copy()
    logger.info(f"Starting with: {dfe.shape}")

    if isinstance(category_col, str):
        cat_cols = [category_col]
    else:
        cat_cols = list(category_col)

    group_cols = ([] if by is None else list(by)) + cat_cols

    # --- required columns check ---
    required_cols = [y_col, yhat_col] + group_cols
    missing_cols = [col for col in required_cols if col not in dfe.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # --- clean category columns ---
    for c in cat_cols:
        logger.info(
            f"Unique values in {c} before cleaning: {dfe[c].nunique(dropna=False)}"
        )
        dfe[c] = dfe[c].astype(str).str.strip()
        # mark empty-like strings as NaN, then drop
        dfe.loc[dfe[c].str.lower().isin({"none", "nan", "null", ""}), c] = np.nan

    before = len(dfe)
    dfe = dfe.dropna(subset=cat_cols)
    logger.info(
        f"Dropped {before - len(dfe)} rows with empty/NaN categories; now {dfe.shape}"
    )

    if len(dfe) == 0:
        logger.info("No valid categories found")
        return pd.DataFrame()

    # --- clean numeric columns ---
    logger.info(f"Cleaning numeric columns: {y_col}, {yhat_col}")
    dfe[y_col] = pd.to_numeric(dfe[y_col], errors="coerce")
    dfe[yhat_col] = pd.to_numeric(dfe[yhat_col], errors="coerce")
    dfe = dfe.dropna(subset=[y_col, yhat_col])
    dfe = dfe[np.isfinite(dfe[y_col]) & np.isfinite(dfe[yhat_col])]
    logger.info(f"After numeric cleaning: {dfe.shape}")

    if len(dfe) == 0:
        logger.info("No valid data after cleaning")
        return pd.DataFrame()

    # --- top_k on full combination ---
    if top_k is not None:
        # frequency of combos in group_cols
        vc = dfe.groupby(group_cols, dropna=False).size().sort_values(ascending=False)
        if len(vc) > 0:
            k = min(top_k, len(vc))
            keep_idx = vc.iloc[:k].index
            key = pd.MultiIndex.from_frame(dfe[group_cols])
            dfe = dfe[key.isin(keep_idx)]
            logger.info(f"After top_{k} filtering (by combo): {dfe.shape}")

    # --- min_n on full combination of cat_cols (not including 'by') ---
    if min_n > 0:
        counts = dfe.groupby(group_cols, dropna=False).size()
        valid_keys = counts[counts >= min_n].index
        logger.info(f"Groups meeting min_n={min_n}: {len(valid_keys)}")
        if len(valid_keys) == 0:
            logger.info("No data left after min_n filtering")
            return pd.DataFrame()
        key = pd.MultiIndex.from_frame(dfe[group_cols])
        dfe = dfe[key.isin(valid_keys)]
        logger.info(f"After min_n filtering: {dfe.shape}")

    if len(dfe) == 0:
        logger.info("No data left after filtering")
        return pd.DataFrame()

    # --- group and compute metrics ---
    logger.info(f"Grouping by: {group_cols}")
    group_sizes = dfe.groupby(group_cols, dropna=False).size()
    small_groups = group_sizes[group_sizes < 2]
    if len(small_groups) > 0:
        logger.info(f"Warning: {len(small_groups)} groups have <2 observations")

    try:
        result = (
            dfe.groupby(group_cols, dropna=False)
            .apply(lambda g: _one_group(g, y_col, yhat_col), include_groups=False)
            .reset_index()
        )

        if add_combined_label:
            result["category"] = (
                result[cat_cols].astype(str).agg(combined_sep.join, axis=1)
            )

        # sort results
        sort_cols = ["n", "rmse"] if "rmse" in result.columns else ["n"]
        sort_asc = [False, True][: len(sort_cols)]
        result = result.sort_values(sort_cols, ascending=sort_asc).reset_index(
            drop=True
        )

        logger.info(f"Final result shape: {result.shape}")
        return result

    except Exception as e:
        logger.info(f"Error in groupby operation: {e}")
        return pd.DataFrame()


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
        "--top_k",
        type=int,
        default=None,
        help="Show only top K results per category (None = all results)",
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix for output filenames"
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
        logger.info(f"Top K results per category: {args.top_k}")
        logger.info(f"Prefix: {args.prefix}")
        logger.info(f"Limit to N rows: {args.N}")
        data_fn = Path(args.data_fn).resolve()
        logger.info(f"Data fn: {data_fn}")

        # Load data
        df = read_csv_parquet_torch(data_fn)
        logger.info(f"Loaded {len(df)} samples")
        logger.info(df["Split mode"].unique())
        if args.N > 0:
            df = df.head(n=args.N)
            logger.info(f"Limited to {len(df)} samples")
        gene_fn = Path(args.gene_fn).resolve()
        logger.info(f"Gene fn: {gene_fn}")
        gene_df = read_csv_parquet_torch(gene_fn)
        logger.info(f"Loaded {len(gene_df)} gene df")
        logger.info(
            f"Before HomoSapines filter. Unique genes: {gene_df['Gene'].nunique()}"
        )
        mask = gene_df["Organism"].notna() & gene_df["Organism"].str.contains(
            "Homo sapiens", regex=False
        )
        gene_df = gene_df[mask]
        logger.info(f"HomoSapines filter. Unique genes: {gene_df['Gene'].nunique()}")
        protein_seq_gene_df = gene_df[
            ["Sequence", "Gene", "Association_Type", "Disease_Name"]
        ]
        logger.info(
            f"After HomoSapines filter. Unique proteins: {gene_df['Sequence'].nunique()}"
        )
        s_seq = sanitize(protein_seq_gene_df["Sequence"], to_lower=True, strip=True)
        s_target = sanitize(df["Target"], to_lower=True, strip=True)
        logger.info(f"Common sequences: {len(np.intersect1d(s_seq, s_target))}")
        df = df.merge(
            protein_seq_gene_df, how="inner", left_on="Target", right_on="Sequence"
        )
        logger.info(f"Unique genes: {df['Gene'].nunique()}")
        logger.info(f"Unique proteins: {df['Target'].nunique()}")
        logger.info(
            f"dtypes: {df.dtypes.get(['Affinity', 'pred_affinity'], pd.Series(dtype=object)).to_dict()}"
        )

        # pick one of the rows shown in your CSV (e.g., Embedding==ESM2, Gene==BRAF, Mutant==Mutant)
        probe = df.query("Embedding=='ESM2' and Gene=='BRAF' and Mutant=='Mutant'")[
            ["Affinity", "pred_affinity"]
        ]

        logger.info(f"head:\n {probe.head()}")
        logger.info(f"na counts: {probe.isna().sum().to_dict()}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        core_vatiables = ["Gene", "Mutant"]
        # Mutant
        res = metrics_per_category(
            df,
            core_vatiables,
            # "Mutant",
            by=["Embedding"],
            top_k=args.top_k,
            min_n=args.min_n,
        )
        save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_mutant.csv")

        core_vatiables_2 = ["Gene", "Mutant", "Disease_Name", "Association_Type"]
        # Mutant
        res = metrics_per_category(
            df,
            core_vatiables_2,
            # "Mutant",
            by=["Embedding"],
            top_k=args.top_k,
            min_n=args.min_n,
        )
        save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_mutant_2.csv")

        # Target class
        res = metrics_per_category(
            df,
            core_vatiables + ["Target Class"],
            by=["Embedding"],
            top_k=args.top_k,
            min_n=args.min_n,
        )

        save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_target_class.csv")

        res = metrics_per_category(
            df,
            core_vatiables_2 + ["Target Class"],
            by=["Embedding"],
            top_k=args.top_k,
            min_n=args.min_n,
        )

        save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_target_class_2.csv")

        # Target class
        # res = metrics_per_category(
        #     df,
        #     "Target Class",
        #     by=["Embedding"],
        #     top_k=args.top_k,
        #     min_n=args.min_n,
        # )

        # save_csv_parquet_torch(
        #     res, output_dir / f"{args.prefix}_by_target_class_embeddings.csv"
        # )

        # # Target class
        # res = metrics_per_category(
        #     df,
        #     "Target Class",
        #     by=["Split mode", "Embedding"],
        #     top_k=args.top_k,
        #     min_n=args.min_n,
        # )

        # save_csv_parquet_torch(
        #     res, output_dir / f"{args.prefix}_by_target_class_split_mode.csv"
        # )

        # # Finer taxonomies
        # res = metrics_per_category(
        #     df,
        #     "Target_Class_Level_1",
        #     by=["Embedding"],
        #     top_k=args.top_k,
        #     min_n=args.min_n,
        # )
        # save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_class_level_1.csv")

        # # Target class level 2
        # res = metrics_per_category(
        #     df,
        #     "Target_Class_Level_2",
        #     by=["Embedding"],
        #     top_k=args.top_k,
        #     min_n=args.min_n,
        # )
        # save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_class_level_2.csv")

        # # Target class level 3
        # res = metrics_per_category(
        #     df,
        #     "Target_Class_Level_3",
        #     by=["Embedding"],
        #     top_k=args.top_k,
        #     min_n=args.min_n,
        # )
        # save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_class_level_3.csv")

        # # Target class level 4
        # res = metrics_per_category(
        #     df,
        #     "Target_Class_Level_4",
        #     by=["Embedding"],
        #     top_k=args.top_k,
        #     min_n=args.min_n,
        # )
        # save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_class_level_4.csv")

        # # Target class level 5
        # res = metrics_per_category(
        #     df,
        #     "Target_Class_Level_5",
        #     by=["Embedding"],
        #     top_k=args.top_k,
        #     min_n=args.min_n,
        # )
        # save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_class_level_5.csv")

        # # Target class level 6
        # res = metrics_per_category(
        #     df,
        #     "Target_Class_Level_6",
        #     by=["Embedding"],
        #     top_k=args.top_k,
        #     min_n=args.min_n,
        # )
        # save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_class_level_6.csv")

        # # EC hierarchy (full)
        # df = normalize_ec(df, ec_col="EC number")

        # res = metrics_per_category(
        #     df,
        #     "EC_top",
        #     by=["Embedding"],
        #     top_k=args.top_k,
        #     min_n=args.min_n,
        # )
        # save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_ec_embedding.csv")

        # res = metrics_per_category(
        #     df,
        #     "EC_top",
        #     by=["Split mode", "Embedding"],
        #     top_k=args.top_k,
        #     min_n=args.min_n,
        # )
        # save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_ec_split_mode.csv")

        # # FDA approved vs not
        # res = metrics_per_category(
        #     df,
        #     "FDA Approved",
        #     by=["Embedding"],
        #     min_n=30,
        # )
        # save_csv_parquet_torch(res, output_dir / f"{args.prefix}_by_fda_approved.csv")

    except Exception as e:
        logger.exception("Analysis failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
