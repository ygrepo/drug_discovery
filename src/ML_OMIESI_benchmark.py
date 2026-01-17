# ML_OMIESI_benchmark.py - Binary classification benchmark for OMIESI dataset
import sys
from pathlib import Path
import argparse
import os
import ast
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List

# Scikit-learn imports for classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logging, get_logger
from src.ML_benchmark_util import save_model

logger = get_logger(__name__)
SEED = 42

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA2IDX = {aa: i for i, aa in enumerate(AA20)}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run OMIESI binary classification benchmark"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train.csv, val.csv, test.csv files",
    )
    p.add_argument("--log_fn", type=str, default="logs/omiesi_benchmark.log")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--output_dir", type=str, default="output/data/omiesi_benchmark")
    p.add_argument("--model_dir", type=str, default="output/models/omiesi_benchmark")
    p.add_argument("--prefix", type=str, default=None)

    p.add_argument(
        "--use_fingerprints",
        action="store_true",
        help="Include SMILES_fingerprint column as features (must exist in CSV)",
    )

    # Mutation feature encoding
    p.add_argument(
        "--mutation_encoding",
        type=str,
        default="onehot",
        choices=["onehot", "physchem", "both"],
        help="How to encode WA/Pos/MA features.",
    )
    p.add_argument(
        "--use_pos_norm",
        action="store_true",
        help="Normalize Pos by protein length (requires Protein column). If missing, falls back to raw Pos.",
    )

    # Scaling control
    p.add_argument(
        "--scale_for_linear",
        action="store_true",
        help="Apply StandardScaler to mutation+embedding blocks for LR/SVM/MLP.",
    )

    return p.parse_args()


# -----------------------------
# Robust vector parsing helpers
# -----------------------------
def _safe_parse_vector(x, fallback_dim: int = 0) -> np.ndarray:
    """
    Parse vectors stored as:
      - python lists
      - numpy arrays
      - strings like "[0.1 0.2 ...]" or "[0.1, 0.2, ...]"
      - JSON-like strings
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)

    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return (
                np.zeros((fallback_dim,), dtype=np.float32)
                if fallback_dim > 0
                else np.array([], dtype=np.float32)
            )

        # literal_eval handles python list formatting
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return np.asarray(obj, dtype=np.float32)
        except Exception:
            pass

        # JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return np.asarray(obj, dtype=np.float32)
        except Exception:
            pass

        # fallback: numeric parse
        s2 = s.strip("[]()")
        s2 = s2.replace(",", " ")
        arr = np.fromstring(s2, sep=" ", dtype=np.float32)
        if arr.size > 0:
            return arr

        return (
            np.zeros((fallback_dim,), dtype=np.float32)
            if fallback_dim > 0
            else np.array([], dtype=np.float32)
        )

    return (
        np.zeros((fallback_dim,), dtype=np.float32)
        if fallback_dim > 0
        else np.array([], dtype=np.float32)
    )


def _stack_vector_column(df: pd.DataFrame, col: str, fallback_dim: int) -> np.ndarray:
    vecs = [_safe_parse_vector(v, fallback_dim=fallback_dim) for v in df[col].values]
    dims = [v.size for v in vecs if v.size > 0]
    dim = max(dims) if dims else fallback_dim

    out = np.zeros((len(vecs), dim), dtype=np.float32)
    for i, v in enumerate(vecs):
        if v.size == dim:
            out[i] = v
        elif v.size > 0:
            out[i, : min(dim, v.size)] = v[:dim]
    return out


# -----------------------------
# Data loading
# -----------------------------
def load_omiesi_data(data_path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def detect_embedding_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect embedding columns of the form '*_embedding'.
    Example: ESMv1_embedding, ESM2_embedding, MUTAPLM_embedding, ProteinCLIP_embedding
    """
    emb_cols = [c for c in df.columns if c.endswith("_embedding")]
    emb_cols = sorted(emb_cols)
    logger.info(f"Detected embedding columns: {emb_cols}")
    return emb_cols


# -----------------------------
# Mutation feature encoding
# -----------------------------
def _onehot_aa(series: pd.Series) -> np.ndarray:
    n = len(series)
    mat = np.zeros((n, 20), dtype=np.float32)
    for i, aa in enumerate(series.astype(str).values):
        idx = AA2IDX.get(aa, None)
        if idx is not None:
            mat[i, idx] = 1.0
    return mat


def encode_mutation_features(
    df: pd.DataFrame,
    mutation_encoding: str = "onehot",
    use_pos_norm: bool = False,
) -> np.ndarray:
    if not {"WA", "Pos", "MA"}.issubset(df.columns):
        raise ValueError("Missing one or more required columns: WA, Pos, MA")

    pos = df["Pos"].astype(float).values.reshape(-1, 1).astype(np.float32)

    if use_pos_norm and "Protein" in df.columns:
        prot_len = (
            df["Protein"].astype(str).map(len).values.astype(np.float32).reshape(-1, 1)
        )
        prot_len = np.clip(prot_len, 1.0, None)
        pos_feat = pos / prot_len
        logger.info("Using Pos normalized by protein length")
    else:
        pos_feat = pos
        if use_pos_norm:
            logger.warning(
                "Requested --use_pos_norm but Protein column missing. Using raw Pos."
            )

    wa_oh = _onehot_aa(df["WA"])
    ma_oh = _onehot_aa(df["MA"])
    onehot_block = np.hstack([pos_feat, wa_oh, ma_oh])  # 41 dims

    if mutation_encoding == "onehot":
        return onehot_block

    aa_properties = {
        "A": [89.1, 1.8, 6.0, 0, 0, 0],
        "R": [174.2, -4.5, 10.8, 0, 1, 1],
        "N": [132.1, -3.5, 5.4, 0, 0, 1],
        "D": [133.1, -3.5, 1.9, 0, -1, 1],
        "C": [121.0, 2.5, 10.8, 0, 0, 0],
        "Q": [146.1, -3.5, 5.7, 0, 0, 1],
        "E": [147.1, -3.5, 4.2, 0, -1, 1],
        "G": [75.1, -0.4, 6.0, 0, 0, 0],
        "H": [155.2, -3.2, 7.6, 1, 1, 1],
        "I": [131.2, 4.5, 6.0, 0, 0, 0],
        "L": [131.2, 3.8, 6.0, 0, 0, 0],
        "K": [146.2, -3.9, 9.7, 0, 1, 1],
        "M": [149.2, 1.9, 5.7, 0, 0, 0],
        "F": [165.2, 2.8, 5.5, 1, 0, 0],
        "P": [115.1, -1.6, 6.3, 0, 0, 0],
        "S": [105.1, -0.8, 5.7, 0, 0, 1],
        "T": [119.1, -0.7, 5.6, 0, 0, 1],
        "W": [204.2, -0.9, 5.9, 1, 0, 0],
        "Y": [181.2, -1.3, 5.7, 1, 0, 1],
        "V": [117.1, 4.2, 6.0, 0, 0, 0],
    }

    wa_props = np.vstack(
        [aa_properties.get(a, [0] * 6) for a in df["WA"].astype(str).values]
    ).astype(np.float32)
    ma_props = np.vstack(
        [aa_properties.get(a, [0] * 6) for a in df["MA"].astype(str).values]
    ).astype(np.float32)
    delta_props = (ma_props - wa_props).astype(np.float32)

    physchem_block = np.hstack([pos_feat, wa_props, ma_props, delta_props])  # 19 dims

    if mutation_encoding == "physchem":
        return physchem_block

    # both
    return np.hstack([onehot_block, physchem_block])


# -----------------------------
# Feature preparation
# -----------------------------
def prepare_single_embedding_features(
    df: pd.DataFrame,
    embedding_col: str,
    use_fingerprints: bool = True,
    mutation_encoding: str = "onehot",
    use_pos_norm: bool = False,
    fp_dim_default: int = 2048,
    emb_dim_default: int = 512,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Prepare features for ONE embedding column name (e.g. 'ESM2_embedding').
    """
    if "Y" not in df.columns:
        raise ValueError("Missing required target column: Y")
    if embedding_col not in df.columns:
        raise ValueError(f"Embedding column not found: {embedding_col}")

    mut = encode_mutation_features(
        df, mutation_encoding=mutation_encoding, use_pos_norm=use_pos_norm
    )
    n_mut = mut.shape[1]

    emb = _stack_vector_column(df, embedding_col, fallback_dim=emb_dim_default)
    n_emb = emb.shape[1]

    blocks = {
        "mutation": (0, n_mut),
        "embedding": (n_mut, n_mut + n_emb),
    }

    parts = [mut, emb]

    if use_fingerprints:
        if "SMILES_fingerprint" not in df.columns:
            logger.warning(
                "Requested fingerprints but SMILES_fingerprint column missing. Skipping fingerprints."
            )
        else:
            fp = _stack_vector_column(
                df, "SMILES_fingerprint", fallback_dim=fp_dim_default
            )
            n_fp = fp.shape[1]
            parts.append(fp)
            blocks["fingerprint"] = (n_mut + n_emb, n_mut + n_emb + n_fp)

    X = np.hstack(parts).astype(np.float32)
    y = df["Y"].astype(int).values
    return X, y, blocks


# -----------------------------
# Scaling wrapper (optional)
# -----------------------------
class BlockStandardScaler(BaseEstimator, TransformerMixin):
    """
    StandardScaler applied ONLY to specified blocks (feature slices).
    Keeps binary fingerprints untouched.
    """

    def __init__(self, blocks: List[Tuple[int, int]]):
        self.blocks = blocks
        self.scalers: List[StandardScaler] = []

    def fit(self, X, y=None):
        self.scalers = []
        for a, b in self.blocks:
            sc = StandardScaler()
            sc.fit(X[:, a:b])
            self.scalers.append(sc)
        return self

    def transform(self, X):
        X2 = X.copy()
        for sc, (a, b) in zip(self.scalers, self.blocks):
            X2[:, a:b] = sc.transform(X2[:, a:b])
        return X2


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_classifier(
    model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
) -> dict:
    y_pred = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="binary", zero_division=0),
    }
    if y_pred_proba is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_test, y_pred_proba)
    return metrics


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)

    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    try:
        logger.info("Starting OMIESI binary classification benchmark")
        logger.info(f"Arguments: {vars(args)}")

        output_dir = Path(args.output_dir)
        model_dir = Path(args.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        data_dir = Path(args.data_dir)
        prefix = args.prefix if args.prefix else ""
        train_path = data_dir / f"{prefix}train_embeddings.csv"
        val_path = data_dir / f"{prefix}val_embeddings.csv"
        test_path = data_dir / f"{prefix}test_embeddings.csv"

        for file_path in [train_path, val_path, test_path]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

        train_df = load_omiesi_data(train_path)
        val_df = load_omiesi_data(val_path)
        test_df = load_omiesi_data(test_path)

        # Auto-detect embedding columns from training set
        embedding_cols = detect_embedding_columns(train_df)
        if not embedding_cols:
            raise ValueError(
                "No embedding columns found (expected '*_embedding' columns)."
            )

        # Define base models
        results = []

        for embedding_col in embedding_cols:
            embedding_name = embedding_col.replace("_embedding", "")

            logger.info(f"\n{'='*70}")
            logger.info(f"Embedding: {embedding_col}")
            logger.info(f"{'='*70}")

            # Prepare features
            try:
                X_train, y_train, blocks_train = prepare_single_embedding_features(
                    train_df,
                    embedding_col=embedding_col,
                    use_fingerprints=args.use_fingerprints,
                    mutation_encoding=args.mutation_encoding,
                    use_pos_norm=args.use_pos_norm,
                )
                X_test, y_test, blocks_test = prepare_single_embedding_features(
                    test_df,
                    embedding_col=embedding_col,
                    use_fingerprints=args.use_fingerprints,
                    mutation_encoding=args.mutation_encoding,
                    use_pos_norm=args.use_pos_norm,
                )
            except Exception as e:
                logger.warning(f"Skipping {embedding_col} due to feature error: {e}")
                continue

            # scale blocks for LR/SVM/MLP: scale mutation + embedding, keep fingerprint raw
            scale_blocks = [blocks_train["mutation"], blocks_train["embedding"]]

            # class imbalance support
            pos = int(np.sum(y_train == 1))
            neg = int(np.sum(y_train == 0))
            scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

            classifiers = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=300,
                    random_state=SEED,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
                "GradientBoosting": GradientBoostingClassifier(
                    n_estimators=200,
                    random_state=SEED,
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=500,
                    random_state=SEED,
                    eval_metric="logloss",
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                ),
                "LogisticRegression": LogisticRegression(
                    random_state=SEED,
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
                "SVM": SVC(
                    random_state=SEED,
                    probability=True,
                    class_weight="balanced",
                    kernel="rbf",
                ),
                "MLP": MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    random_state=SEED,
                    max_iter=800,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            }

            for model_name, clf in classifiers.items():
                logger.info(f"Training {model_name} on {embedding_name}...")

                if args.scale_for_linear and model_name in {
                    "LogisticRegression",
                    "SVM",
                    "MLP",
                }:
                    model = Pipeline(
                        steps=[
                            ("block_scaler", BlockStandardScaler(blocks=scale_blocks)),
                            ("clf", clf),
                        ]
                    )
                else:
                    model = Pipeline(steps=[("clf", clf)])

                model.fit(X_train, y_train)

                metrics = evaluate_classifier(
                    model, X_test, y_test, model_name=f"{model_name}_{embedding_name}"
                )
                metrics["Embedding_Type"] = embedding_name
                metrics["Mutation_Encoding"] = args.mutation_encoding
                metrics["Use_Fingerprints"] = bool(args.use_fingerprints)
                metrics["Use_Pos_Norm"] = bool(args.use_pos_norm)
                metrics["Scaled_Linear_MLP"] = bool(args.scale_for_linear)

                results.append(metrics)

                # Save model
                model_filename = f"{model_name.lower()}_{embedding_name.lower()}.joblib"
                model_path = model_dir / model_filename
                save_model(model, model_name, model_path)
                logger.info(f"Saved model to {model_path}")

        # Save consolidated results
        if results:
            results_df = pd.DataFrame(results)
            column_order = [
                "Model",
                "Embedding_Type",
                "Mutation_Encoding",
                "Use_Fingerprints",
                "Use_Pos_Norm",
                "Scaled_Linear_MLP",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC",
            ]
            columns = [c for c in column_order if c in results_df.columns] + [
                c for c in results_df.columns if c not in column_order
            ]
            results_df = results_df[columns]

            results_path = (
                output_dir
                / f"omiesi_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            results_df.to_csv(results_path, index=False)

            logger.info("\n=== BENCHMARK SUMMARY ===")
            logger.info(results_df.to_string(index=False))

            logger.info("\n=== BEST MODELS PER EMBEDDING (by F1) ===")
            for emb in sorted(results_df["Embedding_Type"].unique()):
                sub = results_df[results_df["Embedding_Type"] == emb]
                best = sub.loc[sub["F1"].idxmax()]
                logger.info(f"{emb}: {best['Model']} (F1={best['F1']:.4f})")
        else:
            logger.warning("No results produced (all embeddings failed or missing).")

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
