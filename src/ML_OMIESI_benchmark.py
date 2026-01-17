# ML_OMIESI_benchmark.py - Binary classification benchmark for OMIESI dataset
import sys
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List

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
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logging, get_logger
from src.ML_benchmark_util import save_model

logger = get_logger(__name__)
SEED = 42


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
    p.add_argument("--output_dir", type=str, default="output/omiesi_benchmark")
    p.add_argument("--model_dir", type=str, default="models/omiesi_benchmark")
    p.add_argument(
        "--use_fingerprints",
        action="store_true",
        help="Include SMILES fingerprints as features",
    )
    p.add_argument(
        "--embedding_models",
        nargs="+",
        default=["ESMv1", "ESM2", "MUTAPLM", "ProteinCLIP"],
        help="Which embedding models to use",
    )
    return p.parse_args()


def load_omiesi_data(data_path: Path) -> pd.DataFrame:
    """Load OMIESI data with expected columns."""
    expected_columns = [
        "WA",
        "Pos",
        "MA",
        "ESMv1_embedding",
        "ESM2_embedding",
        "MUTAPLM_embedding",
        "ProteinCLIP_embedding",
        "SMILES_fingerprint",
        "Y",
    ]

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Check for required columns
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")

    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def encode_amino_acid_features(df: pd.DataFrame) -> np.ndarray:
    """
    Convert WA, Pos, MA mutation information into numerical features.

    Args:
        df: DataFrame with WA, Pos, MA columns

    Returns:
        Feature matrix with amino acid properties
    """
    # Amino acid properties (basic physicochemical properties)
    aa_properties = {
        "A": [
            89.1,
            1.8,
            6.0,
            0,
            0,
            0,
        ],  # Ala: MW, hydrophobicity, pKa, aromatic, charged, polar
        "R": [174.2, -4.5, 10.8, 0, 1, 1],  # Arg
        "N": [132.1, -3.5, 5.4, 0, 0, 1],  # Asn
        "D": [133.1, -3.5, 1.9, 0, -1, 1],  # Asp
        "C": [121.0, 2.5, 10.8, 0, 0, 0],  # Cys
        "Q": [146.1, -3.5, 5.7, 0, 0, 1],  # Gln
        "E": [147.1, -3.5, 4.2, 0, -1, 1],  # Glu
        "G": [75.1, -0.4, 6.0, 0, 0, 0],  # Gly
        "H": [155.2, -3.2, 7.6, 1, 1, 1],  # His
        "I": [131.2, 4.5, 6.0, 0, 0, 0],  # Ile
        "L": [131.2, 3.8, 6.0, 0, 0, 0],  # Leu
        "K": [146.2, -3.9, 9.7, 0, 1, 1],  # Lys
        "M": [149.2, 1.9, 5.7, 0, 0, 0],  # Met
        "F": [165.2, 2.8, 5.5, 1, 0, 0],  # Phe
        "P": [115.1, -1.6, 6.3, 0, 0, 0],  # Pro
        "S": [105.1, -0.8, 5.7, 0, 0, 1],  # Ser
        "T": [119.1, -0.7, 5.6, 0, 0, 1],  # Thr
        "W": [204.2, -0.9, 5.9, 1, 0, 0],  # Trp
        "Y": [181.2, -1.3, 5.7, 1, 0, 1],  # Tyr
        "V": [117.1, 4.2, 6.0, 0, 0, 0],  # Val
    }

    features = []

    for _, row in df.iterrows():
        wa = row["WA"]  # Wild-type amino acid
        ma = row["MA"]  # Mutant amino acid
        pos = row["Pos"]  # Position

        # Get amino acid properties
        wa_props = aa_properties.get(wa, [0] * 6)
        ma_props = aa_properties.get(ma, [0] * 6)

        # Create feature vector
        mutation_features = [
            pos,  # Position (1 feature)
            *wa_props,  # Wild-type AA properties (6 features)
            *ma_props,  # Mutant AA properties (6 features)
            *[
                ma_props[i] - wa_props[i] for i in range(6)
            ],  # Property differences (6 features)
        ]

        features.append(mutation_features)

    feature_matrix = np.array(features, dtype=np.float32)
    logger.info(
        f"Encoded amino acid features: {feature_matrix.shape} (19 features per mutation)"
    )

    return feature_matrix


def prepare_single_embedding_features(
    df: pd.DataFrame, embedding_model: str, use_fingerprints: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix from OMIESI data for a single embedding model.
    Features: [amino_acid_features, embedding_features, fingerprint_features]

    Args:
        df: DataFrame with OMIESI data
        embedding_model: Single embedding model to use (e.g., "ESMv1")
        use_fingerprints: Whether to include SMILES fingerprints

    Returns:
        X: Feature matrix
        y: Target labels
    """
    features = []
    feature_names = []

    # 1. Add amino acid mutation features (WA, Pos, MA -> 19 features)
    aa_features = encode_amino_acid_features(df)
    features.append(aa_features)
    feature_names.extend([f"aa_{i}" for i in range(aa_features.shape[1])])
    logger.info(f"Added amino acid features: {aa_features.shape}")

    # 2. Add single protein embedding
    col_name = f"{embedding_model}_embedding"
    if col_name in df.columns:
        # Convert embedding strings/arrays to numpy arrays
        embeddings = []
        for emb in df[col_name]:
            if isinstance(emb, str):
                # Handle string representation of arrays
                emb_array = np.fromstring(emb.strip("[]"), sep=" ")
            elif isinstance(emb, (list, np.ndarray)):
                emb_array = np.array(emb)
            else:
                logger.warning(
                    f"Unexpected embedding format for {embedding_model}: {type(emb)}"
                )
                emb_array = np.zeros(512)  # Default size
            embeddings.append(emb_array)

        embeddings = np.vstack(embeddings)
        features.append(embeddings)
        feature_names.extend(
            [f"{embedding_model}_{i}" for i in range(embeddings.shape[1])]
        )
        logger.info(f"Added {embedding_model} embeddings: {embeddings.shape}")
    else:
        raise ValueError(f"Embedding column {col_name} not found in data")

    # 3. Add SMILES fingerprints if requested
    if use_fingerprints and "SMILES_fingerprint" in df.columns:
        fingerprints = []
        for fp in df["SMILES_fingerprint"]:
            if isinstance(fp, str):
                fp_array = np.fromstring(fp.strip("[]"), sep=" ")
            elif isinstance(fp, (list, np.ndarray)):
                fp_array = np.array(fp)
            else:
                logger.warning(f"Unexpected fingerprint format: {type(fp)}")
                fp_array = np.zeros(2048)  # Default Morgan fingerprint size
            fingerprints.append(fp_array)

        fingerprints = np.vstack(fingerprints)
        features.append(fingerprints)
        feature_names.extend([f"fp_{i}" for i in range(fingerprints.shape[1])])
        logger.info(f"Added SMILES fingerprints: {fingerprints.shape}")

    # Concatenate all features
    if features:
        X = np.hstack(features)
    else:
        raise ValueError("No features could be extracted from the data")

    # Get target labels
    y = df["Y"].values

    logger.info(f"Final feature matrix shape for {embedding_model}: {X.shape}")
    logger.info(
        f"Feature breakdown: AA({aa_features.shape[1]}) + Embedding({embeddings.shape[1]}) + Fingerprints({fingerprints.shape[1] if use_fingerprints else 0})"
    )
    logger.info(f"Target distribution: {np.bincount(y)}")

    return X, y


def evaluate_classifier(
    model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
) -> dict:
    """Evaluate a classifier and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="binary"),
        "Recall": recall_score(y_test, y_pred, average="binary"),
        "F1": f1_score(y_test, y_pred, average="binary"),
    }

    if y_pred_proba is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"{model_name} Results:")
    for metric, value in metrics.items():
        if metric != "Model":
            logger.info(f"  {metric}: {value:.4f}")

    return metrics


def main():
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)

    try:
        logger.info("Starting OMIESI binary classification benchmark")
        logger.info(f"Arguments: {vars(args)}")

        # Create output directories
        output_dir = Path(args.output_dir)
        model_dir = Path(args.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load data from directory
        data_dir = Path(args.data_dir)

        train_path = data_dir / "train.csv"
        val_path = data_dir / "val.csv"
        test_path = data_dir / "test.csv"

        # Check if all files exist
        for file_path in [train_path, val_path, test_path]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load all three datasets
        train_df = load_omiesi_data(train_path)
        val_df = load_omiesi_data(val_path)
        test_df = load_omiesi_data(test_path)

        logger.info(f"Training set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")

        # Define classifiers
        classifiers = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=SEED, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=SEED
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100, random_state=SEED, eval_metric="logloss"
            ),
            "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
            "SVM": SVC(random_state=SEED, probability=True),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=SEED, max_iter=500
            ),
        }

        # Train and evaluate models for each embedding type
        results = []

        for embedding_model in args.embedding_models:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing embedding: {embedding_model}")
            logger.info(f"{'='*50}")

            # Prepare features for this embedding
            try:
                X_train, y_train = prepare_single_embedding_features(
                    train_df, embedding_model, args.use_fingerprints
                )

                X_val, y_val = prepare_single_embedding_features(
                    val_df, embedding_model, args.use_fingerprints
                )
                X_test, y_test = prepare_single_embedding_features(
                    test_df, embedding_model, args.use_fingerprints
                )

                logger.info(
                    f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features"
                )
                logger.info(f"Validation set: {X_val.shape[0]} samples")
                logger.info(f"Test set: {X_test.shape[0]} samples")

            except ValueError as e:
                logger.warning(f"Skipping {embedding_model}: {e}")
                continue

            # Train each classifier for this embedding
            for model_name, model_class in classifiers.items():
                logger.info(f"\nTraining {model_name} with {embedding_model}...")

                # Create fresh model instance
                if model_name == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=100, random_state=SEED, n_jobs=-1
                    )
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingClassifier(
                        n_estimators=100, random_state=SEED
                    )
                elif model_name == "XGBoost":
                    model = XGBClassifier(
                        n_estimators=100, random_state=SEED, eval_metric="logloss"
                    )
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(random_state=SEED, max_iter=1000)
                elif model_name == "SVM":
                    model = SVC(random_state=SEED, probability=True)
                elif model_name == "MLP":
                    model = MLPClassifier(
                        hidden_layer_sizes=(100, 50), random_state=SEED, max_iter=500
                    )

                # Train model
                model.fit(X_train, y_train)

                # Evaluate on test set
                metrics = evaluate_classifier(
                    model, X_test, y_test, f"{model_name}_{embedding_model}"
                )
                metrics["Embedding_Type"] = embedding_model
                results.append(metrics)

                # Save model
                model_filename = f"{model_name.replace(' ', '_').lower()}_{embedding_model.lower()}.joblib"
                model_path = model_dir / model_filename
                save_model(model, model_path)
                logger.info(f"Saved model to {model_path}")

        # Save consolidated results
        if results:
            results_df = pd.DataFrame(results)

            # Reorder columns for better readability
            column_order = [
                "Model",
                "Embedding_Type",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC",
            ]
            available_columns = [
                col for col in column_order if col in results_df.columns
            ]
            remaining_columns = [
                col for col in results_df.columns if col not in available_columns
            ]
            final_columns = available_columns + remaining_columns
            results_df = results_df[final_columns]

            # Save to CSV
            results_path = (
                output_dir
                / f"omiesi_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            results_df.to_csv(results_path, index=False)
            logger.info(f"Saved consolidated results to {results_path}")

            # Print summary
            logger.info("\n=== BENCHMARK SUMMARY ===")
            logger.info("Results by Embedding Type and Model:")
            logger.info(results_df.to_string(index=False))

            # Print best performing models per embedding
            logger.info("\n=== BEST MODELS PER EMBEDDING ===")
            for embedding in args.embedding_models:
                embedding_results = results_df[
                    results_df["Embedding_Type"] == embedding
                ]
                if not embedding_results.empty:
                    best_model = embedding_results.loc[embedding_results["F1"].idxmax()]
                    logger.info(
                        f"{embedding}: {best_model['Model']} (F1: {best_model['F1']:.4f})"
                    )
        else:
            logger.warning("No results to save - all embeddings may have failed")

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
