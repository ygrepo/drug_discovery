import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
import pickle

from scipy.stats import pearsonr


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import get_logger
from src.data_util import loader_to_numpy, loader_to_numpy_no_smiles
from torch.utils.data import DataLoader

logger = get_logger(__name__)

SEED = 42


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    """
    Calculate regression metrics including RMSE, MAE, MSE, R2, Pearson correlation,
    Median Absolute Error, and Explained Variance.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate Pearson correlation coefficient (returns coefficient and p-value)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    # Calculate Median Absolute Error
    median_ae = median_absolute_error(y_true, y_pred)

    # Calculate Explained Variance Score
    explained_variance = explained_variance_score(y_true, y_pred)

    return rmse, mae, mse, r2, pearson_corr, median_ae, explained_variance


def evaluate_model(
    metrics_df: pd.DataFrame,
    model_name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Train and evaluate a model, and save metrics to a global DataFrame.

    Parameters:
      - model_name: Name of the model.
      - model: The regression model object with a .predict() method.
      - X_train, y_train: Training features and labels.
      - X_val, y_val: Validation features and labels.
      - X_test, y_test: Test features and labels.
      - data_dir: Directory to save the model file.
    """

    # Get predictions and calculate metrics for each split.
    train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, train_pred)

    val_pred = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_pred)

    test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_pred)

    # Prepare a list of datasets and corresponding metrics.
    datasets = ["Training", "Validation", "Test"]
    metrics = [train_metrics, val_metrics, test_metrics]
    rows = []
    for dataset, metric in zip(datasets, metrics):
        rmse, mae, mse, r2, pearson_corr, median_ae, explained_variance = metric
        rows.append(
            {
                "Model": model_name,
                "Dataset": dataset,
                "RMSE": rmse,
                "MAE": mae,
                "MSE": mse,
                "R2": r2,
                "Pearson": pearson_corr,
                "Median_AE": median_ae,
                "Explained_Variance": explained_variance,
            }
        )

    # Update the global metrics DataFrame
    metrics_df = pd.concat([metrics_df, pd.DataFrame(rows)], ignore_index=True)
    return metrics_df


def evaluate_model_with_loaders(
    metrics_df: pd.DataFrame,
    model_name: str,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    drug_col: str,
    protein_col: str,
    y_inverse_fn=None,  # pass dataset.inverse_transform_y if scaled
):
    # Materialize arrays for sklearn
    X_train, y_train, _, _, train_row_idx = loader_to_numpy(
        drug_col=drug_col,
        protein_col=protein_col,
        dl=train_loader,
        smiles_col=None,
        target_id_col=None,
    )
    X_val, y_val, _, _, val_row_idx = loader_to_numpy(
        drug_col=drug_col,
        protein_col=protein_col,
        dl=val_loader,
        smiles_col=None,
        target_id_col=None,
    )
    X_test, y_test, test_row_idx, test_smiles, test_target_ids = loader_to_numpy(
        drug_col=drug_col,
        protein_col=protein_col,
        dl=test_loader,
        smiles_col=None,
        target_id_col=None,
    )

    # Fit and predict
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Inverse-transform predictions/targets if needed
    if y_inverse_fn is not None:
        train_pred = y_inverse_fn(train_pred)
        val_pred = y_inverse_fn(val_pred)
        test_pred = y_inverse_fn(test_pred)
        if y_train is not None:
            y_train = y_inverse_fn(y_train)
        if y_val is not None:
            y_val = y_inverse_fn(y_val)
        if y_test is not None:
            y_test = y_inverse_fn(y_test)

    # Metrics
    rows = []
    for split_name, (y_true, y_hat) in {
        "Training": (y_train, train_pred),
        "Validation": (y_val, val_pred),
        "Test": (y_test, test_pred),
    }.items():
        rmse, mae, mse, r2, pearson_corr, median_ae, explained_variance = (
            calculate_metrics(y_true, y_hat)
        )
        rows.append(
            {
                "Model": model_name,
                "Dataset": split_name,
                "RMSE": rmse,
                "MSE": mse,
                "MAE": mae,
                "R2": r2,
                "Pearson": pearson_corr,
                "Median_AE": median_ae,
                "Explained_Variance": explained_variance,
            }
        )
    metrics_df = pd.concat([metrics_df, pd.DataFrame(rows)], ignore_index=True)

    return metrics_df, (test_row_idx, test_smiles, test_target_ids, test_pred)


def evaluate_model_with_loaders_no_smiles(
    metrics_df: pd.DataFrame,
    data_name: str,
    embedding_name: str,
    model_name: str,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    drug_col: str,
    protein_col: str,
    y_inverse_fn=None,  # pass dataset.inverse_transform_y if scaled
) -> tuple[pd.DataFrame, tuple[list[int], np.ndarray]]:
    # Materialize arrays for sklearn
    X_train, y_train, train_row_idx = loader_to_numpy_no_smiles(
        drug_col=drug_col,
        protein_col=protein_col,
        dl=train_loader,
    )
    X_val, y_val, val_row_idx = loader_to_numpy_no_smiles(
        drug_col=drug_col,
        protein_col=protein_col,
        dl=val_loader,
    )
    X_test, y_test, test_row_idx = loader_to_numpy_no_smiles(
        drug_col=drug_col, protein_col=protein_col, dl=test_loader
    )

    # Fit and predict
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Inverse-transform predictions/targets if needed
    if y_inverse_fn is not None:
        train_pred = y_inverse_fn(train_pred)
        val_pred = y_inverse_fn(val_pred)
        test_pred = y_inverse_fn(test_pred)
        if y_train is not None:
            y_train = y_inverse_fn(y_train)
        if y_val is not None:
            y_val = y_inverse_fn(y_val)
        if y_test is not None:
            y_test = y_inverse_fn(y_test)

    # Metrics
    rows = []
    for split_name, (y_true, y_hat) in {
        "Training": (y_train, train_pred),
        "Validation": (y_val, val_pred),
        "Test": (y_test, test_pred),
    }.items():
        rmse, mae, mse, r2, pearson_corr, median_ae, explained_variance = (
            calculate_metrics(y_true, y_hat)
        )
        rows.append(
            {
                "Data": data_name,
                "Embedding": embedding_name,
                "Model": model_name,
                "Dataset": split_name,
                "RMSE": rmse,
                "MSE": mse,
                "MAE": mae,
                "R2": r2,
                "Pearson": pearson_corr,
                "Median_AE": median_ae,
                "Explained_Variance": explained_variance,
            }
        )
    metrics_df = pd.concat([metrics_df, pd.DataFrame(rows)], ignore_index=True)

    return metrics_df, (test_row_idx, test_pred)


def save_model(model, model_name, model_filename: Path):
    # Save the model to disk
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"{model_name} model saved to {model_filename}")
