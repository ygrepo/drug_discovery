# scripts/ML_benchmark.py
import sys
from pathlib import Path
import argparse
import os

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
from src.utils import setup_logging, get_logger
from src.data_util import (
    load_data,
    DTIDataset,
    append_predictions_csv,
)
from src.ML_benchmark_util import evaluate_model_with_loaders, save_model

logger = get_logger(__name__)

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Create and load PLM model")
    p.add_argument("--log_fn", type=str, default="")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--splitmode", type=str, default="")
    p.add_argument("--embedding", type=str, default="")
    p.add_argument("--model_dir", type=str, default="")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=20)
    p.add_argument("shuffle", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data_dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)

    try:
        # Log configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Logging to: {args.log_fn}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Split mode: {args.splitmode}")
        logger.info(f"Embedding: {args.embedding}")
        logger.info(f"Model dir: {args.model_dir}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Shuffle: {args.shuffle}")
        logger.info(f"Num workers: {args.num_workers}")
        logger.info(f"Data dir: {args.data_dir}")
        data_dir = Path(args.data_dir)
        data_dir = data_dir / f"{args.embedding}_{args.dataset}_{args.splitmode}"
        logger.info(f"Data dir: {data_dir}")

        # --- Load data ---
        train_df, val_df, test_df = load_data(data_dir)

        # --- Build datasets/loaders ---
        train_ds = DTIDataset(
            train_df, y_col="Affinity", scale="zscore"
        )  # or "minmax"/None
        val_ds = DTIDataset(
            val_df, y_col="Affinity", scale=train_ds.scale
        )  # keep same scaling choice
        test_ds = DTIDataset(test_df, y_col="Affinity", scale=train_ds.scale)

        shuffle = args.shuffle
        batch_size = args.batch_size
        num_workers = args.num_workers
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        logger.info("Running models...")

        logger.info("Random Forest")
        # Random Forest
        model_name = "Random Forest"
        model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)

        # --- Evaluate ---
        metrics_df = pd.DataFrame(
            columns=[
                "Model",
                "Dataset",
                "RMSE",
                "MSE",
                "MAE",
                "R2",
                "Pearson",
                "Median_AE",
                "Explained_Variance",
            ]
        )
        metrics_df, (test_smiles, test_pred) = evaluate_model_with_loaders(
            metrics_df,
            model_name,
            model,
            train_loader,
            val_loader,
            test_loader,
            y_inverse_fn=train_ds.inverse_transform_y if train_ds.scale else None,
        )

        # --- Save model ---
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)

        # --- Append test predictions to CSV ---
        output_dir = Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        predictions_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_predictions.csv"
        )
        append_predictions_csv(predictions_filename, test_smiles, test_pred)
        logger.info(
            f"Appended {len(test_smiles)} test predictions to {predictions_filename}"
        )

        logger.info("SVR")
        # SVR
        model_name = "SVR"
        model = SVR(kernel="rbf")
        metrics_df, (test_smiles, test_pred) = evaluate_model_with_loaders(
            metrics_df,
            model_name,
            model,
            train_loader,
            val_loader,
            test_loader,
            y_inverse_fn=train_ds.inverse_transform_y if train_ds.scale else None,
        )
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)
        predictions_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_predictions.csv"
        )
        append_predictions_csv(predictions_filename, test_smiles, test_pred)
        logger.info(
            f"Appended {len(test_smiles)} test predictions to {predictions_filename}"
        )

        logger.info("GBM")
        # GBM
        model_name = "GBM"
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=SEED
        )
        metrics_df, (test_smiles, test_pred) = evaluate_model_with_loaders(
            metrics_df,
            model_name,
            model,
            train_loader,
            val_loader,
            test_loader,
            y_inverse_fn=train_ds.inverse_transform_y if train_ds.scale else None,
        )
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)
        predictions_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_predictions.csv"
        )
        append_predictions_csv(predictions_filename, test_smiles, test_pred)
        logger.info(
            f"Appended {len(test_smiles)} test predictions to {predictions_filename}"
        )

        logger.info("Linear Regression")
        # Linear Regression
        model = LinearRegression()
        model_name = "Linear Regression"
        metrics_df, (test_smiles, test_pred) = evaluate_model_with_loaders(
            metrics_df,
            model_name,
            model,
            train_loader,
            val_loader,
            test_loader,
            y_inverse_fn=train_ds.inverse_transform_y if train_ds.scale else None,
        )
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)
        predictions_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_predictions.csv"
        )
        append_predictions_csv(predictions_filename, test_smiles, test_pred)
        logger.info(
            f"Appended {len(test_smiles)} test predictions to {predictions_filename}"
        )

        logger.info("MLP")
        # MLP
        model = MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            max_iter=200,
            random_state=SEED,
        )
        model_name = "MLP"
        metrics_df, (test_smiles, test_pred) = evaluate_model_with_loaders(
            metrics_df,
            model_name,
            model,
            train_loader,
            val_loader,
            test_loader,
            y_inverse_fn=train_ds.inverse_transform_y if train_ds.scale else None,
        )
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)
        predictions_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_predictions.csv"
        )
        append_predictions_csv(predictions_filename, test_smiles, test_pred)
        logger.info(
            f"Appended {len(test_smiles)} test predictions to {predictions_filename}"
        )

        logger.info("XGBoost")
        # XGBoost
        model = XGBRegressor(random_state=SEED, eval_metric="rmse")
        model_name = "XGBoost"
        metrics_df, (test_smiles, test_pred) = evaluate_model_with_loaders(
            metrics_df,
            model_name,
            model,
            train_loader,
            val_loader,
            test_loader,
            y_inverse_fn=train_ds.inverse_transform_y if train_ds.scale else None,
        )
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)
        predictions_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_predictions.csv"
        )
        append_predictions_csv(predictions_filename, test_smiles, test_pred)
        logger.info(
            f"Appended {len(test_smiles)} test predictions to {predictions_filename}"
        )

        logger.info("Done!")

        output_dir = Path(args.output_dir)
        datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_csv = (
            output_dir
            / f"{datestamp}_ML_metrics_{args.embedding}_{args.dataset}_{args.splitmode}.csv"
        )

        logger.info(f"Saving metrics to {result_csv}")
        # Save metrics_df to CSV
        metrics_df.to_csv(result_csv, index=False)
        logger.info("Metrics saved!")

    except Exception as e:
        logger.exception("Script failed: %s", e)  # or this
        sys.exit(1)


if __name__ == "__main__":
    main()
