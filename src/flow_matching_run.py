# scripts/flow_matching_run.py
import sys
from pathlib import Path
import argparse
import os
import torch
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

from scipy.stats import pearsonr


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import setup_logging, get_logger
from src.data_util import load_data, create_data_loader
from src.flow_matching import (
    FlowConfig,
    DrugProteinFlowMatchingPL,
)
from src.model_util import select_device, init_weights

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


def save_model(model, model_name, model_filename: Path):
    # Save the model to disk
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"{model_name} model saved to {model_filename}")


def parse_args():
    p = argparse.ArgumentParser(description="Create and load PLM model")
    p.add_argument("--log_fn", type=str, default="")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--splitmode", type=str, default="")
    p.add_argument("--model_dir", type=str, default="")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=20)
    p.add_argument("--pin_memory", type=bool, default=True)
    p.add_argument("--shuffle", type=bool, default=True)
    p.add_argument("--check_nan", type=bool, default=True)
    p.add_argument("--scale", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pred_num_samples", type=int, default=50)
    p.add_argument("--pred_steps", type=int, default=None)
    p.add_argument("--pi_alpha", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--accelerator", type=str, default="gpu")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--checkpoints_dir", type=str, default="./checkpoints/flow_matching")
    p.add_argument("--model_log_dir", type=str, default="./logs/flow_matching")
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
        logger.info(f"Data dir: {args.data_dir}")
        data_dir = Path(args.data_dir)
        data_dir = data_dir / f"{args.dataset}_{args.splitmode}"
        logger.info(f"Data dir: {data_dir}")
        logger.info(f"Output dir: {args.output_dir}")

        logger.info(f"Checkpoints dir: {args.checkpoints_dir}")
        logger.info(f"Model log dir: {args.model_log_dir}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Number of workers: {args.num_workers}")
        logger.info(f"Pin memory: {args.pin_memory}")
        logger.info(f"Shuffle: {args.shuffle}")
        logger.info(f"Check NaN: {args.check_nan}")
        logger.info(f"Scale: {args.scale}")
        logger.info(f"Hidden: {args.hidden}")
        logger.info(f"Steps: {args.steps}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Weight decay: {args.weight_decay}")
        logger.info(f"Dropout: {args.dropout}")
        logger.info(f"Prediction num samples: {args.pred_num_samples}")
        logger.info(f"Prediction steps: {args.pred_steps}")
        logger.info(f"PI alpha: {args.pi_alpha}")
        logger.info(f"Patience: {args.patience}")
        logger.info(f"Max epochs: {args.max_epochs}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Accelerator: {args.accelerator}")
        logger.info(f"Devices: {args.devices}")
        logger.info(f"Model dir: {args.model_dir}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"Log level: {args.log_level}")
        logger.info(f"Log fn: {args.log_fn}")
        logger.info(f"Split mode: {args.splitmode}")
        logger.info(f"Dataset: {args.dataset}")

        train_data, val_data, test_data = load_data(data_dir)
        train_loader, val_loader, test_loader, train_dataset = create_data_loader(
            train_data,
            val_data,
            test_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=args.shuffle,
            check_nan=args.check_nan,
            scale=args.scale,
        )

        logger.info("Running models...")
        cfg = FlowConfig(
            hidden=args.hidden,
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
        )
        pl_model = DrugProteinFlowMatchingPL(
            drug_input_dim=train_dataset.drug_input_dim,
            protein_input_dim=train_dataset.protein_input_dim,
            cfg=cfg,
            pred_num_samples=args.pred_num_samples,  # for predict_step
            pred_steps=args.pred_steps,  # use cfg.steps
            pi_alpha=args.pi_alpha,  # 95% PI
        )
        pl_model.model.apply(init_weights)
        model_name = "FlowMatching"
        checkpoint_dir = Path(args.checkpoints_dir) / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=checkpoint_dir,
                save_last=True,
                filename=model_name,
            ),
            EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        csv_logger = CSVLogger(save_dir=Path(args.model_log_dir), name=model_name)

        trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=args.max_epochs,
            callbacks=callbacks,
            logger=csv_logger,
            log_every_n_steps=10,
        )
        logger.info("Training...")
        trainer.fit(pl_model, train_loader, val_loader)
        logger.info("Testing...")
        trainer.test(pl_model, test_loader, ckpt_path="best")

        # metrics_df = pd.DataFrame(
        #     columns=[
        #         "Model",
        #         "Dataset",
        #         "RMSE",
        #         "MAE",
        #         "MSE",
        #         "R2",
        #         "Pearson",
        #         "Median_AE",
        #         "Explained_Variance",
        #     ]
        # )
        # logger.info("Done!")

        # output_dir = Path(args.output_dir)
        # datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # result_csv = (
        #     output_dir / f"{datestamp}_ML_metrics_{args.dataset}_{args.splitmode}.csv"
        # )

        # logger.info(f"Saving metrics to {result_csv}")
        # # Save metrics_df to CSV
        # metrics_df.to_csv(result_csv, index=False)
        # logger.info("Metrics saved!")

    except Exception as e:
        logger.exception("Script failed: %s", e)  # or this
        sys.exit(1)


if __name__ == "__main__":
    main()
