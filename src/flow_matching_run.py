# scripts/flow_matching_run.py
import sys
from pathlib import Path
import argparse
import os
import pandas as pd
import numpy as np
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import setup_logging, get_logger
from src.data_util import load_data, create_DTI_FlowMatching_data_loader
from src.flow_matching import (
    FlowConfig,
    DrugProteinFlowMatchingPL,
)
from src.model_util import init_weights

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Create and load PLM model")
    p.add_argument("--log_fn", type=str, default="")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--splitmode", type=str, default="")
    p.add_argument("--embedding", type=str, default="")
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
    p.add_argument("--t_dim", type=int, default=128)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pred_num_samples", type=int, default=50)
    p.add_argument("--pred_steps", type=int, default=None)
    p.add_argument("--pi_alpha", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--accelerator", type=str, default="gpu")
    p.add_argument(
        "--devices", type=int, default=1, help="Number of devices (GPUs or CPUs) to use"
    )
    p.add_argument("--checkpoints_dir", type=str, default="./checkpoints/flow_matching")
    p.add_argument("--log_every_n_steps", type=int, default=100)
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
        data_dir = data_dir / f"{args.embedding}_{args.dataset}_{args.splitmode}"
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
        logger.info(f"T dim: {args.t_dim}")
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
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"Log level: {args.log_level}")
        logger.info(f"Log every n steps: {args.log_every_n_steps}")
        logger.info(f"Log fn: {args.log_fn}")
        logger.info(f"Split mode: {args.splitmode}")
        logger.info(f"Dataset: {args.dataset}")

        train_data, val_data, test_data = load_data(data_dir)
        train_loader, val_loader, test_loader, train_dataset = (
            create_DTI_FlowMatching_data_loader(
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
        )

        logger.info("Running models...")
        cfg = FlowConfig(
            hidden=args.hidden,
            t_dim=args.t_dim,
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
        )
        pl.seed_everything(42, workers=True)
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
        filename = f"{model_name}_epoch:{{epoch:03d}}_valloss:{{val_loss:.4f}}"

        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=checkpoint_dir,
                save_last=True,
                filename=filename,
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
            log_every_n_steps=args.log_every_n_steps,
            deterministic=True,
            enable_model_summary=True,
            enable_progress_bar=True,
            precision="16-mixed",
        )
        logger.info("Training...")
        trainer.fit(pl_model, train_loader, val_loader)
        logger.info("Testing...")
        trainer.test(pl_model, test_loader, ckpt_path="best")

    except Exception as e:
        logger.exception("Script failed: %s", e)  # or this
        sys.exit(1)


if __name__ == "__main__":
    main()
