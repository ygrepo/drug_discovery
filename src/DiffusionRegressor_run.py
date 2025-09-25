# scripts/flow_matching_run.py
import sys
from pathlib import Path
import argparse
import os
import torch
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
from src.DiffusionRegressor import (
    RegressorCfg,
    DiffusionRegressorPL,
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
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--huber_delta", type=float, default=1.0, help="Huber delta")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    p.add_argument("--hidden", type=int, default=512, help="Hidden dimension")
    p.add_argument("--bilinear_rank", type=int, default=8, help="Bilinear rank")
    p.add_argument("--use_time", type=bool, default=False, help="Use time embedding")
    p.add_argument("--num_timesteps", type=int, default=10, help="Number of timesteps")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    p.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )
    p.add_argument(
        "--checkpoints_dir", type=str, default="output/checkpoints/diffusion_regressor"
    )
    p.add_argument("--log_every_n_steps", type=int, default=100)
    p.add_argument(
        "--model_log_dir", type=str, default="output/logs/diffusion_regressor"
    )
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

        logger.info("Running model...")
        # datamodule/batch must yield keys: "drug" (B,Dd), "protein" (B,Dp), "y" (B,)
        cfg = RegressorCfg(
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden=args.hidden,
            bilinear_rank=args.bilinear_rank,
            use_time=args.use_time,
            num_timesteps=args.num_timesteps,
            standardize_y=True,
            ema_decay=args.ema_decay,
        )

        pl_model = DiffusionRegressorPL(
            drug_input_dim=train_dataset.drug_input_dim,
            protein_input_dim=train_dataset.protein_input_dim,
            cfg=cfg,
            loss="mse",
        )
        pl_model.model.apply(init_weights)
        model_name = "DiffusionRegressor"
        checkpoint_dir = Path(args.checkpoints_dir) / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{model_name}_epoch:{{epoch:03d}}_valloss:{{val_loss:.4f}}"

        callbacks = [
            ModelCheckpoint(
                monitor="val_r2_y",
                mode="max",
                save_top_k=1,
                dirpath=checkpoint_dir,
                save_last=True,
                filename=filename,
            ),
            EarlyStopping(
                monitor="val_r2_y", mode="max", patience=args.patience, min_delta=1e-3
            ),
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
            gradient_clip_val=1.0,
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
