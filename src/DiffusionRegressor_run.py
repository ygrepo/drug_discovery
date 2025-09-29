# scripts/flow_matching_run.py
import sys
from pathlib import Path
import argparse
import os
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


def _default_workers():
    try:
        cpu = os.cpu_count() or 0
        return max(0, min(8, cpu - 1))
    except Exception:
        return 0


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DiffusionRegressorPL")

    # Logging
    p.add_argument("--log_fn", type=str, default=None)
    p.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    # Data/config (required)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument(
        "--splitmode",
        type=str,
        required=True,
        choices=["random", "cold_protein", "cold_drug"],
    )
    p.add_argument(
        "--embedding",
        type=str,
        required=True,
        choices=["ESMv1", "ESM2", "MUTAPLM", "ProteinCLIP"],
    )

    # Paths (you use these later)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--output_dir", type=str, default="output")
    p.add_argument(
        "--checkpoints_dir", type=str, default="output/checkpoints/diffusion_regressor"
    )
    p.add_argument(
        "--model_log_dir", type=str, default="output/logs/diffusion_regressor"
    )

    # Loader
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=_default_workers())
    p.add_argument(
        "--persistent_workers", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--check_nan", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--scale", action=argparse.BooleanOptionalAction, default=False)

    # Model
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--bilinear_rank", type=int, default=8)

    # Conditioning / time
    p.add_argument("--use_time", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--num_timesteps", type=int, default=10)

    # Targets / EMA
    p.add_argument(
        "--standardize_y", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--ema_decay", type=float, default=0.999, help="Set 0 to disable")

    # Loss / opt
    p.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"])
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Trainer
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--monitor", type=str, default="val_loss")
    p.add_argument("--monitor_mode", type=str, default="min", choices=["min", "max"])
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--save_top_k", type=int, default=1)
    p.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed"],
    )
    p.add_argument("--seed", type=int, default=42)

    # Hardware
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")

    # Logging
    p.add_argument("--log_every_n_steps", type=int, default=100)
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--log_fn", type=str, default=None)

    return p.parse_args()


def main():
    args = build_parser()

    # logging: capture returned logger
    setup_logging(Path(args.log_fn), args.log_level)

    # make results dirs
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_log_dir).mkdir(parents=True, exist_ok=True)

    # seed
    pl.seed_everything(args.seed, workers=True)

    try:
        logger.info(f"CWD: {os.getcwd()}")
        logger.info(
            f"Dataset: {args.dataset} | Split: {args.splitmode} | Emb: {args.embedding}"
        )
        base_data = (
            Path(args.data_dir) / f"{args.embedding}_{args.dataset}_{args.splitmode}"
        )
        logger.info(f"Data dir: {base_data}")
        logger.info(f"Output dir: {args.output_dir}")

        train_data, val_data, test_data = load_data(base_data)

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

        cfg = RegressorCfg(
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden=args.hidden,
            bilinear_rank=args.bilinear_rank,
            use_time=args.use_time,
            num_timesteps=args.num_timesteps,
            standardize_y=args.standardize_y,
            ema_decay=args.ema_decay,
            huber_delta=args.huber_delta,
        )

        pl_model = DiffusionRegressorPL(
            drug_input_dim=train_dataset.drug_input_dim,
            protein_input_dim=train_dataset.protein_input_dim,
            cfg=cfg,
            loss=args.loss,
        )
        pl_model.model.apply(init_weights)

        model_name = "DiffusionRegressor"
        checkpoint_dir = Path(args.checkpoints_dir) / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{model_name}_{{epoch:03d}}_{{{args.monitor}:.4f}}"

        callbacks = [
            ModelCheckpoint(
                monitor=args.monitor,
                mode=args.monitor_mode,
                save_top_k=args.save_top_k,
                dirpath=checkpoint_dir,
                save_last=True,
                filename=filename,
            ),
            EarlyStopping(
                monitor=args.monitor,
                mode=args.monitor_mode,
                patience=args.patience,
                min_delta=args.min_delta,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        csv_logger = CSVLogger(save_dir=str(Path(args.model_log_dir)), name=model_name)

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
            gradient_clip_val=args.grad_clip,
            precision=args.precision,
            accumulate_grad_batches=args.accumulate_grad_batches,
        )

        logger.info("Training…")
        trainer.fit(pl_model, train_loader, val_loader)
        logger.info("Testing…")
        trainer.test(pl_model, test_loader, ckpt_path="best")

    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
