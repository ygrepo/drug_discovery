# scripts/load_mutaplm_model.py
import sys
from pathlib import Path

# Make imports robust regardless of CWD (repo layout: <repo>/{model,scripts,configs,...})
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.model_util import (
    select_device,
    create_mutaplm_model,
    load_mutaplm_model,
    check_mutaplm_model,
    check_mutaplm_min,
)
from src.utils import setup_logging
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Create and load MutaPLM model")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "mutaplm_inference.yaml"),
    )
    p.add_argument(
        "--device", type=str, default="auto", help="auto|cpu|cuda|cuda:N|mps"
    )
    p.add_argument(
        "--checkpoint_path", type=str, default=str(REPO_ROOT / "ckpts" / "mutaplm.pth")
    )
    return p.parse_args()


def main():

    args = parse_args()
    logger = setup_logging("load_mutaplm_model", Path(args.log_dir), args.log_level)
    device = select_device(args.device)
    logger.info(f"Using device: {device}")
    model = create_mutaplm_model(Path(args.config), device)
    model = load_mutaplm_model(model, Path(args.checkpoint_path))
    logger.info("Model loaded successfully.")
    check_mutaplm_min(model)
    check_mutaplm_model(model)


if __name__ == "__main__":
    main()
