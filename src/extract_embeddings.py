import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import argparse

from numpy import dot
from numpy.linalg import norm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_util import load_model, load_tokenizer
from src.utils import setup_logging


def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (e.g., 'INFO', 'DEBUG')

    Returns:
        Configured logger instance
    """
    # Create output directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extract_embeddings_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


# def load_model(model_name: str) -> AutoModel:
#     """
#     Load an ESM model safely.

#     - Prefers safetensors if available (no torch.load / pickle)
#     - Works offline with local paths
#     - Enforces HF_HOME for caching on HPC
#     """
#     # Ensure Hugging Face cache points to project space
#     os.environ.setdefault(
#         "HF_HOME",
#         "/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.cache/huggingface",
#     )
#     logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
#     logger.info(f"Loading model: {model_name}")

#     # Prefer safe serialization
#     try:
#         model = AutoModel.from_pretrained(
#             model_name,
#             add_pooling_layer=False,
#             trust_remote_code=True,  # some ESM models need this
#             local_files_only=os.path.isabs(model_name),  # offline if local path
#         ).eval()
#         logger.info(f"✅ Loaded model: {model_name}")
#         return model
#     except ValueError as e:
#         # Catch CVE / torch.load errors
#         if "torch.load" in str(e):
#             raise RuntimeError(
#                 f"Model '{model_name}' requires safetensors or PyTorch ≥2.6.\n"
#                 "Convert the model to safetensors and use the local path instead."
#             ) from e
#         raise


def load_model(model_name: str) -> AutoModel:
    """
    Load an ESM model safely.

    - Prefers safetensors if available (no torch.load / pickle)
    - Works offline with local paths
    - Enforces HF_HOME for caching on HPC
    """
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f"Loading model: {model_name}")

    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load ESM tokenizer."""
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f"Loading Tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def embed_sequence(tokenizer, model, seq):
    """Embed a protein sequence using mean-pooled ESM embeddings."""
    tokens = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens).last_hidden_state
    return outputs.mean(dim=1).squeeze().numpy()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from protein sequences"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=15,
        help="Number of rows to process (for testing)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="ESM model name",
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to the output dataset file",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save log files",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g., 'INFO', 'DEBUG')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Convert paths to absolute paths relative to project root
    project_root = Path(__file__).parent.parent
    log_dir = Path(project_root / args.log_dir).resolve()
    # Set up logging
    logger = setup_logging(log_dir, args.log_level)

    try:
        # Log configuration
        logger.info("Starting training with configuration:")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"  Project root: {project_root}")
        logger.info(f"  Data fn: {args.data_fn}")
        logger.info(f"  Output fn: {args.output_fn}")
        logger.info(f"  Model name: {args.model_name}")
        logger.info(f"  Log directory: {log_dir}")
        logger.info(f"  Log level: {args.log_level}")
        logger.info(f"  Random seed: {args.seed}")

        df = pd.read_csv(
            Path(args.data_fn),
            low_memory=False,
        )
        df.drop(columns=["Unnamed: 0"], inplace=True)
        # Drop missing sequences
        df = df.dropna(subset=["protein1", "protein2"])
        logger.info(f"Loaded dataset: {len(df)} rows")
        if args.n > 0:
            logger.info(f"Sampling {args.n} rows")
            df = df.sample(n=args.n, random_state=args.seed)

        logger.info("Extracting embeddings...")

        # Load HF model
        model_name = args.model_name
        model = load_model(model_name)
        tokenizer = load_tokenizer(model_name)
        logger.info(f"Loaded tokenizer: {model_name}")
        logger.info(f"Loaded model: {model_name}")

        # Embed all protein1 and protein2 sequences
        protein1_embeddings = []
        protein2_embeddings = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                emb1 = embed_sequence(tokenizer, model, row["protein1"])
                emb2 = embed_sequence(tokenizer, model, row["protein2"])
                protein1_embeddings.append(emb1)
                protein2_embeddings.append(emb2)
            except Exception as e:
                logger.error("Embedding error:", e)

        # Save embeddings
        df["protein1_embedding"] = protein1_embeddings
        df["protein2_embedding"] = protein2_embeddings

        # Compute cosine similarity
        logger.info("Computing cosine similarity...")
        df["cosine_similarity"] = [
            cosine_similarity(emb1, emb2)
            for emb1, emb2 in zip(protein1_embeddings, protein2_embeddings)
        ]

        logger.info(f"{df.head()}")
        df.to_csv(
            Path(project_root / args.output_fn),
            index=False,
        )

        logger.info(f"Saved embeddings to {args.output_fn}")

    except Exception as e:
        logger.exception("Training failed", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
