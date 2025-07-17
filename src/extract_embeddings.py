import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


def embed_sequence(tokenizer, model, seq):
    """Embed a protein sequence using mean-pooled ESM embeddings."""
    tokens = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens).last_hidden_state
    return outputs.mean(dim=1).squeeze().numpy()


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
    log_file = log_dir / f"create_features_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def main():
    # Set up logging
    logger = setup_logging(Path("logs"), "DEBUG")

    # Load ESM model
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Loaded tokenizer: {model_name}")
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).eval()
    logger.info(f"Loaded model: {model_name}")

    # Load the dataset
    df = pd.read_csv(Path("../dataset/structural_split/train.csv"), low_memory=False)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    # Drop missing sequences
    df = df.dropna(subset=["protein1", "protein2"])
    logger.info(f"Loaded dataset: {len(df)} rows")

    # Embed all protein1 and protein2 sequences
    protein1_embeddings = []
    protein2_embeddings = []
    # Use a small subset for speed (remove this line to run on full dataset)
    # df2 = df.sample(n=15, random_state=42)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            emb1 = embed_sequence(tokenizer, model, row["protein1"])
            emb2 = embed_sequence(tokenizer, model, row["protein2"])
            protein1_embeddings.append(emb1)
            protein2_embeddings.append(emb2)
        except Exception as e:
            print("Embedding error:", e)

    # Save embeddings
    df["protein1_embedding"] = protein1_embeddings
    df["protein2_embedding"] = protein2_embeddings
    df.to_csv(
        Path("../dataset/structural_split/train_with_embeddings.csv"), index=False
    )


if __name__ == "__main__":
    main()
