import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Download and clean MutaDescribe dataset."""
    # Output directory
    out_dir = Path("../dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Actual CSVs in PharMolix/MutaDescribe
    files_to_download = {
        "structural_split": "structural_split/train.csv",
        "temporal_split": "temporal_split/train.csv",
        "pubs": "pubs/train.csv",
    }

    # Download, clean, and save
    for split_name, hf_path in files_to_download.items():
        print(f"Downloading: {hf_path}")
        local_path = hf_hub_download(
            repo_id="PharMolix/MutaDescribe", repo_type="dataset", filename=hf_path
        )

        df = pd.read_csv(local_path)

        # Drop unwanted auto-generated index columns
        df = df.drop(
            columns=[
                col
                for col in df.columns
                if col.startswith("Unnamed") or col in {"index", "level_0"}
            ],
            errors="ignore",
        )

        out_path = out_dir / f"mutadescribe_{split_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Saved cleaned CSV: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
