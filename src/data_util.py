import sys
from typing import Callable, Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import get_logger

logger = get_logger(__name__)


def load_data(
    data_dir: Path,
    scale_features: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_parquet(data_dir / "train.parquet")
    val_data = pd.read_parquet(data_dir / "val.parquet")
    test_data = pd.read_parquet(data_dir / "test.parquet")
    logger.info(
        f"Loaded dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
    )
    if scale_features:
        logger.info("Scaling features...")
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)
    return train_data, val_data, test_data


def stack_dataset(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
):
    X_train = np.hstack(
        (
            np.vstack(train_data["Drug_Features"]),
            np.vstack(train_data["Target_Features"]),
        )
    )
    X_val = np.hstack(
        (np.vstack(val_data["Drug_Features"]), np.vstack(val_data["Target_Features"]))
    )
    X_test = np.hstack(
        (np.vstack(test_data["Drug_Features"]), np.vstack(test_data["Target_Features"]))
    )
    y_train, y_val, y_test = (
        train_data["Affinity"],
        val_data["Affinity"],
        test_data["Affinity"],
    )

    logger.info(
        f"StackedX_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


class DTIDataset(Dataset):
    """
    Efficient DTI dataset:
      - Pre-stacks columns to NumPy arrays once (zero-copy to tensors in __getitem__)
      - Returns keys compatible with your Lightning module: 'drug', 'protein', 'y'
    """

    def __init__(
        self,
        df,
        drug_col: str = "Drug_Features",
        protein_col: str = "Target_Features",
        y_col: str = "Affinity",
        transform: Optional[
            Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
        check_nan: bool = True,
    ):
        self.transform = transform

        # Extract & stack feature arrays once
        try:
            drug_list = df[drug_col].tolist()
            prot_list = df[protein_col].tolist()
        except KeyError as e:
            raise KeyError(f"Missing expected column: {e}")

        # Enforce array type/shape early
        self.drug = np.stack(
            [np.asarray(x, dtype=np.float32) for x in drug_list], axis=0
        )
        self.prot = np.stack(
            [np.asarray(x, dtype=np.float32) for x in prot_list], axis=0
        )

        # Targets (ensure 2D: (N,1))
        y_np = np.asarray(df[y_col].to_numpy(), dtype=np.float32).reshape(-1, 1)
        self.y = y_np

        if check_nan:
            for name, arr in [
                ("drug", self.drug),
                ("protein", self.prot),
                ("y", self.y),
            ]:
                if not np.isfinite(arr).all():
                    bad = np.argwhere(~np.isfinite(arr))
                    idx = bad[0, 0]
                    raise ValueError(
                        f"Non-finite values found in '{name}' (example index {idx})."
                    )

        # Cache shapes for quick introspection
        self.N = self.drug.shape[0]
        self.Dd = self.drug.shape[1]
        self.Dp = self.prot.shape[1]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Zero-copy: from_numpy creates a torch Tensor view on the underlying memory
        drug = torch.from_numpy(self.drug[idx])  # (Dd,)
        prot = torch.from_numpy(self.prot[idx])  # (Dp,)
        y = torch.from_numpy(self.y[idx])  # (1,)

        sample = {"drug": drug, "protein": prot, "y": y}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __repr__(self) -> str:
        return f"DTIDataset(N={self.N}, drug_dim={self.Dd}, protein_dim={self.Dp})"


def dti_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "drug": torch.stack([b["drug"] for b in batch], dim=0),
        "protein": torch.stack([b["protein"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
    }


class ZScore:
    def __init__(
        self,
        mu: Dict[str, torch.Tensor],
        sigma: Dict[str, torch.Tensor],
        eps: float = 1e-8,
    ):
        self.mu, self.sigma, self.eps = mu, sigma, eps

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k in sample:
            if k in self.mu and k in self.sigma:
                out[k] = (sample[k] - self.mu[k]) / (self.sigma[k] + self.eps)
            else:
                out[k] = sample[k]
        return out


def create_dataset_and_loader(
    df,
    *,
    drug_col: str = "Drug_Features",
    protein_col: str = "Target_Features",
    y_col: str = "Affinity",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,
    check_nan: bool = True,
) -> Tuple[DTIDataset, DataLoader]:
    dataset = DTIDataset(
        df,
        drug_col=drug_col,
        protein_col=protein_col,
        y_col=y_col,
        check_nan=check_nan,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        collate_fn=dti_collate,
    )
    return dataset, loader


def load_and_prepare_data(
    data_dir: Path,
    *,
    drug_col: str = "Drug_Features",
    protein_col: str = "Target_Features",
    y_col: str = "Affinity",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,
    check_nan: bool = True,
) -> Tuple[DTIDataset, DataLoader]:
    train_data, val_data, test_data = load_data(data_dir)
    train_dataset, train_loader = create_dataset_and_loader(
        train_data,
        drug_col=drug_col,
        protein_col=protein_col,
        y_col=y_col,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        check_nan=check_nan,
    )
    val_dataset, val_loader = create_dataset_and_loader(
        val_data,
        drug_col=drug_col,
        protein_col=protein_col,
        y_col=y_col,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        check_nan=check_nan,
    )
    test_dataset, test_loader = create_dataset_and_loader(
        test_data,
        drug_col=drug_col,
        protein_col=protein_col,
        y_col=y_col,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        check_nan=check_nan,
    )
    return (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    )
