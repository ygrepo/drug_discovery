from __future__ import annotations
import sys
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data as GNNData
from torch_geometric.loader import DataLoader as GeoDataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import get_logger

logger = get_logger(__name__)


def load_data(
    data_dir: Path,
    scale_features: bool = False,
    N: Optional[int] = None,  # Add N parameter
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_parquet(data_dir / "train.parquet")
    val_data = pd.read_parquet(data_dir / "val.parquet")
    test_data = pd.read_parquet(data_dir / "test.parquet")

    # Limit rows if N is specified
    if N > 0:
        train_data = train_data.head(N)
        val_data = val_data.head(N)
        test_data = test_data.head(N)
        logger.info(f"Limited datasets to {N} rows each")

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    def __init__(
        self,
        df: pd.DataFrame,
        drug_smiles_col: str = "Drug",
        drug_col: str = "Drug_Features",
        protein_col: str = "Target_Features",
        target_id_col: str = "Target",
        y_col: Optional[str] = "Affinity",
        scale: Optional[str] = None,  # None | "zscore" | "minmax"
        check_nan: bool = True,
    ):
        self.scale = scale

        # Keep originals you want to export later
        self.row_index = df.index.to_numpy()
        self.drug_smiles = df[drug_smiles_col].astype(str).tolist()
        self.target_id = df[target_id_col].astype(str).tolist()

        # Features
        self.drug = np.stack(
            [np.asarray(x, dtype=np.float32) for x in df[drug_col]], axis=0
        )
        self.prot = np.stack(
            [np.asarray(x, dtype=np.float32) for x in df[protein_col]], axis=0
        )

        # Target (optional)
        if y_col is not None and y_col in df.columns:
            self.y = np.asarray(df[y_col].to_numpy(), dtype=np.float32).reshape(-1, 1)
        else:
            self.y = None

        if check_nan:
            for name, arr in [("drug", self.drug), ("protein", self.prot)]:
                if not np.isfinite(arr).all():
                    raise ValueError(f"NaN/Inf found in '{name}'")
            if self.y is not None and not np.isfinite(self.y).all():
                raise ValueError("NaN/Inf found in 'y'")

        self.N, self.drug_input_dim = self.drug.shape
        self.protein_input_dim = self.prot.shape[1]

        # Scale y if present
        self.y_mu = self.y_sigma = self.y_min = self.y_max = None
        if self.y is not None and self.scale is not None:
            if self.scale == "zscore":
                self.y_mu = self.y.mean()
                self.y_sigma = self.y.std() + 1e-8
                self.y = (self.y - self.y_mu) / self.y_sigma
            elif self.scale == "minmax":
                self.y_min = self.y.min()
                self.y_max = self.y.max()
                self.y = (self.y - self.y_min) / (self.y_max - self.y_min + 1e-8)

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        out = {
            "row_index": int(self.row_index[idx]),
            "smiles": self.drug_smiles[idx],
            "target_id": self.target_id[idx],
            "drug": torch.from_numpy(self.drug[idx]),
            "protein": torch.from_numpy(self.prot[idx]),
        }
        if self.y is not None:
            out["y"] = torch.from_numpy(self.y[idx])
        return out

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.scale == "zscore" and self.y_sigma is not None:
            return y_scaled * self.y_sigma + self.y_mu
        if self.scale == "minmax" and self.y_min is not None:
            return y_scaled * (self.y_max - self.y_min) + self.y_min
        return y_scaled


def loader_to_numpy(
    dl: DataLoader,
) -> tuple[np.ndarray, np.ndarray | None, list[str], list[str], list[int]]:
    Xs, Ys, SMILES, TARGETS, ROW_IDX = [], [], [], [], []
    for batch in dl:
        # --- Concatenate drug and protein features ---
        drug = batch["drug"]
        prot = batch["protein"]
        Xb = torch.cat([drug, prot], dim=-1)  # (B, D_d + D_p)
        Xs.append(Xb.cpu().numpy())

        if "y" in batch:
            Ys.append(batch["y"].view(-1).cpu().numpy())
        SMILES.extend(list(batch["smiles"]))
        TARGETS.extend(list(batch["target_id"]))  # from DTIDataset.__getitem__
        ROW_IDX.extend([int(i) for i in batch["row_index"]])

    X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, 0), dtype=np.float32)
    y = np.concatenate(Ys, axis=0) if Ys else None
    return X, y, SMILES, TARGETS, ROW_IDX


def append_predictions(
    model_name: str,
    df: pd.DataFrame,
    row_idx: list[int],
    smiles: list[str],
    target_ids: list[str],
    preds: np.ndarray,
):
    pred_df = pd.DataFrame(
        {
            "Model": [model_name] * len(row_idx),
            "row_index": row_idx,
            "Drug": smiles,
            "Target": target_ids,  # use the ID column name consistently
            "pred_affinity": np.asarray(preds).reshape(-1),
        }
    )
    logger.info(f"Appending {len(pred_df)} predictions to dataframe")
    df = pd.concat(
        [df, pred_df], ignore_index=True, sort=False
    )  # sort=False is more efficient
    df = df.sort_values("row_index").reset_index(drop=True)  # Sort by row_index
    return df


def dti_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "drug": torch.stack([b["drug"] for b in batch], dim=0),
        "protein": torch.stack([b["protein"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
    }


class DTIGraphVecDataset(Dataset):
    """
    Row-wise graph + vector dataset.

    Each row must provide:
      - Drug_Features          : (N_atoms, F) float32-like
      - Drug_Edge_Index        : (2, E) int64-like
      - Target_Features         : (Dp,) float32-like
      - y_col               : scalar (float)

    Scaling (if requested) is fit on this dataset (train), and val/test should
    copy over y_mu/y_sigma or y_min/y_max from the train instance.
    """

    def __init__(
        self,
        df,
        *,
        drug_x_col: str = "Drug_Features",
        drug_edge_index_col: str = "Drug_Edge_Index",
        protein_col: str = "Target_Features",  # align with your flow-matching default
        y_col: str = "Affinity",
        check_nan: bool = True,
        scale: Optional[str] = None,  # None | "zscore" | "minmax"
    ):
        self.df = df.reset_index(drop=True)
        self.drug_x_col = drug_x_col
        self.drug_edge_index_col = drug_edge_index_col
        self.protein_col = protein_col
        self.y_col = y_col
        self.scale = scale

        # --- validate columns
        missing = [
            c
            for c in [drug_x_col, drug_edge_index_col, protein_col, y_col]
            if c not in df.columns
        ]
        if missing:
            raise KeyError(
                f"Missing required columns in df: {missing}. "
                f"Have: {list(df.columns)}"
            )

        # --- extract y as (N, 1) float32
        ys = np.asarray(df[y_col].to_numpy(), dtype=np.float32).reshape(-1, 1)

        # --- fit scaling on THIS dataset (typically train)
        self.y_mu = self.y_sigma = self.y_min = self.y_max = None
        if scale == "zscore":
            self.y_mu = float(ys.mean())
            self.y_sigma = float(ys.std() + 1e-8)
            y_scaled = (ys - self.y_mu) / self.y_sigma
        elif scale == "minmax":
            self.y_min = float(ys.min())
            self.y_max = float(ys.max())
            y_scaled = (ys - self.y_min) / (self.y_max - self.y_min + 1e-8)
        else:
            y_scaled = ys

        self.y = y_scaled.astype(np.float32, copy=False)

        if check_nan and not np.isfinite(self.y).all():
            raise ValueError("NaN/Inf in target after scaling.")

        # --- cache dims (from first row)
        # handle possible ragged drug_x — infer feature dim F, not N_atoms
        first_x = np.asarray(self.df.iloc[0][self.drug_x_col], dtype=np.float32)
        if first_x.ndim != 2:
            raise ValueError(
                f"{self.drug_x_col} must be (N_atoms, F), got shape {first_x.shape}"
            )
        self.drug_input_dim = int(first_x.shape[1])

        first_p = np.asarray(self.df.iloc[0][self.protein_col], dtype=np.float32)
        if first_p.ndim != 1:
            raise ValueError(
                f"{self.protein_col} must be (Dp,), got shape {first_p.shape}"
            )
        self.protein_input_dim = int(first_p.shape[0])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = torch.as_tensor(row[self.drug_x_col], dtype=torch.float32)  # (N,F)
        edge_index = torch.as_tensor(
            row[self.drug_edge_index_col], dtype=torch.long
        )  # (2,E)
        protein = torch.as_tensor(row[self.protein_col], dtype=torch.float32)  # (Dp,)
        y = torch.as_tensor(self.y[idx], dtype=torch.float32).view(1)  # (1,)

        return GNNData(x=x, edge_index=edge_index, y=y, protein=protein)

    # Use the *train* dataset instance to inverse-transform predictions
    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.scale == "zscore":
            return y_scaled * self.y_sigma + self.y_mu
        if self.scale == "minmax":
            return y_scaled * (self.y_max - self.y_min) + self.y_min
        return y_scaled


def create_DTI_FlowMatching_data_loader(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,
    check_nan: bool = True,
    scale: Optional[str] = None,  # <-- "zscore", "minmax", or None
) -> Tuple[DataLoader, DataLoader, DataLoader, DTIDataset]:
    pin_memory = pin_memory and torch.cuda.is_available()
    # Train dataset
    train_dataset = DTIDataset(train_data, check_nan=check_nan, scale=scale)
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        collate_fn=dti_collate,
    )

    # Val/Test must use the SAME scaling params as train
    val_dataset = DTIDataset(val_data, check_nan=check_nan, scale=None)
    val_dataset.y_mu, val_dataset.y_sigma = train_dataset.y_mu, train_dataset.y_sigma
    val_dataset.y_min, val_dataset.y_max = train_dataset.y_min, train_dataset.y_max
    if scale == "zscore":
        val_dataset.y = (val_dataset.y - val_dataset.y_mu) / val_dataset.y_sigma
    elif scale == "minmax":
        val_dataset.y = (val_dataset.y - val_dataset.y_min) / (
            val_dataset.y_max - val_dataset.y_min + 1e-8
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        collate_fn=dti_collate,
    )

    test_dataset = DTIDataset(test_data, check_nan=check_nan, scale=None)
    test_dataset.y_mu, test_dataset.y_sigma = train_dataset.y_mu, train_dataset.y_sigma
    test_dataset.y_min, test_dataset.y_max = train_dataset.y_min, train_dataset.y_max
    if scale == "zscore":
        test_dataset.y = (test_dataset.y - test_dataset.y_mu) / test_dataset.y_sigma
    elif scale == "minmax":
        test_dataset.y = (test_dataset.y - test_dataset.y_min) / (
            test_dataset.y_max - test_dataset.y_min + 1e-8
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        collate_fn=dti_collate,
    )

    return train_loader, val_loader, test_loader, train_dataset


def create_DTI_GNNdata_loader(
    train_df,
    val_df,
    test_df,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,
    check_nan: bool = True,
    scale: Optional[str] = None,  # "zscore" | "minmax" | None (applies to y)
) -> Tuple[GeoDataLoader, GeoDataLoader, GeoDataLoader, DTIGraphVecDataset]:
    pin_memory = pin_memory and torch.cuda.is_available()
    persistent_workers = num_workers > 0

    # --- Train
    train_dataset = DTIGraphVecDataset(train_df, check_nan=check_nan, scale=scale)
    train_loader = GeoDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    # --- Val (use train stats)
    val_dataset = DTIGraphVecDataset(val_df, check_nan=check_nan, scale=None)
    val_dataset.y_mu, val_dataset.y_sigma = train_dataset.y_mu, train_dataset.y_sigma
    val_dataset.y_min, val_dataset.y_max = train_dataset.y_min, train_dataset.y_max
    ys = np.asarray(val_df["Affinity"].to_numpy(), dtype=np.float32).reshape(-1, 1)
    if scale == "zscore":
        val_dataset.y = (ys - val_dataset.y_mu) / (val_dataset.y_sigma + 1e-8)
    elif scale == "minmax":
        val_dataset.y = (ys - val_dataset.y_min) / (
            val_dataset.y_max - val_dataset.y_min + 1e-8
        )

    val_loader = GeoDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    # --- Test (use train stats)
    test_dataset = DTIGraphVecDataset(test_df, check_nan=check_nan, scale=None)
    test_dataset.y_mu, test_dataset.y_sigma = train_dataset.y_mu, train_dataset.y_sigma
    test_dataset.y_min, test_dataset.y_max = train_dataset.y_min, train_dataset.y_max
    ys = np.asarray(test_df["Affinity"].to_numpy(), dtype=np.float32).reshape(-1, 1)
    if scale == "zscore":
        test_dataset.y = (ys - test_dataset.y_mu) / (test_dataset.y_sigma + 1e-8)
    elif scale == "minmax":
        test_dataset.y = (ys - test_dataset.y_min) / (
            test_dataset.y_max - test_dataset.y_min + 1e-8
        )

    test_loader = GeoDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, train_dataset


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
