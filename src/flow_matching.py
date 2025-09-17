import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from torchmetrics.regression import (
    MeanAbsoluteError,
    R2Score,
    PearsonCorrCoef,
    ExplainedVariance,
)


# --- Core MLP + Embeddings (unchanged from before) ---
def mlp(dims, act=nn.SiLU, last_act=False, dropout=0.0, layer_norm=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, 10, half, device=device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


@dataclass
class FlowConfig:
    hidden: int = 256
    t_dim: int = 128
    dropout: float = 0.1
    steps: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4


# --- Vector Field ---
class CondFlowVectorField(nn.Module):
    def __init__(self, drug_input_dim: int, protein_input_dim: int, cfg: FlowConfig):
        super().__init__()
        H = cfg.hidden
        self.drug_enc = mlp([drug_input_dim, H, H])
        self.prot_enc = mlp([protein_input_dim, H, H])
        self.time_emb = nn.Sequential(SinusoidalPosEmb(cfg.t_dim), mlp([cfg.t_dim, H]))
        self.x_proj = mlp([1, H])

        fused_dim = H * 4
        self.fuse = mlp([fused_dim, H * 2, H], dropout=cfg.dropout)
        self.out = nn.Linear(H, 1)

    def forward(self, x_t, t, cond):
        d = self.drug_enc(cond["drug"])
        p = self.prot_enc(cond["protein"])
        tau = self.time_emb(t)
        xv = self.x_proj(x_t)
        h = torch.cat([d, p, tau, xv], dim=-1)
        h = self.fuse(h)
        return self.out(h)


# --- Core Flow model ---
class DrugProteinFlowMatching(nn.Module):
    def __init__(self, drug_input_dim, protein_input_dim, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        self.vtheta = CondFlowVectorField(drug_input_dim, protein_input_dim, cfg)

    def forward(self, y, t, drug, protein):
        """
        Compute v_hat at x_t = (1-t)z + t y.
        Returns v_hat and v_star for loss computation.
        """
        B = y.size(0)
        z = torch.randn_like(y)
        x_t = (1 - t.view(B, 1)) * z + t.view(B, 1) * y
        v_star = y - z
        cond = {"drug": drug, "protein": protein}
        v_hat = self.vtheta(x_t, t, cond)
        return v_hat, v_star

    @torch.no_grad()
    def sample(self, drug, protein, steps=None, z=None):
        B = drug.size(0)
        steps = steps or self.cfg.steps
        dt = 1.0 / steps
        x = z if z is not None else torch.randn(B, 1, device=drug.device)

        for i in range(steps):
            t_scalar = (i + 0.5) / steps
            t = torch.full((B,), t_scalar, device=drug.device)
            v = self.vtheta(x, t, {"drug": drug, "protein": protein})
            x = x + dt * v
        return x


class DrugProteinFlowMatchingPL(pl.LightningModule):
    def __init__(
        self,
        drug_input_dim: int,
        protein_input_dim: int,
        cfg: FlowConfig,
        pred_num_samples: int = 50,
        pred_steps: Optional[int] = None,
        pi_alpha: float = 0.05,
        enable_sampled_eval: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.model = DrugProteinFlowMatching(drug_input_dim, protein_input_dim, cfg)
        self.cfg = cfg
        self.pred_num_samples = pred_num_samples
        self.pred_steps = pred_steps
        self.pi_alpha = pi_alpha

        # -------- Metrics (epoch-level only) --------
        # Train
        self.train_mae = MeanAbsoluteError()
        self.train_r2 = R2Score(num_outputs=1, multioutput="uniform_average")
        self.train_pearson = PearsonCorrCoef(num_outputs=1)
        self.train_ev = ExplainedVariance(num_outputs=1, multioutput="uniform_average")
        # Val
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score(num_outputs=1, multioutput="uniform_average")
        self.val_pearson = PearsonCorrCoef(num_outputs=1)
        self.val_ev = ExplainedVariance(num_outputs=1, multioutput="uniform_average")
        # Test
        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score(num_outputs=1, multioutput="uniform_average")
        self.test_pearson = PearsonCorrCoef(num_outputs=1)
        self.test_ev = ExplainedVariance(num_outputs=1, multioutput="uniform_average")

        # ===== eval-only metrics on prediction mean vs y_true =====
        self.val_mae_y = MeanAbsoluteError()
        self.val_r2_y = R2Score(num_outputs=1, multioutput="uniform_average")
        self.val_pearson_y = PearsonCorrCoef(num_outputs=1)
        self.val_ev_y = ExplainedVariance(num_outputs=1, multioutput="uniform_average")

        self.test_mae_y = MeanAbsoluteError()
        self.test_r2_y = R2Score(num_outputs=1, multioutput="uniform_average")
        self.test_pearson_y = PearsonCorrCoef(num_outputs=1)
        self.test_ev_y = ExplainedVariance(num_outputs=1, multioutput="uniform_average")

    # -------- Training / Val / Test --------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        B = y.size(0)
        t = torch.rand(B, device=self.device)
        v_hat, v_star = self.model(y, t, drug, prot)
        loss = F.mse_loss(v_hat, v_star)

        # log loss as before (step + epoch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # update metrics (NO per-step logging)
        self.train_mae.update(v_hat, v_star)
        self.train_r2.update(v_hat, v_star)
        self.train_pearson.update(v_hat, v_star)
        self.train_ev.update(v_hat, v_star)
        return loss

    def on_train_epoch_end(self):
        # compute & log once per epoch; then reset
        self.log(
            "train_mae",
            self.train_mae.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_r2",
            self.train_r2.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_pearson",
            self.train_pearson.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_ev",
            self.train_ev.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.train_mae.reset()
        self.train_r2.reset()
        self.train_pearson.reset()
        self.train_ev.reset()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        B = y.size(0)
        t = torch.rand(B, device=self.device)
        v_hat, v_star = self.model(y, t, drug, prot)
        loss = F.mse_loss(v_hat, v_star)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # collect metrics (epoch-only)
        self.val_mae.update(v_hat, v_star)
        self.val_r2.update(v_hat, v_star)
        self.val_pearson.update(v_hat, v_star)
        self.val_ev.update(v_hat, v_star)
        if self.enable_sampled_eval:
            with torch.no_grad():
                Y = self.sample_n(drug, prot)  # (B, K)
                y_mean = Y.mean(dim=1, keepdim=True)  # (B, 1)
            self.val_mae_y.update(y_mean, y)
            self.val_r2_y.update(y_mean, y)
            self.val_pearson_y.update(y_mean, y)
            self.val_ev_y.update(y_mean, y)
        return loss

    def on_validation_epoch_end(self):
        self.log(
            "val_mae",
            self.val_mae.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_r2",
            self.val_r2.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_pearson",
            self.val_pearson.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_ev",
            self.val_ev.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.val_mae.reset()
        self.val_r2.reset()
        self.val_pearson.reset()
        self.val_ev.reset()
        if self.enable_sampled_eval:
            self.log(
                "val_mae_y",
                self.val_mae_y.compute(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val_r2_y",
                self.val_r2_y.compute(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val_pearson_y",
                self.val_pearson_y.compute(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val_ev_y",
                self.val_ev_y.compute(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.val_mae_y.reset()
            self.val_r2_y.reset()
            self.val_pearson_y.reset()
            self.val_ev_y.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        B = y.size(0)
        t = torch.rand(B, device=self.device)
        v_hat, v_star = self.model(y, t, drug, prot)
        loss = F.mse_loss(v_hat, v_star)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.test_mae.update(v_hat, v_star)
        self.test_r2.update(v_hat, v_star)
        self.test_pearson.update(v_hat, v_star)
        self.test_ev.update(v_hat, v_star)
        if self.enable_sampled_eval:
            with torch.no_grad():
                Y = self.sample_n(drug, prot)  # (B, K)
                y_mean = Y.mean(dim=1, keepdim=True)  # (B, 1)
            self.test_mae_y.update(y_mean, y)
            self.test_r2_y.update(y_mean, y)
            self.test_pearson_y.update(y_mean, y)
            self.test_ev_y.update(y_mean, y)
        return loss

    def on_test_epoch_end(self):
        self.log(
            "test_mae",
            self.test_mae.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_r2",
            self.test_r2.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_pearson",
            self.test_pearson.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_ev",
            self.test_ev.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.test_mae.reset()
        self.test_r2.reset()
        self.test_pearson.reset()
        self.test_ev.reset()
        if self.enable_sampled_eval:
            self.log(
                "test_mae_y",
                self.test_mae_y.compute(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "test_r2_y",
                self.test_r2_y.compute(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

    # -------- Inference helpers (unchanged) --------
    @torch.no_grad()
    def sample_n(
        self,
        drug: torch.Tensor,
        prot: torch.Tensor,
        n_samples: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        self.model.eval()
        K = n_samples or self.pred_num_samples
        S = steps or self.pred_steps or self.cfg.steps
        ys = []
        for _ in range(K):
            yk = self.model.sample(drug, prot, steps=S)  # (B,1)
            ys.append(yk)
        return torch.cat(ys, dim=1)  # (B, K)

    @torch.no_grad()
    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        drug, prot = batch["drug"], batch["protein"]
        Y = self.sample_n(drug, prot)  # (B, K)
        mean = Y.mean(dim=1, keepdim=True)
        std = Y.std(dim=1, keepdim=True)
        z = (
            1.96
            if abs(self.pi_alpha - 0.05) < 1e-8
            else torch.tensor(
                float(
                    torch.distributions.Normal(0, 1)
                    .icdf(torch.tensor(1 - self.pi_alpha / 2.0))
                    .item()
                ),
                device=self.device,
            )
        )
        lo, hi = mean - z * std, mean + z * std
        out = {"y_samples": Y, "y_mean": mean, "y_std": std, "y_lo": lo, "y_hi": hi}
        if "y" in batch:
            y_true = batch["y"]
            out["y_true"] = y_true
            out["residual"] = mean - y_true
        return out
