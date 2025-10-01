# --- model.py ---
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError, R2Score, ExplainedVariance
from torchmetrics import PearsonCorrCoef


# --- lightning_module.py ---


def mlp(dims, dropout=0.3, act=nn.SiLU):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), act(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class FiLM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gamma = nn.Linear(hidden, hidden)
        self.beta = nn.Linear(hidden, hidden)

    def forward(self, d, p):  # (B,H),(B,H)
        scale = torch.tanh(self.gamma(d))  # or clamp(-s,s)
        shift = self.beta(d)
        return p * (1 + scale) + shift


class LowRankBilinear(nn.Module):
    def __init__(self, hidden, rank=8):
        super().__init__()
        self.U = nn.Parameter(torch.randn(hidden, rank) / math.sqrt(hidden))
        self.V = nn.Parameter(torch.randn(hidden, rank) / math.sqrt(hidden))

    def forward(self, a, b):
        aU = a @ self.U  # (B,R)
        bV = b @ self.V  # (B,R)
        scalar = (aU * bV).sum(
            dim=-1, keepdim=True
        )  # (B,1)  (kept for potential diagnostics)
        vec = torch.cat([aU, bV], dim=-1)  # (B,2R)
        return scalar, vec


class DiffusionRegressorV2(nn.Module):
    """
    Robust conditional regressor (not a diffusion sampler):
      - Enc(d), Enc(p) -> FiLM(p|d) -> low-rank bilinear features + cosine
      - Residual MLP head -> y_hat
    """

    def __init__(
        self,
        drug_input_dim: int,
        protein_input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        bilinear_rank: int = 8,
        use_time: bool = False,
        num_timesteps: int = 10,
    ):
        super().__init__()
        self.use_time = use_time
        self.cosine_feat = True
        H = hidden_dim

        self.drug_enc = mlp([drug_input_dim, H, H], dropout=dropout)
        self.prot_enc = mlp([protein_input_dim, H, H], dropout=dropout)
        self.film = FiLM(H)
        self.bilin = LowRankBilinear(H, rank=bilinear_rank)

        fuse_in = H + H + 2 * bilinear_rank + 1 + (1 if self.cosine_feat else 0)
        if self.use_time and num_timesteps > 1:
            self.t_embed = nn.Embedding(num_timesteps, fuse_in)

        self.pre_norm = nn.LayerNorm(fuse_in)
        self.body = mlp([fuse_in, 2 * H, 2 * H, H], dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(H), nn.SiLU(), nn.Dropout(dropout), nn.Linear(H, 1)
        )

    def forward(self, drug, protein, t=None):
        d = self.drug_enc(drug)
        p = self.prot_enc(protein)
        p_mod = self.film(d, p)

        scalar, s_vec = self.bilin(d, p_mod)
        feats = [d, p_mod, s_vec, scalar]  # scalar is (B,1)

        if self.cosine_feat:
            dn = F.normalize(d, dim=-1)
            pn = F.normalize(p_mod, dim=-1)
            feats.append((dn * pn).sum(dim=-1, keepdim=True))

        fused = torch.cat(feats, dim=-1)
        if self.use_time and t is not None:
            fused = fused + self.t_embed(t.long())

        fused = self.pre_norm(fused)
        h = self.body(fused)
        y_hat = self.head(h).squeeze(-1)  # (B,)
        return y_hat


@dataclass
class RegressorCfg:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    hidden: int = 512
    bilinear_rank: int = 8
    use_time: bool = False
    num_timesteps: int = 10
    huber_delta: float = 1.0
    standardize_y: bool = False
    ema_decay: float = 0.999  # set 0 to disable


class DiffusionRegressorPL(pl.LightningModule):
    """
    Lightning wrapper that mirrors your FM PL module:
      - uses identical metric names (train/val/test: mae, r2, pearson, ev)
      - logs epoch-level metrics
      - optional target standardization + EMA
    """

    def __init__(
        self,
        drug_input_dim: int,
        protein_input_dim: int,
        cfg: RegressorCfg,
        loss: str = "huber",  # "huber" or "mse"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.model = DiffusionRegressorV2(
            drug_input_dim,
            protein_input_dim,
            hidden_dim=cfg.hidden,
            dropout=cfg.dropout,
            bilinear_rank=cfg.bilinear_rank,
            use_time=cfg.use_time,
            num_timesteps=cfg.num_timesteps,
        )
        self.loss_name = loss

        # --- running target stats (buffers) ---
        self.register_buffer("y_mu", torch.tensor(0.0), persistent=False)
        self.register_buffer("y_sigma", torch.tensor(1.0), persistent=False)
        self.register_buffer("y_M2", torch.tensor(0.0), persistent=False)
        self._seen_for_stats = 0  # simple running-estimate counter

        # --- EMA shadow ---
        self.ema_decay = cfg.ema_decay
        self._ema_shadow = None  # dict of tensors or None

        # --- Metrics ---
        self.train_mae = MeanAbsoluteError()
        self.train_r2 = R2Score(multioutput="uniform_average")
        self.train_pearson = PearsonCorrCoef()
        self.train_ev = ExplainedVariance(multioutput="uniform_average")

        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score(multioutput="uniform_average")
        self.val_pearson = PearsonCorrCoef()
        self.val_ev = ExplainedVariance(multioutput="uniform_average")

        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score(multioutput="uniform_average")
        self.test_pearson = PearsonCorrCoef()
        self.test_ev = ExplainedVariance(multioutput="uniform_average")

    # -------- helpers --------
    def _scale_y(self, y):
        if not self.cfg.standardize_y:
            return y
        return (y - self.y_mu) / (self.y_sigma.clamp_min(1e-8))

    def _unscale_y(self, y_std):
        if not self.cfg.standardize_y:
            return y_std
        return y_std * self.y_sigma + self.y_mu

    @torch.no_grad()
    def _update_running_stats(self, y):
        if not self.cfg.standardize_y:
            return
        y = y.detach().to(self.y_mu.device).flatten()
        # vectorized Welford
        for val in y:
            self._seen_for_stats += 1
            delta = val - self.y_mu
            self.y_mu += delta / self._seen_for_stats
            self.y_M2 += delta * (val - self.y_mu)
        var = self.y_M2 / max(self._seen_for_stats, 1)
        self.y_sigma.copy_(torch.sqrt(var.clamp_min(1e-12)))

    def _maybe_init_ema(self):
        if self.ema_decay and (self._ema_shadow is None):
            self._ema_shadow = {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()
                if v.dtype.is_floating_point
            }

    @torch.no_grad()
    def _ema_update(self):
        if not self.ema_decay or self._ema_shadow is None:
            return
        d = self.ema_decay
        for k, v in self.model.state_dict().items():
            if k in self._ema_shadow and v.dtype.is_floating_point:
                self._ema_shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    # -------- Lightning hooks --------
    def on_train_start(self):
        self._maybe_init_ema()

    # -------- Steps --------
    def _loss_fn(self, y_hat, y_std):
        if self.loss_name == "mse":
            return F.mse_loss(y_hat, y_std)
        # default huber
        return F.huber_loss(y_hat, y_std, delta=self.cfg.huber_delta)

    def training_step(self, batch, batch_idx):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        y = y.squeeze(-1) if y.dim() > 1 else y  # Ensure y is 1D
        # Update running stats early in training
        if self.global_step < 100:  # warmup estimates
            self._update_running_stats(y)

        y_std = self._scale_y(y)
        y_hat_std = self.model(drug, prot)  # (B,)

        loss = self._loss_fn(y_hat_std, y_std)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # epoch metrics in original scale
        y_hat = self._unscale_y(y_hat_std).detach()
        self.train_mae.update(y_hat, y)
        self.train_r2.update(y_hat, y)
        self.train_pearson.update(y_hat, y)
        self.train_ev.update(y_hat, y)

        # EMA
        self._ema_update()
        return loss

    def on_train_epoch_end(self):
        try:
            self.log(
                "train_mae",
                self.train_mae.compute(),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train_r2",
                self.train_r2.compute(),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train_pearson",
                self.train_pearson.compute(),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train_ev",
                self.train_ev.compute(),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
        finally:
            self.train_mae.reset()
            self.train_r2.reset()
            self.train_pearson.reset()
            self.train_ev.reset()

    def validation_step(self, batch, batch_idx):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        y = y.squeeze(-1) if y.dim() > 1 else y  # Ensure y is 1D
        y_std = self._scale_y(y)
        y_hat_std = self.model(drug, prot)
        loss = self._loss_fn(y_hat_std, y_std)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # metrics on EMA weights at epoch end (see on_validation_epoch_end)
        # here we log with current weights too:
        y_hat = self._unscale_y(y_hat_std).detach()
        self.val_mae.update(y_hat, y)
        self.val_r2.update(y_hat, y)
        self.val_pearson.update(y_hat, y)
        self.val_ev.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self):
        # --- regular metrics ---
        try:
            self.log("val_mae", self.val_mae.compute(), prog_bar=True, sync_dist=True)
            self.log("val_r2", self.val_r2.compute(), prog_bar=True, sync_dist=True)
            self.log(
                "val_pearson",
                self.val_pearson.compute(),
                prog_bar=False,
                sync_dist=True,
            )
            self.log("val_ev", self.val_ev.compute(), prog_bar=False, sync_dist=True)
        finally:
            self.val_mae.reset()
            self.val_r2.reset()
            self.val_pearson.reset()
            self.val_ev.reset()

        # --- EMA metrics (safe + no_grad + restore mode) ---
        if self._ema_shadow is not None:
            current = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }
            was_training = self.model.training
            with torch.no_grad():
                for k, v in self.model.state_dict().items():
                    if k in self._ema_shadow:
                        v.copy_(self._ema_shadow[k])
                self.model.eval()
                mae_ema = MeanAbsoluteError().to(self.device)
                r2_ema = R2Score().to(self.device)
                pcc_ema = PearsonCorrCoef().to(self.device)
                ev_ema = ExplainedVariance().to(self.device)
                for b in self.trainer.datamodule.val_dataloader():
                    b = {k: v.to(self.device) for k, v in b.items()}
                    y = b["y"]
                    y = y.squeeze(-1) if y.dim() > 1 else y  # Ensure y is 1D
                    y_hat = self._unscale_y(self.model(b["drug"], b["protein"]))
                    mae_ema.update(y_hat, y)
                    r2_ema.update(y_hat, y)
                    pcc_ema.update(y_hat, y)
                    ev_ema.update(y_hat, y)
            self.log("val_mae_ema", mae_ema.compute(), prog_bar=True)
            self.log("val_r2_ema", r2_ema.compute())
            self.log("val_pearson_ema", pcc_ema.compute())
            self.log("val_ev_ema", ev_ema.compute())
            self.model.load_state_dict(current)
            if was_training:
                self.model.train()

    def test_step(self, batch, batch_idx):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        y = y.squeeze(-1) if y.dim() > 1 else y  # Ensure y is 1D
        y_std = self._scale_y(y)
        y_hat_std = self.model(drug, prot)
        loss = self._loss_fn(y_hat_std, y_std)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        y_hat = self._unscale_y(y_hat_std).detach()
        self.test_mae.update(y_hat, y)
        self.test_r2.update(y_hat, y)
        self.test_pearson.update(y_hat, y)
        self.test_ev.update(y_hat, y)
        return loss

    def on_test_epoch_end(self):
        self.log(
            "test_mae",
            self.test_mae.compute(),
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_r2",
            self.test_r2.compute(),
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_pearson",
            self.test_pearson.compute(),
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_ev",
            self.test_ev.compute(),
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.test_mae.reset()
        self.test_r2.reset()
        self.test_pearson.reset()
        self.test_ev.reset()

    # -------- Optimizers --------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
        }
