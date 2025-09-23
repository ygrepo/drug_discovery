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


def mlp(dims, act=nn.SiLU, last_act=False, dropout=0.0, layer_norm=True):
    layers = []
    L = len(dims) - 1
    for i in range(L):
        in_d, out_d = dims[i], dims[i + 1]
        layers.append(nn.Linear(in_d, out_d))
        is_last = i == L - 1
        if (not is_last) or last_act:
            if layer_norm:
                layers.append(nn.LayerNorm(out_d))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SinusoidalPosEmb(nn.Module):
    """t can be int timesteps [0..T-1] or floats; we map to [0,1]."""

    def __init__(self, dim: int = 128, max_freq: float = 1e4):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int or float
        t = t.float()
        # if t is integer steps, normalize to [0,1]
        if t.ndim == 1 and t.max() > 1.0 + 1e-6:
            t = t / (t.max() + 1e-8)

        half = self.dim // 2
        # frequencies spaced geometrically in [1, max_freq]
        # ω_i = max_freq^(i/half), i=0..half-1
        device = t.device
        i = torch.arange(half, device=device).float()
        freqs = self.max_freq ** (i / max(half - 1, 1))

        args = t[:, None] * freqs[None, :]  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
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


# --- Encoders & fusion blocks ---
class DrugEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden], dropout=dropout)

    def forward(self, x):  # (B, in_dim)
        return self.net(x)  # (B, hidden)


class ProteinEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden], dropout=dropout)

    def forward(self, x):
        return self.net(x)


class ConditionFusion(nn.Module):
    """
    Concatenate [drug_emb, protein_emb] -> LayerNorm -> Dropout -> (optional) MLP.
    """

    def __init__(self, hidden: int, use_mlp: bool = True, dropout: float = 0.3):
        super().__init__()
        concat_dim = hidden * 2
        self.ln = nn.LayerNorm(concat_dim)
        self.do = nn.Dropout(dropout)
        self.use_mlp = use_mlp
        if use_mlp:
            self.post = mlp(
                [concat_dim, hidden * 2, hidden], dropout=dropout
            )  # like your denoise_model
        else:
            self.post = nn.Identity()

    def forward(self, drug_emb, prot_emb, t_emb=None):
        c = torch.cat([drug_emb, prot_emb], dim=-1)  # (B, 2H)
        c = self.ln(c)
        if t_emb is not None:
            # ensure same dim as c
            if t_emb.shape[-1] != c.shape[-1]:
                raise ValueError(
                    f"t_emb dim {t_emb.shape[-1]} must equal concat dim {c.shape[-1]}"
                )
            c = c + t_emb
        c = self.do(c)
        c = self.post(c)  # (B, H) if use_mlp else (B, 2H)
        return c


# --- Vector Field with explicit encoders + fusion like your DiffusionGenerativeModel ---
class CondFlowVectorField(nn.Module):
    def __init__(self, drug_input_dim: int, protein_input_dim: int, cfg: FlowConfig):
        super().__init__()
        H = cfg.hidden

        # Encoders (analogous to your drug/protein linear + ReLU but deeper + LN/Dropout via mlp)
        self.drug_enc = DrugEncoder(drug_input_dim, H, dropout=cfg.dropout)
        self.prot_enc = ProteinEncoder(protein_input_dim, H, dropout=cfg.dropout)

        # Time embedding (sinusoidal → MLP to H*2 so it can be added to [drug,protein] concat)
        self.time_emb_raw = SinusoidalPosEmb(cfg.t_dim)
        self.time_proj_for_concat = mlp([cfg.t_dim, H * 2])

        # Fusion like your layer_norm + dropout + "denoise_model"
        self.fusion = ConditionFusion(hidden=H, use_mlp=True, dropout=cfg.dropout)
        # If fusion.use_mlp=True → fused cond dim == H, else 2H. We assume H here.

        # Project x_t (scalar y_t) to hidden
        self.x_proj = mlp([1, H])

        # Final fuse [cond_fused(H), x_proj(H)] → H → 1
        self.final_fuse = mlp([H * 2, H], dropout=cfg.dropout)
        self.out = nn.Linear(H, 1)

    def forward(self, x_t, t, cond):
        # 1) encoders
        d = self.drug_enc(cond["drug"])  # (B, H)
        p = self.prot_enc(cond["protein"])  # (B, H)

        # 2) time embedding in concat space (B, 2H) so it can be *added* like your example
        tau = self.time_emb_raw(t)  # (B, t_dim)
        tau = self.time_proj_for_concat(tau)  # (B, 2H)

        # 3) fuse to a single conditioning vector (B, H)
        c = self.fusion(d, p, t_emb=tau)  # (B, H)

        # 4) project current state x_t and fuse with condition
        xv = self.x_proj(x_t)  # (B, H)
        h = torch.cat([c, xv], dim=-1)  # (B, 2H)
        h = self.final_fuse(h)  # (B, H)

        # 5) predict velocity
        return self.out(h)  # (B, 1)


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
        self.train_r2 = R2Score(multioutput="uniform_average")
        self.train_pearson = PearsonCorrCoef()
        self.train_ev = ExplainedVariance(multioutput="uniform_average")
        # Val
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score(multioutput="uniform_average")
        self.val_pearson = PearsonCorrCoef()
        self.val_ev = ExplainedVariance(multioutput="uniform_average")
        # Test
        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score(multioutput="uniform_average")
        self.test_pearson = PearsonCorrCoef()
        self.test_ev = ExplainedVariance(multioutput="uniform_average")

        # ===== eval-only metrics on prediction mean vs y_true =====
        self.val_mae_y = MeanAbsoluteError()
        self.val_r2_y = R2Score(multioutput="uniform_average")
        self.val_pearson_y = PearsonCorrCoef()
        self.val_ev_y = ExplainedVariance(multioutput="uniform_average")

        self.test_mae_y = MeanAbsoluteError()
        self.test_r2_y = R2Score(multioutput="uniform_average")
        self.test_pearson_y = PearsonCorrCoef()
        self.test_ev_y = ExplainedVariance(multioutput="uniform_average")

        self.enable_sampled_eval = enable_sampled_eval

    # -------- Training / Val / Test --------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        B = y.size(0)

        t = torch.rand(B, device=self.device)
        v_hat, v_star = self.model(y, t, drug, prot)
        loss = F.mse_loss(v_hat, v_star)

        # # ---- Per-step diagnostic logs ----
        # # Sample t and z as usual
        # z = torch.randn_like(y)
        # with torch.no_grad():
        #     batch_size = y.size(0)
        #     self.log(
        #         "batch_mean_abs_y",
        #         y.abs().mean(),
        #         on_step=True,
        #         on_epoch=False,
        #         batch_size=batch_size,
        #     )
        #     self.log(
        #         "batch_mean_abs_y_minus_z",
        #         (y - z).abs().mean(),
        #         on_step=True,
        #         on_epoch=False,
        #         batch_size=batch_size,
        #     )
        #     self.log(
        #         "batch_mean_t",
        #         t.mean(),
        #         on_step=True,
        #         on_epoch=False,
        #         batch_size=batch_size,
        #     )

        # ---- Core loss log ----
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # Update metrics (epoch-level only)
        pred = v_hat.squeeze(-1)
        targ = v_star.squeeze(-1)
        self.train_mae.update(pred, targ)
        self.train_r2.update(pred, targ)
        self.train_pearson.update(pred, targ)
        self.train_ev.update(pred, targ)

        return loss

    def on_train_epoch_end(self):
        # compute & log once per epoch; then reset
        try:
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
        finally:
            self.train_mae.reset()
            self.train_r2.reset()
            self.train_pearson.reset()
            self.train_ev.reset()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        B = y.size(0)

        # Sample t (used for both forward and diagnostics)
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

        # --- metrics (epoch only) ---
        pred = v_hat.squeeze(-1)
        targ = v_star.squeeze(-1)
        self.val_mae.update(pred, targ)
        self.val_r2.update(pred, targ)
        self.val_pearson.update(pred, targ)
        self.val_ev.update(pred, targ)

        if self.enable_sampled_eval:
            with torch.no_grad():
                Y = self.sample_n(drug, prot)  # (B, K)
                y_mean = Y.mean(dim=1, keepdim=True)  # (B, 1)
            self.val_mae_y.update(y_mean, y)
            self.val_r2_y.update(y_mean, y)
            self.val_pearson_y.update(y_mean, y)
            self.val_ev_y.update(y_mean, y)

        # --- diagnostics (always) ---
        # with torch.no_grad():
        #     batch_size = y.size(0)
        #     self.log(
        #         "val_batch_mean_abs_y",
        #         y.abs().mean(),
        #         on_step=True,
        #         on_epoch=False,
        #         batch_size=batch_size,
        #     )
        #     self.log(
        #         "val_batch_mean_abs_y_minus_z",
        #         (y - z).abs().mean(),
        #         on_step=True,
        #         on_epoch=False,
        #         batch_size=batch_size,
        #     )
        #     self.log(
        #         "val_batch_mean_t",
        #         t.mean(),
        #         on_step=True,
        #         on_epoch=False,
        #         batch_size=batch_size,
        #     )

        return loss

    def on_validation_epoch_end(self):
        try:
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
        finally:
            self.val_mae.reset()
            self.val_r2.reset()
            self.val_pearson.reset()
            self.val_ev.reset()
        if self.enable_sampled_eval:
            try:
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
            finally:
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

        pred = v_hat.squeeze(-1)
        targ = v_star.squeeze(-1)
        self.test_mae.update(pred, targ)
        self.test_r2.update(pred, targ)
        self.test_pearson.update(pred, targ)
        self.test_ev.update(pred, targ)
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
            self.log(
                "test_pearson_y",
                self.test_pearson_y.compute(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "test_ev_y",
                self.test_ev_y.compute(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.test_mae_y.reset()
            self.test_r2_y.reset()
            self.test_pearson_y.reset()
            self.test_ev_y.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }

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
