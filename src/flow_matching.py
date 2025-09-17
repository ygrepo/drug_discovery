import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import pytorch_lightning as pl
from typing import Dict, Any, Optional

# (Assume FlowConfig, DrugProteinFlowMatching from previous message are imported)


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
        pred_num_samples: int = 50,  # K samples during predict_step
        pred_steps: Optional[int] = None,  # None -> cfg.steps
        pi_alpha: float = 0.05,  # 95% PI by default
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.model = DrugProteinFlowMatching(drug_input_dim, protein_input_dim, cfg)
        self.cfg = cfg
        self.pred_num_samples = pred_num_samples
        self.pred_steps = pred_steps
        self.pi_alpha = pi_alpha

    # -------- Training / Val / Test --------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        B = y.size(0)
        t = torch.rand(B, device=self.device)
        v_hat, v_star = self.model(y, t, drug, prot)
        loss = F.mse_loss(v_hat, v_star)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        B = y.size(0)
        t = torch.rand(B, device=self.device)
        v_hat, v_star = self.model(y, t, drug, prot)
        loss = F.mse_loss(v_hat, v_star)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        drug, prot, y = batch["drug"], batch["protein"], batch["y"]
        B = y.size(0)
        t = torch.rand(B, device=self.device)
        v_hat, v_star = self.model(y, t, drug, prot)
        loss = F.mse_loss(v_hat, v_star)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

    # -------- Inference helpers --------
    @torch.no_grad()
    def sample_n(
        self,
        drug: torch.Tensor,
        prot: torch.Tensor,
        n_samples: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Draw K samples y ~ p_theta(y|drug,prot).
        Returns tensor of shape (B, K).
        """
        self.model.eval()
        B = drug.size(0)
        K = n_samples or self.pred_num_samples
        S = steps or self.pred_steps or self.cfg.steps

        ys = []
        for _ in range(K):
            yk = self.model.sample(drug, prot, steps=S)  # (B,1)
            ys.append(yk)  # list of (B,1)
        Y = torch.cat(ys, dim=1)  # (B, K)
        return Y

    # Lightning predict API
    @torch.no_grad()
    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            'y_samples': (B, K),
            'y_mean': (B, 1),
            'y_std': (B, 1),
            'y_lo': (B, 1),           # PI lower (1 - alpha)
            'y_hi': (B, 1),           # PI upper
            'y_true': (B, 1) or None,
            'residual': (B, 1) or None
          }
        """
        drug, prot = batch["drug"], batch["protein"]
        Y = self.sample_n(drug, prot)  # (B, K)
        mean = Y.mean(dim=1, keepdim=True)  # (B,1)
        std = Y.std(dim=1, keepdim=True)  # (B,1)

        # symmetric PI under Gaussian assumption
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
        lo = mean - z * std
        hi = mean + z * std

        out = {
            "y_samples": Y,
            "y_mean": mean,
            "y_std": std,
            "y_lo": lo,
            "y_hi": hi,
        }

        if "y" in batch:
            y_true = batch["y"]
            out["y_true"] = y_true
            out["residual"] = mean - y_true

        return out
