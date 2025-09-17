import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GlobalAttention, global_mean_pool
from torchmetrics import MeanAbsoluteError, R2Score, PearsonCorrCoef, ExplainedVariance


# ---- Small helpers ----
def mlp(sizes, act=nn.SiLU, dropout=0.0, ln=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            if ln:
                layers += [nn.LayerNorm(sizes[i + 1])]
            layers += [act()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class DrugEncoder(nn.Module):
    def __init__(
        self, node_in_dim: int, hidden: int = 256, layers: int = 3, dropout: float = 0.1
    ):
        super().__init__()
        # if you have edge features, pass an edge MLP to GINE
        self.gine_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.gine_layers.append(GINEConv(nn=mlp([hidden, hidden], dropout=dropout)))
            self.norms.append(nn.LayerNorm(hidden))
        self.in_proj = nn.Linear(node_in_dim, hidden)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        h = self.act(self.in_proj(batch.x))  # (N, H)
        for conv, norm in zip(self.gine_layers, self.norms):
            h_res = h
            h = conv(h, batch.edge_index)  # add edge_attr if you have it
            h = norm(h)
            h = self.act(h)
            h = self.dropout(h)
            h = h + h_res
        g = global_mean_pool(h, batch.batch)  # (B, H)
        return g


class ProteinEncoder(nn.Module):
    def __init__(self, prot_in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = mlp([prot_in_dim, hidden, hidden], dropout=dropout)

    def forward(self, prot_feats):
        return self.net(prot_feats)


class DrugProteinGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        prot_in_dim: int,
        hidden: int = 256,
        gnn_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.drug_enc = DrugEncoder(
            node_in_dim, hidden=hidden, layers=gnn_layers, dropout=dropout
        )
        self.prot_enc = ProteinEncoder(prot_in_dim, hidden=hidden, dropout=dropout)
        self.fuse = mlp([2 * hidden, hidden], dropout=dropout)
        self.head = nn.Linear(hidden, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, batch, protein_feats=None):
        if protein_feats is None:
            protein_feats = batch.protein
        gd = self.drug_enc(batch)  # (B, H)
        gp = self.prot_enc(protein_feats)  # (B, H)
        h = self.fuse(torch.cat([gd, gp], dim=-1))
        y = self.head(h)  # (B, 1)
        return y


class DrugProteinGNNPL(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        # metrics
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

    def _step(self, batch, stage: str):
        y_hat = self.model(batch, batch.protein).squeeze(-1)  # (B,)
        y_tgt = batch.y.squeeze(-1).float()
        y_hat = y_hat.float()
        loss = F.mse_loss(y_hat, y_tgt)

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )
        metrics = {
            "train": (self.train_mae, self.train_r2, self.train_pearson, self.train_ev),
            "val": (self.val_mae, self.val_r2, self.val_pearson, self.val_ev),
            "test": (self.test_mae, self.test_r2, self.test_pearson, self.test_ev),
        }[stage]
        for m in metrics:
            m.update(y_hat, y_tgt)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def on_train_epoch_end(self):
        self.log("train_mae", self.train_mae.compute(), prog_bar=True, sync_dist=True)
        self.log("train_r2", self.train_r2.compute(), prog_bar=True, sync_dist=True)
        self.log(
            "train_pearson", self.train_pearson.compute(), prog_bar=True, sync_dist=True
        )
        self.log("train_ev", self.train_ev.compute(), prog_bar=True, sync_dist=True)
        self.train_mae.reset()
        self.train_r2.reset()
        self.train_pearson.reset()
        self.train_ev.reset()

    def on_validation_epoch_end(self):
        self.log("val_mae", self.val_mae.compute(), prog_bar=True, sync_dist=True)
        self.log("val_r2", self.val_r2.compute(), prog_bar=True, sync_dist=True)
        self.log(
            "val_pearson", self.val_pearson.compute(), prog_bar=True, sync_dist=True
        )
        self.log("val_ev", self.val_ev.compute(), prog_bar=True, sync_dist=True)
        self.val_mae.reset()
        self.val_r2.reset()
        self.val_pearson.reset()
        self.val_ev.reset()

    def on_test_epoch_end(self):
        self.log("test_mae", self.test_mae.compute(), prog_bar=True, sync_dist=True)
        self.log("test_r2", self.test_r2.compute(), prog_bar=True, sync_dist=True)
        self.log(
            "test_pearson", self.test_pearson.compute(), prog_bar=True, sync_dist=True
        )
        self.log("test_ev", self.test_ev.compute(), prog_bar=True, sync_dist=True)
        self.test_mae.reset()
        self.test_r2.reset()
        self.test_pearson.reset()
        self.test_ev.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }
