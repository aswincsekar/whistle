"""Small classifier on top of openWakeWord's 96-dim speech embedding.

Input shape  : (B, N=15, D=96) — N embeddings per ~1.96 s audio window.
Output shape : (B, num_classes) raw logits.

The hard work — phoneme/speaker/loudness invariance — is already baked into the
pretrained embedding model. The classifier just needs to learn the temporal
pattern of "Hey Bubba" in this rich feature space, so we go small:

  emb_dim -> Conv1d(d=128, k=3) -> BN -> ReLU -> Dropout
          -> Conv1d(d=128, k=3) -> BN -> ReLU -> Dropout
          -> Conv1d(d=64,  k=3) -> BN -> ReLU
          -> AdaptiveAvgPool1d(1) -> Linear(2)

~70k params. Roughly the same size as our BC-ResNet τ=3, but with the
representational power of a much larger pretrained front-end in front of it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OWWClassifier(nn.Module):
    def __init__(self, n_emb: int = 15, emb_dim: int = 96,
                 num_classes: int = 2, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.body = nn.Sequential(
            nn.Conv1d(emb_dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, emb_dim) -> (B, num_classes)"""
        # Treat embedding dim as channels for Conv1D over time.
        x = x.transpose(1, 2)             # (B, emb_dim, N)
        x = self.body(x).squeeze(-1)       # (B, hidden/2)
        return self.head(x)

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_owww_model(cfg) -> nn.Module:
    """Construct OWWClassifier from a config namespace."""
    return OWWClassifier(
        n_emb=getattr(cfg.model, "n_emb", 15),
        emb_dim=getattr(cfg.model, "emb_dim", 96),
        num_classes=cfg.model.num_classes,
        hidden=getattr(cfg.model, "hidden", 128),
        dropout=getattr(cfg.model, "dropout", 0.2),
    )
