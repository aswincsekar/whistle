"""Composite module: raw audio -> mel -> classifier.

We expose two variants for export:
  * MelClassifier (audio in, logits out) — single self-contained graph.
  * Backbone only (mel in, logits out) — useful when the mobile runtime
    already computes mel features (e.g. via Accelerate/NNAPI).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..features import LogMelSpectrogram, FeatureConfig
from .bc_resnet import build_model


class MelClassifier(nn.Module):
    """End-to-end module: float32 PCM (B, T) -> logits (B, K)."""

    def __init__(self, cfg):
        super().__init__()
        self.feat_cfg = FeatureConfig.from_cfg(cfg)
        self.frontend = LogMelSpectrogram(self.feat_cfg)
        self.backbone = build_model(cfg)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.frontend(audio)
        return self.backbone(mel)
