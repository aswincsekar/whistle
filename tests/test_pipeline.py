"""Lightweight pytest sanity checks (no training)."""
from __future__ import annotations

import numpy as np
import torch

from whistle.config import load_config
from whistle.features import FeatureConfig, LogMelSpectrogram, mel_filterbank
from whistle.models import build_model
from whistle.models.wakeword import MelClassifier


def test_mel_filterbank_shape():
    fb = mel_filterbank(16000, 512, 40, 20, 7600)
    assert fb.shape == (40, 257)
    assert fb.sum(axis=1).min() > 0  # every mel bin has non-zero energy


def test_logmel_forward():
    cfg = FeatureConfig()
    f = LogMelSpectrogram(cfg).eval()
    x = torch.randn(2, 19200)
    y = f(x)
    assert y.shape == (2, 40, 121)
    assert not torch.isnan(y).any()


def test_model_forward(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.model.tau = 1
    model = build_model(cfg).eval()
    mel = torch.zeros(2, cfg.features.n_mels, 121)
    out = model(mel)
    assert out.shape == (2, 2)


def test_end_to_end_module():
    cfg = load_config("configs/default.yaml")
    cfg.model.tau = 1
    m = MelClassifier(cfg).eval()
    x = torch.randn(1, 19200)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 2)
