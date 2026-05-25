"""Log-mel spectrogram front-end.

Designed to be:
  1. Cheap (torch-only ops, no torchaudio kaldi C++ deps at inference).
  2. Reproducible bit-for-bit in mobile code (we publish the mel filterbank
     numerically; iOS/Android compute STFT + matmul + log).
  3. Exportable to ONNX / TFLite as part of the model graph if desired.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 7600.0
    log_offset: float = 1e-6
    per_utt_normalize: bool = True

    @property
    def win_length(self) -> int:
        return int(self.sample_rate * self.win_length_ms / 1000.0)

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.hop_length_ms / 1000.0)

    @classmethod
    def from_cfg(cls, cfg) -> "FeatureConfig":
        f = cfg.features
        return cls(
            sample_rate=cfg.audio.sample_rate,
            n_fft=f.n_fft,
            win_length_ms=f.win_length_ms,
            hop_length_ms=f.hop_length_ms,
            n_mels=f.n_mels,
            fmin=f.fmin,
            fmax=f.fmax,
            log_offset=f.log_offset,
            per_utt_normalize=f.per_utt_normalize,
        )


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """HTK-style triangular mel filterbank — matches librosa's `htk=True`.

    Returns: (n_mels, n_fft//2 + 1).
    """
    mel_min, mel_max = _hz_to_mel(np.array([fmin])), _hz_to_mel(np.array([fmax]))
    mel_points = np.linspace(mel_min[0], mel_max[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_l, f_c, f_r = bins[m - 1], bins[m], bins[m + 1]
        if f_c == f_l:
            f_c = f_l + 1
        if f_r == f_c:
            f_r = f_c + 1
        for k in range(f_l, f_c):
            fb[m - 1, k] = (k - f_l) / max(1, (f_c - f_l))
        for k in range(f_c, f_r):
            fb[m - 1, k] = (f_r - k) / max(1, (f_r - f_c))
    return fb


class LogMelSpectrogram(nn.Module):
    """Audio waveform -> log-mel spectrogram, torch-native, ONNX-exportable."""

    def __init__(self, cfg: FeatureConfig):
        super().__init__()
        self.cfg = cfg
        window = torch.hann_window(cfg.win_length, periodic=True)
        # Zero-pad window to n_fft so torch.stft accepts it directly.
        if cfg.win_length < cfg.n_fft:
            pad = cfg.n_fft - cfg.win_length
            left = pad // 2
            right = pad - left
            window = torch.nn.functional.pad(window, (left, right))
        self.register_buffer("window", window, persistent=False)

        fb = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
        self.register_buffer("mel_fb", torch.from_numpy(fb), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T) or (T,) waveform in [-1, 1] -> (B, n_mels, frames) log-mel."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Manual centered STFT — keeps the graph ONNX-friendly and avoids the
        # `torch.stft` resize warning that fires when the runtime can't preallocate
        # the output buffer in advance.
        n_fft = self.cfg.n_fft
        hop = self.cfg.hop_length
        pad = n_fft // 2
        x_pad = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
        # Frame: (B, num_frames, n_fft)
        frames = x_pad.unfold(dimension=-1, size=n_fft, step=hop)
        # Apply window and rFFT
        windowed = frames * self.window
        spec = torch.fft.rfft(windowed, n=n_fft, dim=-1)   # (B, T, F)
        spec = spec.transpose(-1, -2)                       # (B, F, T)
        power = spec.real.pow(2) + spec.imag.pow(2)         # (B, F, T)
        mel = torch.matmul(self.mel_fb, power)        # (B, n_mels, T)
        log_mel = torch.log(mel + self.cfg.log_offset)

        if self.cfg.per_utt_normalize:
            mu = log_mel.mean(dim=(-1, -2), keepdim=True)
            sd = log_mel.std(dim=(-1, -2), keepdim=True).clamp_min(1e-5)
            log_mel = (log_mel - mu) / sd
        return log_mel


def num_frames(num_samples: int, hop: int) -> int:
    """Number of mel frames for a centered STFT."""
    return int(math.floor(num_samples / hop)) + 1
