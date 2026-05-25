"""BC-ResNet — Broadcasted-Residual ConvNet for keyword spotting.

Kim, B. et al. "Broadcasted Residual Learning for Efficient Keyword Spotting",
Interspeech 2021. https://arxiv.org/abs/2106.04140

Why this model:
  - State of the art for KWS at <200k params (tau=8 hits ~98% Speech Commands).
  - Factorizes 2-D depthwise conv into a frequency-depthwise + a time-depthwise
    branch. Cheap and INT8-friendly (no exotic ops).
  - At inference the time path is a depthwise 1-D conv along the time axis —
    same op the mobile runtime sees.

We keep it stateless (windowed classifier). For streaming we feed overlapping
windows from a ring buffer in the runtime layer; the model itself is causal-free
which makes ONNX/TFLite export trivial.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SubSpectralNorm(nn.Module):
    """Sub-spectral norm: split frequency axis into S groups and BN each.
    Helps tiny KWS models — gives the network distinct stats per frequency band.
    """

    def __init__(self, num_features: int, num_sub: int = 5):
        super().__init__()
        self.num_sub = num_sub
        self.bn = nn.BatchNorm2d(num_features * num_sub)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        s = self.num_sub
        # Make sure freq divides into S groups; pad if needed.
        pad = (s - f % s) % s
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            f = f + pad
        x = x.view(b, c, s, f // s, t).permute(0, 1, 2, 3, 4).contiguous()
        x = x.view(b, c * s, f // s, t)
        x = self.bn(x)
        x = x.view(b, c, s, f // s, t).view(b, c, f, t)
        if pad:
            x = x[:, :, :-pad, :]
        return x


class _TransitionBlock(nn.Module):
    """Channel-changing block: 1x1 pointwise + depthwise freq conv + transition."""

    def __init__(self, in_c: int, out_c: int, stride_f: int = 1, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.f_branch = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            # Depthwise conv along frequency
            nn.Conv2d(out_c, out_c, kernel_size=(3, 1), stride=(stride_f, 1),
                      padding=(1, 0), groups=out_c, bias=False),
            _SubSpectralNorm(out_c, num_sub=5),
        )
        # Broadcasted residual: average over freq -> depthwise time conv -> broadcast back
        self.t_branch = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(1, 3), padding=(0, dilation),
                      dilation=(1, dilation), groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.stride_f = stride_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.f_branch(x)                         # (B, C', F', T)
        # Broadcasted residual: average freq -> time conv -> broadcast
        z = y.mean(dim=2, keepdim=True)              # (B, C', 1, T)
        z = self.t_branch(z)                         # (B, C', 1, T)
        out = self.act(y + z + self.dropout(y))
        return out


class _NormalBlock(nn.Module):
    """Identity-residual variant (no channel change)."""

    def __init__(self, c: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.f_branch = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(3, 1), padding=(1, 0), groups=c, bias=False),
            _SubSpectralNorm(c, num_sub=5),
        )
        self.t_branch = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, 3), padding=(0, dilation),
                      dilation=(1, dilation), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=1, bias=False),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.f_branch(x)
        z = y.mean(dim=2, keepdim=True)
        z = self.t_branch(z)
        return self.act(x + y + z + self.dropout(y))


class BCResNet(nn.Module):
    """BC-ResNet with width factor tau."""

    # Channel plan from the paper, scaled by tau.
    _BASE_C = [16, 8, 12, 16, 20]
    # (block_type, repeat, dilation, stride_f)
    _STAGES = [
        ("transition", 1, 1, 1),
        ("normal",     1, 1, 1),
        ("transition", 1, 2, 2),
        ("normal",     1, 2, 1),
        ("transition", 1, 4, 2),
        ("normal",     2, 4, 1),
        ("transition", 1, 8, 1),
        ("normal",     2, 8, 1),
    ]

    def __init__(self, n_mels: int = 40, num_classes: int = 2, tau: int = 2, dropout: float = 0.1):
        super().__init__()
        c = [max(8, int(round(x * tau))) for x in self._BASE_C]
        self.stem = nn.Sequential(
            nn.Conv2d(1, c[0], kernel_size=(5, 5), stride=(2, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )

        layers: list[nn.Module] = []
        cur = c[0]
        plan = [c[1], c[2], c[3], c[4]]   # output channels after each "transition"
        i_plan = 0
        for kind, repeat, dil, stride in self._STAGES:
            if kind == "transition":
                out = plan[i_plan]; i_plan = min(i_plan + 1, len(plan) - 1)
                layers.append(_TransitionBlock(cur, out, stride_f=stride, dilation=dil, dropout=dropout))
                cur = out
            else:
                for _ in range(repeat):
                    layers.append(_NormalBlock(cur, dilation=dil, dropout=dropout))
        self.blocks = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(cur, cur, kernel_size=(5, 5), padding=(0, 2), groups=cur, bias=False),
            nn.BatchNorm2d(cur),
            nn.ReLU(inplace=True),
            nn.Conv2d(cur, num_classes, kernel_size=1),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T) — log-mel features."""
        x = mel.unsqueeze(1)                      # (B, 1, F, T)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)                          # (B, K, F', T')
        # Global average over remaining freq+time
        x = x.mean(dim=(2, 3))                    # (B, K)
        return x

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg) -> nn.Module:
    name = cfg.model.name
    if name == "bc_resnet":
        return BCResNet(
            n_mels=cfg.features.n_mels,
            num_classes=cfg.model.num_classes,
            tau=cfg.model.tau,
            dropout=getattr(cfg.model, "dropout", 0.1),
        )
    raise ValueError(f"unknown model {name}")
