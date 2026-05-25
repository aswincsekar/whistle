"""Train the wake-word classifier.

Design choices worth flagging:
  * Loss = BCEWithLogitsLoss on a single output logit (positive class).
    Two-class CE collapses to this anyway and BCE gives more direct
    control over thresholding at inference.
  * Hard-negative mining: at end of every Nth epoch, run the model over all
    training negatives, take the top-K by predicted probability that
    weren't actual positives, and feed them back via BalancedBatchSampler.
  * EMA shadow weights — small models with aggressive augmentation are noisy;
    EMA gives a meaningful win at zero serving cost.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import load_config, repo_root
from .data.augment import AugmentConfig, Augmenter, load_audio_pool
from .data.dataset import BalancedBatchSampler, Sample, WakeWordDataset, load_manifest
from .features import FeatureConfig, LogMelSpectrogram
from .models import build_model


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    """Standard EMA shadow weights — used for eval and export."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
              smoothing: float = 0.0) -> torch.Tensor:
    if smoothing > 0:
        targets = targets * (1 - smoothing) + 0.5 * smoothing
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.where(targets > 0.5, p, 1 - p)
    return ((1 - pt) ** gamma * ce).mean()


@dataclass
class TrainState:
    epoch: int = 0
    best_val_auc: float = 0.0
    best_path: str = ""
    last_path: str = ""


def _positive_logit(logits: torch.Tensor) -> torch.Tensor:
    """Pull the wake-word logit out regardless of whether the model emits
    1 logit (BCE) or 2 logits (softmax-style)."""
    if logits.dim() == 1:
        return logits
    if logits.size(-1) == 1:
        return logits.squeeze(-1)
    return logits[..., 1] - logits[..., 0]   # log-ratio, monotone in P(positive)


@torch.no_grad()
def evaluate(model: nn.Module, frontend: nn.Module, loader: DataLoader,
             device: torch.device) -> dict:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    for audio, y in loader:
        audio = audio.to(device, non_blocking=True)
        mel = frontend(audio)
        logits = model(mel)
        s = torch.sigmoid(_positive_logit(logits)).detach().cpu().numpy()
        scores.extend(s.tolist())
        labels.extend(y.tolist())
    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    # AUC
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = float(roc_auc_score(labels_arr, scores_arr)) if labels_arr.sum() > 0 else float("nan")
        fpr, tpr, thr = roc_curve(labels_arr, scores_arr)
    except Exception:
        auc, fpr, tpr, thr = float("nan"), np.array([]), np.array([]), np.array([])

    # FRR @ low-FAR operating point
    frr_at = float("nan")
    if fpr.size:
        target_fpr = 0.01
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
        frr_at = float(1.0 - tpr[idx])

    return {
        "auc": auc,
        "frr_at_1pct_fpr": frr_at,
        "n_pos": int(labels_arr.sum()),
        "n_neg": int((labels_arr == 0).sum()),
        "scores": scores_arr,
        "labels": labels_arr,
    }


def mine_hard_negatives(
    model: nn.Module, frontend: nn.Module, dataset: WakeWordDataset,
    device: torch.device, top_k: int = 1024, batch_size: int = 128
) -> list[int]:
    """Score every negative in `dataset` and return indices with highest model
    probability — these become the next epoch's hard negatives."""
    model.eval()
    neg_idx = [i for i, s in enumerate(dataset.samples) if s.label == 0]
    if not neg_idx:
        return []
    loader = DataLoader(
        torch.utils.data.Subset(dataset, neg_idx),
        batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True,
    )
    scores: list[float] = []
    with torch.no_grad():
        for audio, _ in loader:
            audio = audio.to(device, non_blocking=True)
            mel = frontend(audio)
            logits = model(mel)
            scores.extend(torch.sigmoid(_positive_logit(logits)).cpu().numpy().tolist())
    order = np.argsort(scores)[::-1][:top_k]
    return [neg_idx[int(i)] for i in order]


def train(cfg_path: str, init_from: str | None = None,
          lr_override: float | None = None, epochs_override: int | None = None) -> dict:
    """Train the wake-word model.

    If `init_from` is given, load that checkpoint's EMA weights as the starting
    point. Typical use: fine-tune a converged model on additional/different
    data with a smaller learning rate. Pair with `lr_override` (e.g. 5e-4 instead
    of 3e-3) and `epochs_override` (e.g. 20 instead of 60).
    """
    cfg = load_config(cfg_path)
    if lr_override is not None:
        cfg.train.lr = float(lr_override)
    if epochs_override is not None:
        cfg.train.epochs = int(epochs_override)
    root = repo_root()
    _seed_all(cfg.train.seed)
    device = _device()
    print(f"[whistle] device = {device}, model = {cfg.model.name}, tau = {cfg.model.tau}")
    if init_from:
        print(f"[whistle] init_from = {init_from}  lr = {cfg.train.lr}  epochs = {cfg.train.epochs}")

    # Manifests
    manifest_dir = root / cfg.data.manifest_dir
    train_samples = load_manifest(manifest_dir / "train.jsonl")
    val_samples = load_manifest(manifest_dir / "val.jsonl")

    # Augmentation pools
    noise_pool = load_audio_pool(root / cfg.data.noise_dir, cfg.audio.sample_rate, max_files=400)
    rir_pool = load_audio_pool(root / cfg.data.rirs_dir, cfg.audio.sample_rate, max_files=200)
    print(f"[whistle] noise samples = {len(noise_pool)}, RIRs = {len(rir_pool)}")
    augmenter = Augmenter(AugmentConfig.from_cfg(cfg), noise_pool, rir_pool, cfg.audio.sample_rate)

    # Datasets
    train_ds = WakeWordDataset(
        train_samples, cfg.audio.sample_rate, cfg.audio.window_seconds,
        augmenter=augmenter, training=True, seed=cfg.train.seed,
    )
    val_ds = WakeWordDataset(
        val_samples, cfg.audio.sample_rate, cfg.audio.window_seconds,
        augmenter=None, training=False, seed=0,
    )

    # Balanced sampler
    labels = [s.label for s in train_samples]
    sampler = BalancedBatchSampler(
        labels,
        batch_size=cfg.train.batch_size,
        positive_ratio=cfg.train.positive_ratio,
        hard_negative_ratio=cfg.train.hard_negative_ratio,
        seed=cfg.train.seed,
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, num_workers=2)

    # Model + front-end (front-end is also a Module, lives on device)
    feat_cfg = FeatureConfig.from_cfg(cfg)
    frontend = LogMelSpectrogram(feat_cfg).to(device)
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[whistle] params = {n_params:,}")

    # Optional: load EMA weights from a prior checkpoint (fine-tune mode).
    if init_from:
        blob = torch.load(init_from, map_location=device, weights_only=False)
        state = blob.get("ema_state") or blob.get("model_state")
        if state is None:
            raise ValueError(f"{init_from} has neither 'ema_state' nor 'model_state'")
        ms = model.state_dict()
        loaded = 0; skipped = 0
        for k, v in state.items():
            if k in ms and ms[k].shape == v.shape:
                ms[k] = v.clone(); loaded += 1
            else:
                skipped += 1
        model.load_state_dict(ms, strict=False)
        print(f"[whistle] loaded {loaded} tensors (skipped {skipped}) from {init_from}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    ema = EMA(model, decay=cfg.train.ema_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    ckpt_dir = root / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    run_dir = root / "runs" / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(run_dir))
    state = TrainState()

    for epoch in range(cfg.train.epochs):
        state.epoch = epoch
        model.train()
        t0 = time.time()
        ep_loss = 0.0; ep_n = 0
        for step, (audio, y) in enumerate(train_loader):
            audio = audio.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            with torch.amp.autocast(device_type=device.type, enabled=cfg.train.amp and device.type == "cuda"):
                mel = frontend(audio)
                logits = _positive_logit(model(mel))
                if cfg.train.loss == "focal":
                    loss = focal_bce(logits, y, gamma=cfg.train.focal_gamma,
                                     smoothing=cfg.train.label_smoothing)
                else:
                    targets = y * (1 - cfg.train.label_smoothing) + 0.5 * cfg.train.label_smoothing
                    loss = F.binary_cross_entropy_with_logits(logits, targets)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            ema.update(model)

            ep_loss += loss.item() * audio.size(0); ep_n += audio.size(0)

        sched.step()
        train_loss = ep_loss / max(1, ep_n)
        # Eval (use EMA weights)
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({**backup, **ema.state_dict()}, strict=False)
        metrics = evaluate(model, frontend, val_loader, device)
        model.load_state_dict(backup, strict=False)

        dt = time.time() - t0
        print(f"[ep {epoch:03d}] loss={train_loss:.4f}  val_auc={metrics['auc']:.4f}  "
              f"frr@1%fpr={metrics['frr_at_1pct_fpr']:.3f}  dt={dt:.1f}s")
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/auc", metrics["auc"], epoch)
        writer.add_scalar("val/frr_at_1pct_fpr", metrics["frr_at_1pct_fpr"], epoch)

        # Save
        state.last_path = str(ckpt_dir / "last.pt")
        torch.save(_checkpoint_blob(model, ema, cfg, metrics), state.last_path)
        if metrics["auc"] > state.best_val_auc:
            state.best_val_auc = metrics["auc"]
            state.best_path = str(ckpt_dir / "best.pt")
            torch.save(_checkpoint_blob(model, ema, cfg, metrics), state.best_path)

        # Hard-negative mining every 5 epochs after warmup
        if epoch >= max(3, cfg.train.warmup_epochs) and (epoch + 1) % 5 == 0:
            hard = mine_hard_negatives(model, frontend, train_ds, device, top_k=1024)
            sampler.hard_indices = hard
            print(f"[whistle] mined {len(hard)} hard negatives")

    writer.close()
    return {
        "best_val_auc": state.best_val_auc,
        "best_ckpt": state.best_path,
        "last_ckpt": state.last_path,
        "params": n_params,
    }


def _checkpoint_blob(model: nn.Module, ema: EMA, cfg, metrics: dict) -> dict:
    return {
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "config": cfg._raw,
        "metrics": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items() if k not in ("scores", "labels")},
    }
