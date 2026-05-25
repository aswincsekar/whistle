"""Evaluation: ROC/DET curves, FAR-per-hour, FRR at operating point.

For wake words the natural unit isn't a frame-level FPR but *false accepts per
hour of continuous audio*. We compute it via the stream detector on the test
manifest's negatives (assumed to be ~1s windows; for true continuous audio
support, point this at a long-audio negatives corpus).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, repo_root
from .data.dataset import WakeWordDataset, load_manifest
from .features import FeatureConfig, LogMelSpectrogram
from .models import build_model
from .train import _positive_logit, _device


def _load_ckpt(model, ckpt_path: str, use_ema: bool = True) -> None:
    blob = torch.load(ckpt_path, map_location="cpu")
    state = blob["ema_state"] if (use_ema and "ema_state" in blob) else blob["model_state"]
    # EMA state only contains float params; fall back to model state for the rest.
    model_state = model.state_dict()
    model_state.update({k: v for k, v in state.items() if k in model_state})
    model.load_state_dict(model_state, strict=False)


def ckpt_config(ckpt_path: str):
    """Return the config that was saved alongside the checkpoint.

    Lets callers rebuild the model at the right size without needing the
    on-disk config to match the trained run.
    """
    from types import SimpleNamespace
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = blob.get("config")
    if raw is None:
        return None
    def _ns(o):
        if isinstance(o, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in o.items()})
        if isinstance(o, list):
            return [_ns(v) for v in o]
        return o
    cfg = _ns(raw)
    cfg._raw = raw  # type: ignore[attr-defined]
    return cfg


def eval_checkpoint(cfg_path: str, ckpt_path: str, split: str = "test", use_ema: bool = True) -> dict:
    # Use the checkpoint's saved config for the *model* (so tau / mel params match
    # the trained weights), but keep paths / eval knobs from the on-disk config so
    # the user can point eval at a different manifest dir.
    on_disk = load_config(cfg_path)
    ckpt_cfg = ckpt_config(ckpt_path)
    cfg = ckpt_cfg if ckpt_cfg is not None else on_disk
    cfg.data = on_disk.data            # paths come from the local file
    cfg.eval = on_disk.eval            # sweep + target_far_per_hour too
    root = repo_root()
    device = _device()

    samples = load_manifest(root / cfg.data.manifest_dir / f"{split}.jsonl")
    ds = WakeWordDataset(samples, cfg.audio.sample_rate, cfg.audio.window_seconds,
                         augmenter=None, training=False, seed=0)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=2)

    feat_cfg = FeatureConfig.from_cfg(cfg)
    frontend = LogMelSpectrogram(feat_cfg).to(device).eval()
    model = build_model(cfg).to(device).eval()
    _load_ckpt(model, ckpt_path, use_ema=use_ema)

    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for audio, y in loader:
            audio = audio.to(device, non_blocking=True)
            mel = frontend(audio)
            logits = _positive_logit(model(mel))
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(y.tolist())
    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    # Sweep thresholds
    sweep_start, sweep_stop, sweep_step = cfg.eval.threshold_sweep
    thresholds = np.arange(sweep_start, sweep_stop + 1e-9, sweep_step)
    n_pos = int(labels_arr.sum()); n_neg = int((labels_arr == 0).sum())
    total_neg_hours = n_neg * cfg.audio.window_seconds / 3600.0

    curves = []
    for t in thresholds:
        pred = (scores_arr >= t).astype(int)
        tp = int(((pred == 1) & (labels_arr == 1)).sum())
        fp = int(((pred == 1) & (labels_arr == 0)).sum())
        fn = int(((pred == 0) & (labels_arr == 1)).sum())
        frr = fn / max(1, n_pos)
        far_per_hour = fp / max(1e-9, total_neg_hours)
        curves.append({"threshold": float(t), "frr": frr, "far_per_hour": far_per_hour, "tp": tp, "fp": fp, "fn": fn})

    target = cfg.eval.target_far_per_hour
    # Operating point: smallest FRR where FAR/h <= target.
    candidates = [c for c in curves if c["far_per_hour"] <= target]
    op = min(candidates, key=lambda c: c["frr"]) if candidates else None

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels_arr, scores_arr)) if n_pos else float("nan")
    except Exception:
        auc = float("nan")

    summary = {
        "checkpoint": ckpt_path,
        "split": split,
        "auc": auc,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "neg_hours": round(total_neg_hours, 3),
        "operating_point": op,
        "curve": curves,
    }
    out_path = Path(ckpt_path).with_name(f"eval_{split}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[whistle] eval saved -> {out_path}")
    if op:
        print(f"  AUC={auc:.4f}  FRR={op['frr']:.3f} @ FAR={op['far_per_hour']:.3f}/h  (thr={op['threshold']:.2f})")
    return summary
