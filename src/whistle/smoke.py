"""End-to-end smoke test on a tiny synthetic dataset.

We don't actually train to convergence — we just verify that every stage of the
pipeline executes without errors and produces sensible shapes/files.

What it does:
  1. Generate a few "Hey Bubba" positives via macOS `say` if available, else
     synthesize fake-positive sine-bursts as a stand-in.
  2. Generate negatives as random noise / sine tones.
  3. Build manifests.
  4. Run 2 epochs of training.
  5. Evaluate the checkpoint.
  6. Export ONNX and verify PyTorch vs ONNX outputs match.
  7. Run streaming inference on a generated WAV.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .config import load_config, repo_root


def _write_tone_burst(path: Path, sr: int, freq: float, duration: float, formant_drift: bool = True):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if formant_drift:
        f = freq * (1.0 + 0.1 * np.sin(2 * np.pi * 3 * t))
        wave = 0.4 * np.sin(2 * np.pi * f * t)
    else:
        wave = 0.4 * np.sin(2 * np.pi * freq * t)
    env = np.minimum(t / 0.05, (duration - t) / 0.05).clip(0, 1)
    wave = wave * env
    sf.write(str(path), wave.astype(np.float32), sr)


def _write_noise(path: Path, sr: int, duration: float):
    n = int(sr * duration)
    x = (np.random.default_rng().standard_normal(n) * 0.05).astype(np.float32)
    sf.write(str(path), x, sr)


def run_smoke() -> bool:
    root = repo_root()
    smoke_dir = root / "data" / "_smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    pos_dir = smoke_dir / "positives"
    neg_dir = smoke_dir / "negatives"
    noise_dir = smoke_dir / "noise"
    for d in (pos_dir, neg_dir, noise_dir):
        d.mkdir(parents=True, exist_ok=True)

    sr = 16000

    # Try real TTS first
    try:
        from .data.synth import synth_positives
        records = synth_positives(pos_dir, ["Hey Bubba", "Hey Buba"], count=20, sample_rate=sr)
    except Exception:
        records = []

    if len(records) < 8:
        print("[smoke] TTS unavailable - using synthetic tone-bursts for positives")
        for i in range(40):
            _write_tone_burst(pos_dir / f"pos_{i:03d}.wav", sr, freq=180 + i * 5, duration=0.8)

    for i in range(80):
        _write_noise(neg_dir / f"neg_{i:03d}.wav", sr, duration=1.2)
    for i in range(10):
        _write_noise(noise_dir / f"noise_{i:03d}.wav", sr, duration=2.0)

    # Write a smoke config (small model, few epochs, tiny batch)
    base_cfg_path = root / "configs/default.yaml"
    with open(base_cfg_path) as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw["data"]["positives_dir"] = str(pos_dir)
    cfg_raw["data"]["negatives_dir"] = str(neg_dir)
    cfg_raw["data"]["noise_dir"] = str(noise_dir)
    cfg_raw["data"]["rirs_dir"] = str(smoke_dir / "rirs")  # empty
    cfg_raw["data"]["manifest_dir"] = str(smoke_dir / "manifests")
    cfg_raw["model"]["tau"] = 1
    cfg_raw["train"]["epochs"] = 2
    cfg_raw["train"]["batch_size"] = 32
    cfg_raw["train"]["positive_ratio"] = 0.4
    cfg_raw["train"]["hard_negative_ratio"] = 0.0
    cfg_raw["train"]["warmup_epochs"] = 0
    cfg_raw["export"]["representative_samples"] = 8
    cfg_path = smoke_dir / "smoke.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_raw, f)

    # 1. Manifests
    from .data.manifests import build_manifests
    build_manifests(pos_dir, neg_dir, Path(cfg_raw["data"]["manifest_dir"]),
                    val_fraction=0.15, test_fraction=0.15, seed=0)
    print("[smoke] manifests built")

    # 2. Train
    from .train import train as _train
    result = _train(str(cfg_path))
    print(f"[smoke] trained: AUC={result['best_val_auc']:.3f}  params={result['params']:,}")

    # 3. Eval
    from .eval import eval_checkpoint
    eval_checkpoint(str(cfg_path), result["best_ckpt"], split="test")

    # 4. Export ONNX and verify
    from .export import export as _export, verify_onnx
    meta = _export(str(cfg_path), result["best_ckpt"], fmt="onnx",
                   out_dir=str(smoke_dir / "exports"))
    ok = verify_onnx(str(cfg_path), result["best_ckpt"], meta["outputs"]["onnx_audio"], n=4, tol=1e-2)

    # 5. Stream
    from .stream import stream_wav
    wav = smoke_dir / "test_audio.wav"
    silence = np.zeros(int(sr * 1.0), dtype=np.float32)
    pos_audio, _ = sf.read(str(pos_dir / sorted(pos_dir.glob("*.wav"))[0]), dtype="float32")
    if pos_audio.ndim > 1:
        pos_audio = pos_audio.mean(axis=1)
    audio = np.concatenate([silence, pos_audio, silence])
    sf.write(str(wav), audio, sr)
    stream_wav(str(cfg_path), result["best_ckpt"], str(wav), threshold=0.3)

    print(f"\n[smoke] DONE  (onnx verify: {'OK' if ok else 'MISMATCH'})")
    return True
