"""Streaming wake-word detector.

  * `stream_wav(path, ...)` runs a sliding window over a WAV file.
  * `stream_mic(...)` does the same off a microphone (needs sounddevice).

Each window is `cfg.audio.window_seconds`, advanced by `cfg.audio.hop_seconds`.
We smooth the per-window probabilities with a short moving average and emit a
detection on the rising edge above a configurable threshold (with a cooldown).
"""
from __future__ import annotations

import collections
import queue
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import soundfile as sf
import torch

from .config import load_config, repo_root
from .eval import _load_ckpt, ckpt_config
from .features import FeatureConfig, LogMelSpectrogram
from .models import build_model
from .train import _device, _positive_logit


def _load_model(cfg_path: str, ckpt_path: str) -> tuple:
    # Prefer the config saved inside the checkpoint — that's the one whose
    # `model.tau`, mel parameters, etc. match the weights on disk. The on-disk
    # config file may have drifted (e.g. different tau for a later run).
    ckpt_cfg = ckpt_config(ckpt_path)
    cfg = ckpt_cfg if ckpt_cfg is not None else load_config(cfg_path)
    device = _device()
    feat_cfg = FeatureConfig.from_cfg(cfg)
    frontend = LogMelSpectrogram(feat_cfg).to(device).eval()
    model = build_model(cfg).to(device).eval()
    _load_ckpt(model, ckpt_path, use_ema=True)
    return cfg, frontend, model, device


def _iter_windows(audio: np.ndarray, sr: int, window_s: float, hop_s: float) -> Iterator[np.ndarray]:
    window = int(sr * window_s); hop = int(sr * hop_s)
    if audio.shape[0] < window:
        audio = np.pad(audio, (0, window - audio.shape[0]))
    for start in range(0, audio.shape[0] - window + 1, hop):
        yield audio[start : start + window]


class Detector:
    """Stateful sliding-window detector with smoothing + cooldown.

    Includes an energy gate: windows quieter than `min_dbfs` skip the model
    entirely and report p=0. This prevents per-utterance-normalized silence
    from being amplified into ambiguous features that fire the classifier.
    Every shipping wake-word stack does this — it's basically required.
    """

    def __init__(self, cfg, frontend, model, device,
                 threshold: float = 0.6, smooth_n: int = 3,
                 cooldown_s: float = 1.5, min_dbfs: float = -45.0):
        self.cfg = cfg; self.frontend = frontend; self.model = model; self.device = device
        self.threshold = threshold
        self.smooth = collections.deque(maxlen=smooth_n)
        self.last_fire_t = -1e9
        self.cooldown_s = cooldown_s
        self.min_dbfs = min_dbfs

    @torch.no_grad()
    def step(self, window_audio: np.ndarray, t_now: float) -> tuple[float, bool]:
        # Energy gate — skip silent windows.
        rms = float(np.sqrt(np.mean(window_audio ** 2) + 1e-12))
        db = 20 * np.log10(rms + 1e-9)
        if db < self.min_dbfs:
            self.smooth.append(0.0)
            sm = sum(self.smooth) / len(self.smooth)
            return sm, False

        x = torch.from_numpy(window_audio.astype(np.float32)).unsqueeze(0).to(self.device)
        mel = self.frontend(x)
        logits = _positive_logit(self.model(mel))
        prob = float(torch.sigmoid(logits).item())
        self.smooth.append(prob)
        sm = sum(self.smooth) / len(self.smooth)
        fired = sm >= self.threshold and (t_now - self.last_fire_t) >= self.cooldown_s
        if fired:
            self.last_fire_t = t_now
        return sm, fired


def stream_wav(cfg_path: str, ckpt_path: str, wav_path: str, threshold: float = 0.6,
               callback: Callable[[float, float], None] | None = None) -> list[float]:
    cfg, frontend, model, device = _load_model(cfg_path, ckpt_path)
    sr = cfg.audio.sample_rate
    audio, file_sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    det = Detector(cfg, frontend, model, device, threshold=threshold)
    fires: list[float] = []
    for i, win in enumerate(_iter_windows(audio, sr, cfg.audio.window_seconds, cfg.audio.hop_seconds)):
        t = i * cfg.audio.hop_seconds
        p, fired = det.step(win, t)
        if callback:
            callback(t, p)
        if fired:
            fires.append(t)
            print(f"  [WAKE]  t={t:6.2f}s   p={p:.3f}")
    print(f"[stream] {len(fires)} detections in {len(audio)/sr:.1f}s of audio")
    return fires


def stream_mic(cfg_path: str, ckpt_path: str, threshold: float = 0.6,
               device_id: int | None = None, verbose: bool = False,
               min_dbfs: float = -45.0, cooldown_s: float = 1.5) -> None:
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError("sounddevice not installed: pip install sounddevice") from e

    cfg, frontend, model, dev = _load_model(cfg_path, ckpt_path)
    sr = cfg.audio.sample_rate
    window = int(sr * cfg.audio.window_seconds)
    hop = int(sr * cfg.audio.hop_seconds)

    det = Detector(cfg, frontend, model, dev, threshold=threshold,
                   min_dbfs=min_dbfs, cooldown_s=cooldown_s)
    ring = np.zeros(window, dtype=np.float32)
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print(status)
        q.put(indata.copy().reshape(-1))

    print(f"[stream] listening at {sr} Hz, threshold={threshold}. Ctrl-C to stop.")
    last_print = 0.0
    with sd.InputStream(channels=1, samplerate=sr, blocksize=hop, dtype="float32", callback=cb, device=device_id):
        try:
            t0 = time.time()
            while True:
                chunk = q.get()
                if chunk.shape[0] >= window:
                    chunk = chunk[-window:]
                    ring[:] = chunk
                else:
                    ring = np.concatenate([ring[chunk.shape[0]:], chunk])
                t_now = time.time() - t0
                p, fired = det.step(ring, t_now)
                if verbose and t_now - last_print >= 0.2:
                    # mic level (RMS dBFS) + smoothed wake probability
                    rms = float(np.sqrt(np.mean(ring ** 2) + 1e-12))
                    db  = 20 * np.log10(rms + 1e-9)
                    n = int(min(40, max(0, p * 40)))
                    bar = "#" * n + "-" * (40 - n)
                    print(f"\rt={t_now:6.1f}s  mic={db:+6.1f}dBFS  p={p:.3f} |{bar}|", end="", flush=True)
                    last_print = t_now
                if fired:
                    print(f"\n  [WAKE]  t={t_now:.2f}s  p={p:.3f}")
        except KeyboardInterrupt:
            print("\n[stream] stopped.")
