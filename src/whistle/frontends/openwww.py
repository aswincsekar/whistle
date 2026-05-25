"""openWakeWord feature pipeline as a drop-in front-end.

Two ONNX models from openWakeWord v0.5.1:
  * melspectrogram.onnx — raw 16 kHz audio -> (B, 1, T, 32) log-mel features
  * embedding_model.onnx — (N, 76, 32, 1) mel-window batch -> (N, 1, 1, 96)

The embedding model was originally Google's `speech_embedding` model, trained
self-supervised on a huge corpus and supervised on phoneme / speaker tasks.
Its 96-dim output is much more invariant to speaker / loudness / mic than
raw mel-spectrograms — which is why openWakeWord's tiny classifiers
generalize way better than our from-scratch BC-ResNet did.

Window shapes:
  * input audio  : ~1.96 s (31360 samples)
  * mel frames   : ~196 frames (10 ms hop)
  * sliding embedding windows : 76 frames each, stride 8 -> 15 embeddings
  * classifier input shape    : (B, 15, 96)

We default to running BOTH ONNX models via ONNX Runtime CPU sessions in the
dataloader workers; the GPU sees only the small classifier. This pairs well
with our existing PyTorch augmentation pipeline (which operates on raw
audio): augment -> melspec -> embedding -> classifier.

For mobile deployment we ship all three ONNX files and chain them at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Default IO shapes — chosen to match openWakeWord's own pipeline exactly so the
# pretrained embedding gets features identical to what it saw at training time.
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 31_360         # 1.96 s @ 16 kHz
MEL_HOP_SAMPLES = 160           # 10 ms hop
EMB_INPUT_FRAMES = 76           # 76 mel frames per embedding window
EMB_STRIDE_FRAMES = 8           # slide by 80 ms between embeddings
N_EMBEDDINGS = 15               # how many embeddings come out per audio window
EMB_DIM = 96


@dataclass
class OWWPaths:
    melspec_onnx: str
    embedding_onnx: str

    @classmethod
    def default(cls) -> "OWWPaths":
        # Look for the files next to the repo root by default.
        from ..config import repo_root
        base = repo_root() / "models" / "owww"
        return cls(
            melspec_onnx=str(base / "melspectrogram.onnx"),
            embedding_onnx=str(base / "embedding_model.onnx"),
        )


class OWWFeatures:
    """CPU-resident melspec + embedding pipeline using ONNX Runtime.

    Designed to be created once per dataloader worker. Stateless after init
    (no caches), so reading from multiple threads is safe.
    """

    def __init__(self, paths: Optional[OWWPaths] = None,
                 num_threads: int = 1):
        import onnxruntime as ort
        paths = paths or OWWPaths.default()
        if not Path(paths.melspec_onnx).exists():
            raise FileNotFoundError(f"missing {paths.melspec_onnx} — run "
                                    "`make download-owww` or copy from openWakeWord v0.5.1 release")
        if not Path(paths.embedding_onnx).exists():
            raise FileNotFoundError(f"missing {paths.embedding_onnx}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        self.mel = ort.InferenceSession(paths.melspec_onnx, opts, providers=["CPUExecutionProvider"])
        self.emb = ort.InferenceSession(paths.embedding_onnx, opts, providers=["CPUExecutionProvider"])
        self.mel_in = self.mel.get_inputs()[0].name
        self.emb_in = self.emb.get_inputs()[0].name

    def melspec(self, audio: np.ndarray) -> np.ndarray:
        """audio (T,) or (1, T) float32 16 kHz PCM -> (T_frames, 32) mel features."""
        if audio.ndim == 1:
            audio = audio[None]
        mel = self.mel.run(None, {self.mel_in: audio.astype(np.float32)})[0]
        # (1, 1, T, 32) -> (T, 32)
        return mel[0, 0]

    def mel_for_window(self, audio: np.ndarray) -> np.ndarray:
        """audio -> mel features of fixed shape, ready for GPU-side embedding."""
        if audio.shape[-1] != WINDOW_SAMPLES:
            audio = _pad_or_crop(audio, WINDOW_SAMPLES)
        mel = self.melspec(audio)                       # (T, 32)
        # Some ONNX builds emit T = 196, some 193. Pad / crop to a stable 196 so
        # downstream sliding-window math always produces exactly N_EMBEDDINGS.
        target_T = EMB_INPUT_FRAMES + (N_EMBEDDINGS - 1) * EMB_STRIDE_FRAMES  # = 196
        if mel.shape[0] > target_T:
            mel = mel[:target_T]
        elif mel.shape[0] < target_T:
            pad = target_T - mel.shape[0]
            mel = np.concatenate([mel, np.repeat(mel[-1:], pad, axis=0)], axis=0)
        return mel.astype(np.float32)                   # (196, 32)

    def features(self, audio: np.ndarray) -> np.ndarray:
        """Full pipeline: audio -> (N, 96) embedding sequence.

        Runs both melspec and embedding via ORT (CPU). Use `mel_for_window`
        plus a GPU-resident embedding module for faster training.
        """
        mel = self.mel_for_window(audio)
        chunks = _sliding_windows(mel, EMB_INPUT_FRAMES, EMB_STRIDE_FRAMES)
        if chunks.shape[0] != N_EMBEDDINGS:
            # Should never happen with the fixed-target mel, but guard anyway.
            chunks = chunks[:N_EMBEDDINGS] if chunks.shape[0] > N_EMBEDDINGS else \
                np.concatenate([chunks, np.repeat(chunks[-1:], N_EMBEDDINGS - chunks.shape[0], axis=0)])
        chunks = chunks[..., None].astype(np.float32)   # (N, 76, 32, 1)
        emb = self.emb.run(None, {self.emb_in: chunks})[0]
        return emb.reshape(N_EMBEDDINGS, EMB_DIM)


def _pad_or_crop(x: np.ndarray, target: int) -> np.ndarray:
    if x.shape[-1] == target:
        return x
    if x.shape[-1] < target:
        pad = target - x.shape[-1]
        left = pad // 2
        right = pad - left
        return np.pad(x, (left, right) if x.ndim == 1 else ((0, 0), (left, right)))
    # crop centered
    excess = x.shape[-1] - target
    start = excess // 2
    return x[..., start : start + target]


class OWWEmbeddingTorch:
    """Lazy GPU-resident wrapper around the embedding ONNX (via onnx2torch).

    Takes mel features of shape (B, 196, 32) and returns (B, N=15, 96)
    embeddings. The embedding model is loaded once and frozen — only the
    classifier trains.
    """

    _module = None  # cached across calls

    @classmethod
    def get(cls, device, paths: Optional[OWWPaths] = None):
        import torch
        if cls._module is not None and cls._module.device == device:
            return cls._module
        from onnx2torch import convert
        paths = paths or OWWPaths.default()
        m = convert(paths.embedding_onnx).eval().to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        wrapper = _OWWTorchWrap(m, device)
        cls._module = wrapper
        return wrapper


class _OWWTorchWrap:
    def __init__(self, emb_module, device):
        self.emb = emb_module
        self.device = device

    def __call__(self, mel):
        """mel: (B, 196, 32) on `device` -> (B, 15, 96)"""
        import torch
        B, T, F = mel.shape
        # Slide windows: (B, N, 76, 32)
        windows = mel.unfold(1, EMB_INPUT_FRAMES, EMB_STRIDE_FRAMES)
        # unfold returns (B, N, 32, 76); transpose to (B, N, 76, 32)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        N = windows.shape[1]
        # Flatten to (B*N, 76, 32, 1) for the embedding model
        x = windows.reshape(B * N, EMB_INPUT_FRAMES, F, 1)
        out = self.emb(x)                              # (B*N, 1, 1, 96)
        return out.reshape(B, N, EMB_DIM)


def _sliding_windows(mel: np.ndarray, win: int, hop: int) -> np.ndarray:
    """mel (T, 32) -> (N, win, 32) with stride hop."""
    T = mel.shape[0]
    n = (T - win) // hop + 1
    if n <= 0:
        return np.zeros((0, win, mel.shape[1]), dtype=mel.dtype)
    # Use stride tricks for zero-copy.
    sh = (n, win, mel.shape[1])
    st = (mel.strides[0] * hop, mel.strides[0], mel.strides[1])
    return np.lib.stride_tricks.as_strided(mel, shape=sh, strides=st).copy()
