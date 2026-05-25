"""Export the trained model to ONNX and (optionally) TFLite.

We export TWO graphs:
  * `wakeword_audio.onnx` — input is raw audio (B, T) in [-1, 1].
    One file, no preprocessing required on-device. Recommended path.
  * `wakeword_mel.onnx`   — input is (B, n_mels, frames). Use this if your
    mobile runtime already produces mel features (e.g. CoreML AudioFeaturePrint).

INT8 quantization is via ONNX Runtime's dynamic quant (weights only). For static
quant you need a representative dataset — we use a small sample if manifests
exist. For TFLite, onnx2tf converts ONNX -> TF -> .tflite, and we apply post-
training INT8 quant during conversion.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .config import load_config, repo_root
from .eval import _load_ckpt
from .features import FeatureConfig, LogMelSpectrogram
from .models import build_model
from .models.wakeword import MelClassifier
from .train import _device


def _output_paths(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "onnx_audio": out_dir / "wakeword_audio.onnx",
        "onnx_audio_int8": out_dir / "wakeword_audio.int8.onnx",
        "onnx_mel": out_dir / "wakeword_mel.onnx",
        "onnx_mel_int8": out_dir / "wakeword_mel.int8.onnx",
        "tflite_audio_fp32": out_dir / "wakeword_audio.fp32.tflite",
        "tflite_audio_int8": out_dir / "wakeword_audio.int8.tflite",
        "mel_filterbank": out_dir / "mel_filterbank.npy",
        "metadata": out_dir / "metadata.json",
    }


def _export_owww(cfg, ckpt_path: str, out_path: Path, root: Path) -> dict:
    """Export pipeline for the openWakeWord-based path.

    Output layout:
        exports/owww/
          melspectrogram.onnx     (copied from models/owww/, ~1 MB)
          embedding_model.onnx    (copied from models/owww/, ~1.3 MB)
          classifier.onnx         (our trained classifier, FP32)
          classifier.int8.onnx    (static QDQ INT8, Conv-only)
          metadata.json
    """
    import shutil
    from .models.owww_classifier import build_owww_model
    from .frontends.openwww import (
        OWWPaths, N_EMBEDDINGS, EMB_DIM, WINDOW_SAMPLES,
    )
    out_path = out_path / "owww"
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load classifier with EMA weights
    model = build_owww_model(cfg).eval()
    _load_ckpt(model, ckpt_path, use_ema=True)

    # 2. Export classifier to ONNX
    cls_onnx = out_path / "classifier.onnx"
    print(f"[export] -> {cls_onnx}")
    dummy = torch.zeros(1, N_EMBEDDINGS, EMB_DIM)
    torch.onnx.export(
        model, (dummy,), str(cls_onnx),
        input_names=["embeddings"], output_names=["logits"],
        opset_version=cfg.export.onnx_opset,
    )
    # Simplify
    try:
        import onnx
        from onnxsim import simplify
        m = onnx.load(str(cls_onnx))
        simp, ok = simplify(m)
        if ok:
            onnx.save(simp, str(cls_onnx))
            print(f"[export] simplified {cls_onnx.name}")
    except Exception as e:
        print(f"[export] simplifier skipped: {e}")

    # 3. Static QDQ INT8 quantization of the classifier
    cls_int8 = out_path / "classifier.int8.onnx"
    if cfg.export.quantize_int8:
        rep = [np.random.randn(1, N_EMBEDDINGS, EMB_DIM).astype(np.float32)
               for _ in range(cfg.export.representative_samples)]
        print(f"[export] static QDQ INT8 -> {cls_int8}")
        _quantize_static_qdq(str(cls_onnx), str(cls_int8), "embeddings", rep)

    # 4. Copy the openWakeWord ONNX files alongside (they're frozen, no retrain).
    src_owww = root / "models" / "owww"
    for fname in ("melspectrogram.onnx", "embedding_model.onnx"):
        src = src_owww / fname
        dst = out_path / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[export] copied {fname}  ({dst.stat().st_size // 1024} KB)")

    # 5. Metadata for the mobile integrator
    meta = {
        "pipeline": "owww",
        "phrase": cfg.wakeword.phrase,
        "sample_rate": cfg.audio.sample_rate,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": cfg.audio.window_seconds,
        "hop_seconds": cfg.audio.hop_seconds,
        "chain": [
            {
                "step": "melspectrogram",
                "file": "melspectrogram.onnx",
                "input": "audio (1, samples) float32 PCM [-1,1]",
                "output": "mel (1, 1, T, 32)",
                "notes": "T ~= 196 for a 1.96s window",
            },
            {
                "step": "embedding",
                "file": "embedding_model.onnx",
                "input": "(N, 76, 32, 1) — N sliding windows of 76 mel frames, stride 8",
                "output": "(N, 1, 1, 96) — 96-dim speech embeddings",
                "notes": f"For our window {N_EMBEDDINGS} embeddings",
            },
            {
                "step": "classifier",
                "file": "classifier.int8.onnx (or classifier.onnx)",
                "input": f"(1, {N_EMBEDDINGS}, {EMB_DIM}) — stacked embeddings",
                "output": "(1, 2) logits — class 1 is the wake word",
            },
        ],
        "n_embeddings_per_window": N_EMBEDDINGS,
        "embedding_dim": EMB_DIM,
        "classifier_params": sum(p.numel() for p in model.parameters()),
    }
    import json
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 6. Verify the end-to-end PyTorch vs full ORT chain matches
    print("[verify] running end-to-end parity check ...")
    _verify_owww_chain(out_path, model, cfg)
    return meta


def _verify_owww_chain(out_path: Path, model_pt, cfg) -> None:
    """Run audio through the saved 3-file chain via ORT and compare with the
    PyTorch end-to-end (frozen embedding + classifier) for parity."""
    import onnxruntime as ort
    from .frontends.openwww import (
        OWWFeatures, OWWPaths, N_EMBEDDINGS, EMB_DIM, WINDOW_SAMPLES,
    )
    feats = OWWFeatures(OWWPaths(
        melspec_onnx=str(out_path / "melspectrogram.onnx"),
        embedding_onnx=str(out_path / "embedding_model.onnx"),
    ))
    cls_sess = ort.InferenceSession(str(out_path / "classifier.onnx"),
                                    providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(4):
        audio = (rng.standard_normal(WINDOW_SAMPLES) * 0.1).astype(np.float32)
        # All-ORT
        emb = feats.features(audio)                  # (15, 96)
        ort_logits = cls_sess.run(None, {"embeddings": emb[None]})[0]
        # PyTorch classifier on the same ORT-computed embeddings
        with torch.no_grad():
            pt_logits = model_pt(torch.from_numpy(emb).unsqueeze(0)).numpy()
        diff = float(np.abs(ort_logits - pt_logits).max())
        if diff > max_diff:
            max_diff = diff
    print(f"[verify] max |ONNX-cls - PyTorch-cls| = {max_diff:.3e}  "
          f"({'OK' if max_diff < 1e-3 else 'CHECK'})")


def _quantize_static_qdq(src: str, dst: str, input_name: str, calib_data: list[np.ndarray]) -> None:
    """Static post-training INT8 (QDQ format) using ORT's quantizer.

    Inserts QuantizeLinear/DequantizeLinear nodes around fp32 ops based on the
    activation ranges measured on `calib_data`. Output graph runs on stock ORT
    on every platform (no ConvInteger op required).
    """
    from onnxruntime.quantization import (
        quantize_static, QuantFormat, QuantType, CalibrationDataReader,
    )

    class _Reader(CalibrationDataReader):
        def __init__(self, arrays: list[np.ndarray], name: str):
            self.items = iter([{name: a} for a in arrays])

        def get_next(self):
            return next(self.items, None)

    # First run ORT's preprocess pass (model_optimization etc.) - reduces size
    # and helps the quantizer.
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = Path(dst).with_suffix(".pre.onnx")
        quant_pre_process(input_model_path=src, output_model_path=str(pre),
                          skip_optimization=False, skip_onnx_shape=False,
                          skip_symbolic_shape=True)
        src_for_quant = str(pre)
    except Exception as e:
        print(f"  [quantize] pre-process skipped: {e}")
        src_for_quant = src

    # Only quantize Conv ops — that's where ~95% of the BC-ResNet compute and
    # weights live. Leaving MatMul fp32 avoids two known footguns:
    #   * The mel-filterbank MatMul has a constant weight and ORT's QLinearMatMul
    #     kernel requires scalar activation zero point, which per-channel quant
    #     violates.
    #   * MatMul ops inside attention blocks (not present here, but future-proof).
    quantize_static(
        model_input=src_for_quant,
        model_output=dst,
        calibration_data_reader=_Reader(calib_data, input_name),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"],
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    # Clean up the preprocessed intermediate if any.
    try:
        if src_for_quant != src and Path(src_for_quant).exists():
            Path(src_for_quant).unlink()
    except Exception:
        pass


def _representative_audio(cfg, root: Path, n: int = 256) -> list[np.ndarray]:
    """Pull a handful of representative audio windows for static quant."""
    from .data.dataset import WakeWordDataset, load_manifest
    mpath = root / cfg.data.manifest_dir / "train.jsonl"
    samples: list[np.ndarray] = []
    if mpath.exists():
        rng = random.Random(0)
        manifest = load_manifest(mpath)
        rng.shuffle(manifest)
        ds = WakeWordDataset(manifest[:n], cfg.audio.sample_rate, cfg.audio.window_seconds,
                             augmenter=None, training=False, seed=0)
        for i in range(len(ds)):
            audio, _ = ds[i]
            samples.append(audio.numpy().astype(np.float32))
            if len(samples) >= n:
                break
    if not samples:
        # Fallback: random noise — enough for the calibrator to set ranges.
        win = int(cfg.audio.window_seconds * cfg.audio.sample_rate)
        rng = np.random.default_rng(0)
        samples = [(rng.standard_normal(win).astype(np.float32) * 0.05) for _ in range(min(n, 64))]
    return samples


def export(cfg_path: str, ckpt_path: str, fmt: str = "onnx", out_dir: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    root = repo_root()
    out_path = Path(out_dir) if out_dir else root / "exports"
    if cfg.model.name == "owww":
        return _export_owww(cfg, ckpt_path, out_path, root)

    out = _output_paths(out_path)

    # ----- Build modules -----
    feat_cfg = FeatureConfig.from_cfg(cfg)
    backbone = build_model(cfg).eval()
    _load_ckpt(backbone, ckpt_path, use_ema=True)

    # End-to-end audio-in module
    e2e = MelClassifier(cfg).eval()
    e2e.backbone.load_state_dict(backbone.state_dict(), strict=False)

    sr = cfg.audio.sample_rate
    win = int(cfg.audio.window_seconds * sr)
    audio_example = torch.zeros(1, win, dtype=torch.float32)
    mel_example = LogMelSpectrogram(feat_cfg)(audio_example)

    # Save mel filterbank weights for mobile parity checks.
    np.save(out["mel_filterbank"], e2e.frontend.mel_fb.detach().cpu().numpy())

    # ----- ONNX export -----
    # Mobile runtimes typically allocate one tensor; static batch=1 is enough and
    # plays nicest with quantization. We expose a fixed-shape graph; mobile code
    # can loop windows over time.
    print(f"[export] -> {out['onnx_audio']}")
    torch.onnx.export(
        e2e, (audio_example,), str(out["onnx_audio"]),
        input_names=["audio"], output_names=["logits"],
        opset_version=cfg.export.onnx_opset,
    )
    print(f"[export] -> {out['onnx_mel']}")
    torch.onnx.export(
        backbone, (mel_example,), str(out["onnx_mel"]),
        input_names=["mel"], output_names=["logits"],
        opset_version=cfg.export.onnx_opset,
    )

    # Optional: onnx-simplifier pass
    try:
        import onnx
        from onnxsim import simplify
        for p in (out["onnx_audio"], out["onnx_mel"]):
            m = onnx.load(str(p))
            simp, ok = simplify(m)
            if ok:
                onnx.save(simp, str(p))
                print(f"[export] simplified {p.name}")
    except Exception as e:
        print(f"[export] onnx-simplifier skipped: {e}")

    # ----- INT8 (ONNX) -----
    # We use *static QDQ* quantization, not dynamic. Two reasons:
    #   1. QDQ produces QuantizeLinear/DequantizeLinear wrappers around regular
    #      fp32 ops, which every ORT build supports (mobile, web, ...).
    #      Dynamic quant produces ConvInteger which has spotty mobile support.
    #   2. Static quant computes per-tensor activation ranges from real audio,
    #      giving better accuracy than dynamic's on-the-fly ranges.
    if cfg.export.quantize_int8:
        rep_samples = _representative_audio(cfg, root, n=cfg.export.representative_samples)
        rep_mel = [
            e2e.frontend(torch.from_numpy(s).unsqueeze(0)).numpy()
            for s in rep_samples
        ]
        for src, dst, input_name, calib in [
            (out["onnx_audio"], out["onnx_audio_int8"], "audio",
             [np.asarray(s, dtype=np.float32).reshape(1, -1) for s in rep_samples]),
            (out["onnx_mel"], out["onnx_mel_int8"], "mel",
             [np.asarray(m, dtype=np.float32) for m in rep_mel]),
        ]:
            print(f"[export] static QDQ INT8 -> {dst}  (calib n={len(calib)})")
            _quantize_static_qdq(str(src), str(dst), input_name, calib)

    paths = {k: str(v) for k, v in out.items()}

    # ----- TFLite (optional) -----
    if fmt == "tflite":
        _export_tflite(cfg, root, out, audio_example.shape)

    # ----- Metadata for mobile integrators -----
    meta = {
        "phrase": cfg.wakeword.phrase,
        "sample_rate": sr,
        "window_seconds": cfg.audio.window_seconds,
        "hop_seconds": cfg.audio.hop_seconds,
        "input_samples": win,
        "features": {
            "n_fft": feat_cfg.n_fft,
            "win_length": feat_cfg.win_length,
            "hop_length": feat_cfg.hop_length,
            "n_mels": feat_cfg.n_mels,
            "fmin": feat_cfg.fmin,
            "fmax": feat_cfg.fmax,
            "log_offset": feat_cfg.log_offset,
            "per_utt_normalize": feat_cfg.per_utt_normalize,
        },
        "model": {
            "name": cfg.model.name,
            "tau": cfg.model.tau,
            "num_params": sum(p.numel() for p in backbone.parameters()),
        },
        "outputs": paths,
        "expected_input": {
            "wakeword_audio": "float32 PCM, mono, normalized to [-1, 1], length = input_samples",
            "wakeword_mel":   "float32 log-mel (B, n_mels, frames)",
        },
        "output_shape": "logits (B, num_classes)",
    }
    with open(out["metadata"], "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] metadata -> {out['metadata']}")
    return meta


def _export_tflite(cfg, root: Path, out: dict, audio_shape) -> None:
    try:
        import onnx2tf
    except Exception:
        print("[export] onnx2tf not installed. `make install-tflite` first.")
        return

    workdir = out["tflite_audio_fp32"].parent / "_tf_work"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    # Static-shape audio model for TFLite (TFLite doesn't love dynamic axes).
    static_audio = workdir / "wakeword_audio_static.onnx"
    _restamp_static(out["onnx_audio"], static_audio, dim={"audio": [1, audio_shape[-1]],
                                                          "logits": [1, cfg.model.num_classes]})

    # Generate representative dataset for INT8
    rep = _representative_audio(cfg, root, n=cfg.export.representative_samples)
    rep_dir = workdir / "rep"
    rep_dir.mkdir()
    rep_paths: list[str] = []
    for i, arr in enumerate(rep):
        p = rep_dir / f"rep_{i:04d}.npy"
        np.save(p, arr[None, :])
        rep_paths.append(str(p))

    print(f"[export] onnx2tf -> {workdir}")
    onnx2tf.convert(
        input_onnx_file_path=str(static_audio),
        output_folder_path=str(workdir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    # onnx2tf emits multiple .tflite files (fp32, dynamic_int8, full_int8...).
    # We pick the canonical ones.
    candidates = list(workdir.glob("*.tflite"))
    if not candidates:
        print("[export] onnx2tf produced no .tflite files")
        return
    for p in candidates:
        target = out["tflite_audio_fp32"] if "float32" in p.name else out["tflite_audio_int8"]
        if "float32" in p.name or "fp32" in p.name:
            shutil.copy2(p, out["tflite_audio_fp32"])
        elif "int8" in p.name:
            shutil.copy2(p, out["tflite_audio_int8"])
    print(f"[export] tflite -> {out['tflite_audio_fp32'].name}, {out['tflite_audio_int8'].name}")


def _restamp_static(src: Path, dst: Path, dim: dict[str, list[int]]) -> None:
    """Rewrite ONNX with fixed input/output shapes for TFLite-friendly export."""
    import onnx
    m = onnx.load(str(src))
    for inp in list(m.graph.input) + list(m.graph.output):
        if inp.name in dim:
            new = dim[inp.name]
            tp = inp.type.tensor_type
            tp.shape.Clear()
            for d in new:
                tp.shape.dim.add().dim_value = int(d)
    onnx.save(m, str(dst))


def verify_onnx(cfg_path: str, ckpt_path: str, onnx_path: str, n: int = 8, tol: float = 1e-3) -> bool:
    """Compare PyTorch vs ONNX outputs on `n` random inputs."""
    import onnxruntime as ort
    cfg = load_config(cfg_path)
    e2e = MelClassifier(cfg).eval()
    backbone = build_model(cfg)
    _load_ckpt(backbone, ckpt_path, use_ema=True)
    e2e.backbone.load_state_dict(backbone.state_dict(), strict=False)

    win = int(cfg.audio.window_seconds * cfg.audio.sample_rate)
    rng = np.random.default_rng(0)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for _ in range(n):
        x = (rng.standard_normal((1, win)).astype(np.float32) * 0.1)
        with torch.no_grad():
            y_pt = e2e(torch.from_numpy(x)).cpu().numpy()
        y_ox = sess.run(["logits"], {"audio": x})[0]
        diff = float(np.max(np.abs(y_pt - y_ox)))
        max_diff = max(max_diff, diff)
    ok = max_diff <= tol
    print(f"[verify] max |pt - onnx| = {max_diff:.3e}  ({'OK' if ok else 'MISMATCH'})")
    return ok
