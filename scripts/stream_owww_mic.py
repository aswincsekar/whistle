"""Live mic test for the v4 (openWakeWord-chain) model.

Runs the 3 ONNX models the APK uses, in series, on the Mac mic.
Same chain as android-demo/.../OWWWakeWordDetector.kt — useful for
parity testing before sideloading.

Usage:
    python scripts/stream_owww_mic.py \\
        [--export-dir exports/owww] \\
        [--threshold 0.45]
"""
from __future__ import annotations

import argparse
import collections
import queue
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


# Match training-time front-end constants.
SR = 16_000
WINDOW_SAMPLES = 31_360       # 1.96 s
HOP_SAMPLES = 1_600           # 100 ms
EMB_WIN_FRAMES = 76           # mel frames per embedding window
EMB_STRIDE = 8                # mel frames between consecutive embeddings
N_EMBEDDINGS = 15
MEL_TARGET_T = EMB_WIN_FRAMES + (N_EMBEDDINGS - 1) * EMB_STRIDE   # 196
MEL_BINS = 32
EMB_DIM = 96


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="exports/owww",
                    help="Directory with melspectrogram.onnx + embedding_model.onnx + classifier(.int8).onnx")
    ap.add_argument("--classifier", default=None,
                    help="Classifier file inside --export-dir (default: classifier.int8.onnx, falls back to .onnx).")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-dbfs", type=float, default=-45.0)
    ap.add_argument("--cooldown-s", type=float, default=1.5)
    ap.add_argument("--smooth-n", type=int, default=3)
    ap.add_argument("--device", type=int, default=None,
                    help="sounddevice input device id (omit for default).")
    args = ap.parse_args()

    try:
        import sounddevice as sd
    except Exception as e:
        print(f"sounddevice missing: {e}", file=sys.stderr); sys.exit(2)

    base = Path(args.export_dir)
    mel_path = base / "melspectrogram.onnx"
    emb_path = base / "embedding_model.onnx"
    cls_name = args.classifier or ("classifier.int8.onnx" if (base / "classifier.int8.onnx").exists() else "classifier.onnx")
    cls_path = base / cls_name

    for p in (mel_path, emb_path, cls_path):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr); sys.exit(2)

    print(f"[owww] mel  : {mel_path}")
    print(f"[owww] emb  : {emb_path}")
    print(f"[owww] cls  : {cls_path}")

    opts = ort.SessionOptions(); opts.intra_op_num_threads = 2
    mel_sess = ort.InferenceSession(str(mel_path), opts, providers=["CPUExecutionProvider"])
    emb_sess = ort.InferenceSession(str(emb_path), opts, providers=["CPUExecutionProvider"])
    cls_sess = ort.InferenceSession(str(cls_path), opts, providers=["CPUExecutionProvider"])
    mel_in = mel_sess.get_inputs()[0].name
    emb_in = emb_sess.get_inputs()[0].name
    cls_in = cls_sess.get_inputs()[0].name

    ring = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    smooth = collections.deque(maxlen=args.smooth_n)
    last_fire = -1e9
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status: print(status, file=sys.stderr)
        q.put(indata.copy().reshape(-1))

    def step(window: np.ndarray) -> tuple[float, float, float]:
        rms = float(np.sqrt(np.mean(window ** 2) + 1e-12))
        db = 20.0 * np.log10(rms + 1e-9)
        if db < args.min_dbfs:
            return 0.0, 0.0, db

        # 1) mel
        mel = mel_sess.run(None, {mel_in: window[None]})[0]  # (1, 1, T, 32)
        mel = mel[0, 0]                                      # (T, 32)
        if mel.shape[0] > MEL_TARGET_T:
            mel = mel[:MEL_TARGET_T]
        elif mel.shape[0] < MEL_TARGET_T:
            pad = MEL_TARGET_T - mel.shape[0]
            mel = np.concatenate([mel, np.repeat(mel[-1:], pad, axis=0)], axis=0)

        # 2) 15 windows
        chunks = np.stack([
            mel[i * EMB_STRIDE : i * EMB_STRIDE + EMB_WIN_FRAMES] for i in range(N_EMBEDDINGS)
        ])[..., None].astype(np.float32)                      # (15, 76, 32, 1)
        emb = emb_sess.run(None, {emb_in: chunks})[0]         # (15, 1, 1, 96)
        emb = emb.reshape(1, N_EMBEDDINGS, EMB_DIM).astype(np.float32)

        # 3) classifier
        logits = cls_sess.run(None, {cls_in: emb})[0][0]
        raw = 1.0 / (1.0 + np.exp(-(logits[1] - logits[0])))
        return float(raw), float(raw), db

    last_print = 0.0
    with sd.InputStream(channels=1, samplerate=SR, blocksize=HOP_SAMPLES,
                        dtype="float32", callback=cb, device=args.device):
        print(f"[owww] listening. threshold={args.threshold}. Ctrl-C to stop.")
        try:
            t0 = time.time()
            while True:
                chunk = q.get()
                ring = np.concatenate([ring[chunk.shape[0]:], chunk])[-WINDOW_SAMPLES:]

                raw, _raw2, db = step(ring)
                smooth.append(raw)
                sm = sum(smooth) / len(smooth)
                t_now = time.time() - t0

                fired = sm >= args.threshold and (t_now - last_fire) >= args.cooldown_s
                if fired:
                    last_fire = t_now
                    print(f"\n  [WAKE]  t={t_now:.2f}s  smoothed={sm:.3f}  raw={raw:.3f}")

                if t_now - last_print >= 0.2:
                    n = int(min(40, max(0, sm * 40)))
                    bar = "#" * n + "-" * (40 - n)
                    print(f"\rt={t_now:6.1f}s  mic={db:+6.1f}dBFS  sm={sm:.3f}  raw={raw:.3f} |{bar}|",
                          end="", flush=True)
                    last_print = t_now
        except KeyboardInterrupt:
            print("\n[owww] stopped.")


if __name__ == "__main__":
    main()
