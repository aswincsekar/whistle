"""Mic test for the openWakeWord-trained 'Hey Bubba' model.

Uses openWakeWord's actual streaming detector (it handles ring buffer,
8-feature internal state, and threshold the same way the stock 'hey jarvis'
model does). This is the fair comparison.

Usage:
    python scripts/stream_owww_trained.py [--model models/baseline-v5/hey_bubba_v0.1.onnx]
"""
from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/hey_bubba_owww_trained.onnx",
                    help="Path to the OWW-trained ONNX classifier file.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cooldown-s", type=float, default=1.5)
    args = ap.parse_args()

    try:
        from openwakeword import Model as OWWModel
        import sounddevice as sd
    except Exception as e:
        print(f"missing dep: {e}", file=sys.stderr); sys.exit(2)

    p = Path(args.model)
    if not p.exists():
        print(f"missing model: {p}", file=sys.stderr); sys.exit(2)

    # The model's keyword key is derived from the filename (basename without .onnx)
    kw = p.stem
    print(f"[trained] loading custom model '{kw}' from {p}")
    model = OWWModel(wakeword_models=[str(p)], inference_framework="onnx")

    sr = 16_000
    chunk_samples = 1_280
    q: queue.Queue = queue.Queue()
    def cb(indata, frames, time_info, status):
        if status: print(status, file=sys.stderr)
        x = (indata.reshape(-1) * 32767).clip(-32768, 32767).astype(np.int16)
        q.put(x)

    last_fire = -1e9
    last_print = 0.0
    with sd.InputStream(channels=1, samplerate=sr, blocksize=chunk_samples,
                        dtype="float32", callback=cb):
        print(f"[trained] listening for 'Hey Bubba' / 'Heybubba' (threshold={args.threshold}). Ctrl-C to stop.")
        try:
            t0 = time.time()
            while True:
                chunk = q.get()
                preds = model.predict(chunk)
                # The model produces one score per registered model
                score = float(next(iter(preds.values())))
                rms = float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2) + 1e-12))
                db = 20.0 * np.log10(rms + 1e-9)
                t_now = time.time() - t0

                if score >= args.threshold and (t_now - last_fire) >= args.cooldown_s:
                    last_fire = t_now
                    print(f"\n  [WAKE]  t={t_now:.2f}s  score={score:.3f}  mic={db:+.1f}dBFS")
                if t_now - last_print >= 0.2:
                    n = int(min(40, max(0, score * 40)))
                    bar = "#" * n + "-" * (40 - n)
                    print(f"\rt={t_now:6.1f}s  mic={db:+6.1f}dBFS  score={score:.3f} |{bar}|", end="", flush=True)
                    last_print = t_now
        except KeyboardInterrupt:
            print("\n[trained] stopped.")


if __name__ == "__main__":
    main()
