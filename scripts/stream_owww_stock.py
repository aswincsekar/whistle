"""Live mic test using openWakeWord's stock pretrained models.

This is the pipeline validation step from "Step 1". If their stock model
(e.g. hey_jarvis) behaves cleanly here — fires only on the trigger phrase,
silence stays at 0, no false fires on conversational speech — then their
training recipe is sound and our v4 issues are purely on our end. If their
stock model ALSO has trouble, the issue is environmental (mic, device, etc.)
and we'd save ourselves a multi-hour training run.

Usage:
    python scripts/stream_owww_stock.py --keyword hey_jarvis
    python scripts/stream_owww_stock.py --keyword alexa
    python scripts/stream_owww_stock.py --keyword hey_mycroft
"""
from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", default="hey_jarvis",
                    help="One of: alexa, hey_mycroft, hey_jarvis, hey_rhasspy, timer, weather")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cooldown-s", type=float, default=1.5)
    args = ap.parse_args()

    try:
        from openwakeword import Model as OWWModel
        import sounddevice as sd
    except Exception as e:
        print(f"missing dep: {e}", file=sys.stderr); sys.exit(2)

    # ONNX inference. Pass just one keyword so the score is cleanly that one.
    model = OWWModel(wakeword_models=[args.keyword], inference_framework="onnx")
    sr = 16_000
    chunk_samples = 1_280   # 80 ms, what openWakeWord ingests per call

    q: queue.Queue = queue.Queue()
    def cb(indata, frames, time_info, status):
        if status: print(status, file=sys.stderr)
        # OWW expects int16
        x = (indata.reshape(-1) * 32767).clip(-32768, 32767).astype(np.int16)
        q.put(x)

    last_fire = -1e9
    last_print = 0.0
    with sd.InputStream(channels=1, samplerate=sr, blocksize=chunk_samples,
                        dtype="float32", callback=cb):
        print(f"[owww-stock] listening for '{args.keyword}' "
              f"(threshold={args.threshold}). Ctrl-C to stop.")
        try:
            t0 = time.time()
            while True:
                chunk = q.get()
                preds = model.predict(chunk)
                score = float(preds[args.keyword])
                rms = float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2) + 1e-12))
                db = 20.0 * np.log10(rms + 1e-9)
                t_now = time.time() - t0

                if score >= args.threshold and (t_now - last_fire) >= args.cooldown_s:
                    last_fire = t_now
                    print(f"\n  [WAKE]  t={t_now:.2f}s  score={score:.3f}  mic={db:+.1f}dBFS")

                if t_now - last_print >= 0.2:
                    n = int(min(40, max(0, score * 40)))
                    bar = "#" * n + "-" * (40 - n)
                    print(f"\rt={t_now:6.1f}s  mic={db:+6.1f}dBFS  score={score:.3f} |{bar}|",
                          end="", flush=True)
                    last_print = t_now
        except KeyboardInterrupt:
            print("\n[owww-stock] stopped.")


if __name__ == "__main__":
    main()
