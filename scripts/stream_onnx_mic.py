"""Live mic test using the ONNX file that ships in the APK.

Same model bytes, same streaming logic (energy gate + smoothing + cooldown)
as android-demo/app/src/main/kotlin/ai/bubba/wake/WakeWordDetector.kt.
Run this on macOS to validate the model before sideloading; if it works here
and not on the phone, the issue is the audio path on Android.

Usage:
    python scripts/stream_onnx_mic.py \\
        --onnx exports/wakeword_audio.int8.onnx \\
        --threshold 0.35
"""
from __future__ import annotations

import argparse
import collections
import queue
import sys
import time

import numpy as np
import onnxruntime as ort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="exports/wakeword_audio.int8.onnx",
                    help="ONNX model file (audio-in variant).")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--window-s", type=float, default=1.2)
    ap.add_argument("--hop-s", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.35)
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

    print(f"[onnx] loading {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    sr = args.sample_rate
    window = int(sr * args.window_s)
    hop = int(sr * args.hop_s)
    print(f"[onnx] sr={sr}  window={window}  hop={hop}  thr={args.threshold}  minDBFS={args.min_dbfs}")

    ring = np.zeros(window, dtype=np.float32)
    smooth = collections.deque(maxlen=args.smooth_n)
    last_fire = -1e9
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status: print(status, file=sys.stderr)
        q.put(indata.copy().reshape(-1))

    last_print = 0.0
    with sd.InputStream(channels=1, samplerate=sr, blocksize=hop,
                        dtype="float32", callback=cb, device=args.device):
        print(f"[onnx] listening. Ctrl-C to stop.")
        try:
            t0 = time.time()
            while True:
                chunk = q.get()
                # Slide ring buffer
                ring = np.concatenate([ring[chunk.shape[0]:], chunk])[-window:]

                # Energy gate (RMS dBFS)
                rms = float(np.sqrt(np.mean(ring ** 2) + 1e-12))
                db = 20.0 * np.log10(rms + 1e-9)
                t_now = time.time() - t0

                if db < args.min_dbfs:
                    smooth.append(0.0)
                    raw = 0.0
                else:
                    out = sess.run([output_name], {input_name: ring[None].astype(np.float32)})[0][0]
                    raw = 1.0 / (1.0 + np.exp(-(out[1] - out[0])))
                    smooth.append(float(raw))
                sm = sum(smooth) / len(smooth)

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
            print("\n[onnx] stopped.")


if __name__ == "__main__":
    main()
